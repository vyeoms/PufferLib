// Removed vector dependency for MinGRU activations - now uses raw pointers

#ifndef PUFFERLIB_MODELS_CU
#define PUFFERLIB_MODELS_CU

#include <cstdint>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "kernels.cu"

// Signatures used by encoder and decoder. Writing custom nets in 4.0 requires a fair bit of code,
// because you are responsible for defining your own activation and gradient buffers.
// In practice, this is fairly simple. See our Encoder and Decoder for examples.
// You probably only ever need a custom Encoder
typedef void (*init_weights_fn)(void* weights, ulong* seed, cudaStream_t stream);
typedef void (*reg_params_fn)(void* weights, Allocator* alloc);
typedef void (*reg_train_fn)(void* weights, void* buf, Allocator* acts, Allocator* grads, int B_TT);
typedef void (*reg_rollout_fn)(void* weights, void* buf, Allocator* alloc, int B);
typedef void* (*create_weights_fn)(void* self);
typedef void  (*free_weights_fn)(void* weights);
typedef void  (*free_activations_fn)(void* activations);
typedef PrecisionTensor (*forward_fn)(void* weights, void* activations, PrecisionTensor input, cudaStream_t stream);
typedef void (*encoder_backward_fn)(void* weights, void* activations,
    PrecisionTensor grad, cudaStream_t stream);
typedef PrecisionTensor (*decoder_backward_fn)(void* weights, void* activations,
    FloatTensor grad_logits, FloatTensor grad_logstd, FloatTensor grad_value, cudaStream_t stream);
typedef PrecisionTensor (*network_forward_fn)(void* weights, PrecisionTensor x,
    PrecisionTensor state, void* activations, cudaStream_t stream);
typedef PrecisionTensor (*network_forward_train_fn)(void* weights, PrecisionTensor x,
    PrecisionTensor state, void* activations, cudaStream_t stream);
typedef PrecisionTensor (*network_backward_fn)(void* weights,
    PrecisionTensor grad, void* activations, cudaStream_t stream);

struct Encoder {
    forward_fn forward;
    encoder_backward_fn backward;
    init_weights_fn init_weights;
    reg_params_fn reg_params;
    reg_train_fn reg_train;
    reg_rollout_fn reg_rollout;
    create_weights_fn create_weights;
    free_weights_fn free_weights;
    free_activations_fn free_activations;
    int in_dim, out_dim;
    float c_shift;
    size_t activation_size;  // sizeof(EncoderActivations) or custom override
};

struct Decoder {
    forward_fn forward;
    decoder_backward_fn backward;
    init_weights_fn init_weights;
    reg_params_fn reg_params;
    reg_train_fn reg_train;
    reg_rollout_fn reg_rollout;
    create_weights_fn create_weights;
    free_weights_fn free_weights;
    free_activations_fn free_activations;
    int hidden_dim, output_dim;
    bool continuous;
    size_t activation_size;
};

struct Network {
    network_forward_fn forward;
    network_forward_train_fn forward_train;
    network_backward_fn backward;
    init_weights_fn init_weights;
    reg_params_fn reg_params;
    reg_train_fn reg_train;
    reg_rollout_fn reg_rollout;
    create_weights_fn create_weights;
    free_weights_fn free_weights;
    free_activations_fn free_activations;
    int hidden, num_layers, horizon;
    int expansion;
    bool l2norm_layers;
    size_t activation_size;
};

struct EncoderWeights {
    PrecisionTensor weight;
    int in_dim, out_dim;
};

struct EncoderActivations {
    PrecisionTensor out, saved_input, wgrad_scratch;
};

// The core of 4.0 is the MinGRU fused scan operation. This allows us to parallelize
// training across the sequence dimension and scale to longer sequences
__device__ __forceinline__ void log_coeffs_and_values_fwd(float gate, float hidden,
        float* log_coeff_out, float* log_value_out) {
    float abs_gate = fabsf(gate);
    float sp_neg = log1pf(expf(-abs_gate));
    float softplus_gate = (gate >= 0.0f) ? gate + sp_neg : sp_neg;
    float softplus_neg_gate = (gate >= 0.0f) ? sp_neg : -gate + sp_neg;
    *log_coeff_out = -softplus_gate;
    float log_tilde_h = (hidden >= 0.0f) ? logf(hidden + 0.5f) : -softplus_fwd(-hidden);
    *log_value_out = -softplus_neg_gate + log_tilde_h;
}

__device__ __forceinline__ void log_coeffs_and_values_bwd(float grad_log_coeffs, float grad_log_values,
        float gate, float hidden, float* grad_gate_out, float* grad_hidden_out) {
    float sig_gate = sigmoid(gate);
    *grad_gate_out = -grad_log_coeffs * sig_gate + grad_log_values * (1.0f - sig_gate);
    *grad_hidden_out = (hidden >= 0.0f) ? grad_log_values / (hidden + 0.5f) : grad_log_values * sigmoid(-hidden);
}

__global__ void mingru_gate(precision_t* out, precision_t* next_state,
        const precision_t* combined, const precision_t* state_in,
        const precision_t* x_in, int H, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * H;
    if (idx >= N) {
        return;
    }

    int b = idx / H;
    int h = idx % H;

    // combined = linear(x_in) = (B, H) -> (B, 3*H)
    int combined_base = b * 3 * H;
    float hidden = to_float(combined[combined_base + h]);
    float gate = to_float(combined[combined_base + H + h]);
    float proj = to_float(combined[combined_base + 2*H + h]);
    float state = to_float(state_in[idx]);
    float x = to_float(x_in[idx]);

    // mingru_gate computation
    float gate_sigmoid = sigmoid(gate);
    float hidden_tilde = (hidden >= 0.0f) ? hidden + 0.5f : fast_sigmoid(hidden);
    float mingru_out = lerp(state, hidden_tilde, gate_sigmoid);

    // next_state is mingru_out (for recurrence)
    next_state[idx] = from_float(mingru_out);

    // Highway connection: sigmoid(proj) * mingru_out + (1 - sigmoid(proj)) * x (highway gate)
    float proj_sigmoid = sigmoid(proj);
    out[idx] = from_float(proj_sigmoid * mingru_out + (1.0f - proj_sigmoid) * x);
}

// Prefix scan buffers
struct PrefixScan {
    precision_t* combined_ptr = nullptr;
    precision_t* state_ptr = nullptr;
    precision_t* input_ptr = nullptr;  // (B, T, H) original input before projection (for highway gate)
    int B = 0, T = 0, H = 0;
    FloatTensor a_star, s_vals, log_values_buf;
    PrecisionTensor out, next_state;
    PrecisionTensor grad_combined, grad_state;
    PrecisionTensor grad_input;        // (B, T, H) highway gate gradient w.r.t. input
};

// Checkpointing trades off partial recomputation for memory bandwidth.
#define CHECKPOINT_INTERVAL 4
__global__ void mingru_scan_forward(PrefixScan scan) {
    int T_seq = scan.T, H = scan.H, B = scan.B;
    precision_t* __restrict__ out = scan.out.data;
    precision_t* __restrict__ next_state = scan.next_state.data;
    float* __restrict__ a_star_buf = scan.a_star.data;
    float* __restrict__ s_buf = scan.s_vals.data;
    float* __restrict__ log_values_buf = scan.log_values_buf.data;
    const precision_t* __restrict__ combined = scan.combined_ptr;
    const precision_t* __restrict__ state = scan.state_ptr;
    const precision_t* __restrict__ input = scan.input_ptr;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) {
        return;
    }

    int b = idx / H;
    int h = idx % H;

    int bH = b * H;
    int H3 = 3 * H;
    int H2 = 2 * H;
    int bHT = bH * T_seq;
    int out_base = bHT + h;
    int cbase = 3 * bHT;

    float a_star = 0.0f;
    float log_value = 0.0f;

    // Handle t=0 outside the loop: use log(state), coeff = 0
    float s = __logf(to_float(state[bH + h]));
    log_value = s;

    int T_out = T_seq + 1;
    int buf_base = b * T_out * H + h;
    int buf_curr = buf_base;
    a_star_buf[buf_curr] = a_star;
    s_buf[buf_curr] = s;
    log_values_buf[buf_curr] = log_value;

    const precision_t* combined_h_base = &combined[cbase + h];
    const precision_t* combined_g_base = &combined[cbase + H + h];
    const precision_t* combined_p_base = &combined[cbase + H2 + h];

    // Loop t=1..T_seq with sparse checkpointing
    float scan_result = 0.0f;
    int out_curr = out_base;
    int t_offset = 0;

    for (int t = 1; t < T_seq + 1; t++) {
        float hidden_val = to_float(combined_h_base[t_offset]);
        float gate_val = to_float(combined_g_base[t_offset]);
        float proj_val = to_float(combined_p_base[t_offset]);
        float x_val = to_float(input[out_base + (t - 1) * H]);

        float log_coeff_val;
        log_coeffs_and_values_fwd(gate_val, hidden_val, &log_coeff_val, &log_value);

        // a_star[t] = sum_{i=0}^t log_coeffs[i]
        a_star += log_coeff_val;

        float z = log_value - a_star;
        s = logaddexp(s, z);

        scan_result = __expf(a_star + s);
        float proj_sigmoid = sigmoid(proj_val);

        // out = sigmoid(proj) * scan_result + (1 - sigmoid(proj)) * x (highway gate)
        out[out_curr] = from_float(proj_sigmoid * scan_result + (1.0f - proj_sigmoid) * x_val);

        buf_curr += H;
        out_curr += H;
        t_offset += H3;

        if (t % CHECKPOINT_INTERVAL == 0) {
            a_star_buf[buf_curr] = a_star;
            s_buf[buf_curr] = s;
            log_values_buf[buf_curr] = log_value;
        }
    }

    // Write timestep T to next_state (raw scan_result, no proj, for recurrence)
    next_state[bH + h] = from_float(scan_result);
}

// Reads sparse checkpoints from forward pass, recomputes intermediate values in chunks
__global__ void mingru_scan_backward(PrefixScan scan,
        const precision_t* __restrict__ grad_out,
        const precision_t* __restrict__ grad_next_state) {
    int T_seq = scan.T, H = scan.H, B = scan.B;
    precision_t* __restrict__ grad_combined = scan.grad_combined.data;
    precision_t* __restrict__ grad_state = scan.grad_state.data;
    precision_t* __restrict__ grad_input = scan.grad_input.data;
    const precision_t* __restrict__ combined = scan.combined_ptr;
    const precision_t* __restrict__ state = scan.state_ptr;
    const precision_t* __restrict__ input = scan.input_ptr;
    const float* __restrict__ a_star_buf = scan.a_star.data;
    const float* __restrict__ s_buf = scan.s_vals.data;
    const float* __restrict__ log_values_buf = scan.log_values_buf.data;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) {
        return;
    }

    int b = idx / H;
    int h = idx % H;

    int bHT = b * H * T_seq;
    int cbase = 3 * bHT;
    int H3 = 3 * H;
    int H2 = 2 * H;
    const int state_idx = b * H + h;
    const int out_base = bHT + h;

    const precision_t* combined_h_base = &combined[cbase + h];
    const precision_t* combined_g_base = &combined[cbase + H + h];
    const precision_t* combined_p_base = &combined[cbase + H2 + h];

    precision_t* grad_combined_h_base = &grad_combined[cbase + h];
    precision_t* grad_combined_g_base = &grad_combined[cbase + H + h];
    precision_t* grad_combined_p_base = &grad_combined[cbase + H2 + h];

    int T_out = T_seq + 1;
    int buf_base = b * T_out * H + h;

    float acc = 0.0;
    float s_val_next = 0.0;
    float carry_grad_a = 0.0;

    for (int chunk_end = T_seq; chunk_end > 0; chunk_end -= CHECKPOINT_INTERVAL) {
        int chunk_start = (chunk_end > CHECKPOINT_INTERVAL) ? (chunk_end - CHECKPOINT_INTERVAL) : 0;
        int chunk_len = chunk_end - chunk_start;

        // Chunk storage in registers
        float chunk_a_star[CHECKPOINT_INTERVAL];
        float chunk_s[CHECKPOINT_INTERVAL];
        float chunk_log_values[CHECKPOINT_INTERVAL];
        float chunk_hidden[CHECKPOINT_INTERVAL];
        float chunk_gate[CHECKPOINT_INTERVAL];

        // Load checkpoint from global memory
        int ckpt_buf_idx = buf_base + chunk_start * H;
        float recomp_a_star = a_star_buf[ckpt_buf_idx];
        float recomp_s = s_buf[ckpt_buf_idx];
        float recomp_log_value = log_values_buf[ckpt_buf_idx];

        // Recompute and store from chunk_start to chunk_end
        for (int i = 0; i < chunk_len; ++i) {
            int t = chunk_start + 1 + i;
            int t_offset = (t - 1) * H3;
            float hv = to_float(combined_h_base[t_offset]);
            float gv = to_float(combined_g_base[t_offset]);

            float lc;
            log_coeffs_and_values_fwd(gv, hv, &lc, &recomp_log_value);
            recomp_a_star += lc;

            float z = recomp_log_value - recomp_a_star;
            recomp_s = logaddexp(recomp_s, z);

            chunk_a_star[i] = recomp_a_star;
            chunk_s[i] = recomp_s;
            chunk_log_values[i] = recomp_log_value;
            chunk_hidden[i] = hv;
            chunk_gate[i] = gv;
        }

        for (int i = chunk_len - 1; i >= 0; --i) {
            int t = chunk_start + 1 + i;
            int t_offset = (t - 1) * H3;

            float a_star_t = chunk_a_star[i];
            float s_t = chunk_s[i];
            float log_value_t = chunk_log_values[i];
            float hidden_val = chunk_hidden[i];
            float gate_val = chunk_gate[i];

            float proj_val = to_float(combined_p_base[t_offset]);
            int input_idx = out_base + (t - 1) * H;
            float x_val = to_float(input[input_idx]);

            float scan_result = __expf(a_star_t + s_t);
            float z = log_value_t - a_star_t;

            float grad_out_val = to_float(grad_out[input_idx]);
            float grad_scan_from_next = (t == T_seq) ? to_float(grad_next_state[state_idx]) : 0.0f;
            float proj_sigmoid = sigmoid(proj_val);

            // Highway gate gradients: out = sigmoid(proj) * scan_result + (1 - sigmoid(proj)) * x
            float grad_scan_result = grad_scan_from_next + grad_out_val * proj_sigmoid;
            float grad_proj = grad_out_val * (scan_result - x_val) * proj_sigmoid * (1.0f - proj_sigmoid);
            grad_input[input_idx] = from_float(grad_out_val * (1.0f - proj_sigmoid));

            float grad_log_h = grad_scan_result * scan_result;
            float grad_s = grad_log_h;

            if (t == T_seq) {
                acc = grad_s;
            } else {
                acc = grad_s + acc * __expf(s_t - s_val_next);
            }
            float grad_z = acc * __expf(z - s_t);
            s_val_next = s_t;

            float grad_a = grad_log_h + carry_grad_a - grad_z;
            carry_grad_a = grad_a;

            float grad_g, grad_h;
            log_coeffs_and_values_bwd(grad_a, grad_z, gate_val, hidden_val, &grad_g, &grad_h);

            grad_combined_h_base[t_offset] = from_float(grad_h);
            grad_combined_g_base[t_offset] = from_float(grad_g);
            grad_combined_p_base[t_offset] = from_float(grad_proj);
        }
    }

    int ckpt_0_idx = buf_base;
    float a_star_0 = a_star_buf[ckpt_0_idx];
    float s_0 = s_buf[ckpt_0_idx];
    float log_value_0 = log_values_buf[ckpt_0_idx];

    float scan_result_0 = __expf(a_star_0 + s_0);
    float z_0 = log_value_0 - a_star_0;

    float grad_scan_result_0 = 0.0f;
    float grad_log_h_0 = grad_scan_result_0 * scan_result_0;
    float grad_s_0 = grad_log_h_0;

    acc = grad_s_0 + acc * __expf(s_0 - s_val_next);
    float grad_z_0 = acc * __expf(z_0 - s_0);

    grad_state[state_idx] = from_float(grad_z_0 / to_float(state[state_idx]));
}

__global__ void sum_rows_to_precision_kernel(precision_t* __restrict__ dst,
        const float* __restrict__ src, int R, int C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= C) {
        return;
    }
    float sum = 0.0f;
    for (int r = 0; r < R; r++) {
        sum += src[r * C + col];
    }
    dst[col] = from_float(sum);
}

__global__ void assemble_decoder_grad(
        precision_t* __restrict__ dst, const float* __restrict__ grad_logits,
        const float* __restrict__ grad_value, int B_TT, int od, int od_plus_1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B_TT * od_plus_1) {
        return;
    }
    int row = idx / od_plus_1, col = idx % od_plus_1;
    dst[idx] = from_float((col < od) ? grad_logits[row * od + col] : grad_value[row]);
}

static PrecisionTensor encoder_forward(void* w, void* activations, PrecisionTensor input, cudaStream_t stream) {
    EncoderWeights* ew = (EncoderWeights*)w;
    EncoderActivations* a = (EncoderActivations*)activations;
    if (a->saved_input.data) puf_copy(&a->saved_input, &input, stream);
    puf_mm(&input, &ew->weight, &a->out, stream);
    return a->out;
}

static void encoder_backward(void* w, void* activations, PrecisionTensor grad, cudaStream_t stream) {
    EncoderActivations* a = (EncoderActivations*)activations;
    puf_mm_tn(&grad, &a->saved_input, &a->wgrad_scratch, stream);
}

static void encoder_init_weights(void* w, ulong* seed, cudaStream_t stream) {
    EncoderWeights* ew = (EncoderWeights*)w;
    PrecisionTensor wt = {
        .data = ew->weight.data,
        .shape = {ew->out_dim, ew->in_dim},
    };
    puf_kaiming_init(&wt, std::sqrt(2.0f), (*seed)++, stream);
}

static void encoder_reg_params(void* w, Allocator* alloc) {
    EncoderWeights* ew = (EncoderWeights*)w;
    ew->weight = {.shape = {ew->out_dim, ew->in_dim}};
    alloc_register(alloc,&ew->weight);
}

static void encoder_reg_train(void* w, void* activations, Allocator* acts, Allocator* grads, int B_TT) {
    EncoderWeights* ew = (EncoderWeights*)w;
    EncoderActivations* a = (EncoderActivations*)activations;
    *a = (EncoderActivations){
        .out =              {.shape = {B_TT, ew->out_dim}},
        .saved_input =      {.shape = {B_TT, ew->in_dim}},
        .wgrad_scratch =    {.shape = {ew->out_dim, ew->in_dim}},
    };
    alloc_register(acts,&a->out);
    alloc_register(acts,&a->saved_input);
    alloc_register(grads,&a->wgrad_scratch);
}

static void encoder_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    EncoderWeights* ew = (EncoderWeights*)w;
    EncoderActivations* a = (EncoderActivations*)activations;
    a->out = {.shape = {B, ew->out_dim}};
    alloc_register(alloc,&a->out);
}

static void* encoder_create_weights(void* self) {
    Encoder* e = (Encoder*)self;
    EncoderWeights* ew = (EncoderWeights*)calloc(1, sizeof(EncoderWeights));
    ew->in_dim = e->in_dim; ew->out_dim = e->out_dim;
    return ew;
}

static void encoder_free_weights(void* weights) {
    free(weights);
}

static void encoder_free_activations(void* activations) {
    free(activations);
}

struct DecoderWeights {
    PrecisionTensor weight, logstd;
    int hidden_dim, output_dim;
    bool continuous;
};

struct DecoderActivations {
    PrecisionTensor out, grad_out, saved_input, grad_input, wgrad_scratch, logstd_scratch;
};

static PrecisionTensor decoder_forward(void* w, void* activations, PrecisionTensor input, cudaStream_t stream) {
    DecoderWeights* dw = (DecoderWeights*)w;
    DecoderActivations* a = (DecoderActivations*)activations;
    if (a->saved_input.data) {
        puf_copy(&a->saved_input, &input, stream);
    }
    puf_mm(&input, &dw->weight, &a->out, stream);
    return a->out;
}

static void decoder_init_weights(void* w, ulong* seed, cudaStream_t stream) {
    DecoderWeights* dw = (DecoderWeights*)w;
    PrecisionTensor wt = {
        .data = dw->weight.data,
        .shape = {dw->output_dim + 1, dw->hidden_dim},
    };
    puf_kaiming_init(&wt, 1.0f, (*seed)++, stream);
}

static void decoder_reg_params(void* w, Allocator* alloc) {
    DecoderWeights* dw = (DecoderWeights*)w;
    dw->weight = {.shape = {dw->output_dim + 1, dw->hidden_dim}};
    alloc_register(alloc,&dw->weight);
    if (dw->continuous) {
        dw->logstd = {.shape = {1, dw->output_dim}};
        alloc_register(alloc,&dw->logstd);
    }
}

static void decoder_reg_train(void* w, void* activations, Allocator* acts, Allocator* grads, int B_TT) {
    DecoderWeights* dw = (DecoderWeights*)w;
    DecoderActivations* a = (DecoderActivations*)activations;
    int od1 = dw->output_dim + 1;
    *a = (DecoderActivations){
        .out =              {.shape = {B_TT, od1}},
        .grad_out =         {.shape = {B_TT, od1}},
        .saved_input =      {.shape = {B_TT, dw->hidden_dim}},
        .grad_input =       {.shape = {B_TT, dw->hidden_dim}},
        .wgrad_scratch =    {.shape = {od1, dw->hidden_dim}},
        .logstd_scratch =   {.shape = {1, dw->output_dim}},
    };
    alloc_register(acts,&a->out);
    alloc_register(acts,&a->saved_input);
    alloc_register(acts,&a->grad_out);
    alloc_register(acts,&a->grad_input);
    alloc_register(grads,&a->wgrad_scratch);
    if (dw->continuous) alloc_register(grads,&a->logstd_scratch);
}

static void decoder_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    DecoderWeights* dw = (DecoderWeights*)w;
    DecoderActivations* a = (DecoderActivations*)activations;
    a->out = {.shape = {B, dw->output_dim + 1}};
    alloc_register(alloc,&a->out);
}

static void* decoder_create_weights(void* self) {
    Decoder* d = (Decoder*)self;
    DecoderWeights* dw = (DecoderWeights*)calloc(1, sizeof(DecoderWeights));
    dw->hidden_dim = d->hidden_dim; dw->output_dim = d->output_dim; dw->continuous = d->continuous;
    return dw;
}

static void decoder_free_weights(void* weights) {
    free(weights);
}

static void decoder_free_activations(void* activations) {
    free(activations);
}

static PrecisionTensor decoder_backward(void* w, void* activations,
    FloatTensor grad_logits, FloatTensor grad_logstd, FloatTensor grad_value, cudaStream_t stream) {
    DecoderWeights* dw = (DecoderWeights*)w;
    DecoderActivations* a = (DecoderActivations*)activations;
    int B_TT = a->saved_input.shape[0];
    int od = dw->output_dim, od1 = od + 1;
    assemble_decoder_grad<<<grid_size(B_TT * od1), BLOCK_SIZE, 0, stream>>>(
        a->grad_out.data, grad_logits.data, grad_value.data, B_TT, od, od1);
    puf_mm_tn(&a->grad_out, &a->saved_input, &a->wgrad_scratch, stream);
    if (dw->continuous && grad_logstd.data != nullptr) {
        sum_rows_to_precision_kernel<<<grid_size(dw->output_dim), BLOCK_SIZE, 0, stream>>>(
            a->logstd_scratch.data, grad_logstd.data, B_TT, dw->output_dim);
    }
    puf_mm_nn(&a->grad_out, &dw->weight, &a->grad_input, stream);
    return a->grad_input;
}

#define SV2_EPS 1e-8f

__global__ void sv2_l2norm_fwd(precision_t* out, float* norms_out,
        const precision_t* input, int num_rows, int row_len) {
    int row = blockIdx.x;
    if (row >= num_rows) return;
    __shared__ float sdata[BLOCK_SIZE];
    float sum = 0;
    for (int j = threadIdx.x; j < row_len; j += blockDim.x) {
        float v = to_float(input[row * row_len + j]);
        sum += v * v;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float norm = sqrtf(sdata[0]);
    if (threadIdx.x == 0 && norms_out) norms_out[row] = norm;
    float inv = 1.0f / fmaxf(norm, SV2_EPS);
    for (int j = threadIdx.x; j < row_len; j += blockDim.x)
        out[row * row_len + j] = from_float(to_float(input[row * row_len + j]) * inv);
}

__global__ void sv2_l2norm_dots(float* dots, const precision_t* grad_y,
        const precision_t* y, int num_rows, int row_len) {
    int row = blockIdx.x;
    if (row >= num_rows) return;
    __shared__ float sdata[BLOCK_SIZE];
    float sum = 0;
    for (int j = threadIdx.x; j < row_len; j += blockDim.x)
        sum += to_float(grad_y[row * row_len + j]) * to_float(y[row * row_len + j]);
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) dots[row] = sdata[0];
}

__global__ void sv2_l2norm_bwd(precision_t* grad_x, const precision_t* grad_y,
        const precision_t* y, const float* dots, const float* norms,
        int N, int row_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int row = idx / row_len;
    float inv = 1.0f / fmaxf(norms[row], SV2_EPS);
    grad_x[idx] = from_float((to_float(grad_y[idx]) - to_float(y[idx]) * dots[row]) * inv);
}

struct MinGRUActivations {
    int num_layers;
    // Rollout
    PrecisionTensor* combined;       // (B rollout, 3*T)[num_layers]
    PrecisionTensor out;             // (B rollout, T)
    PrecisionTensor next_state;      // (B rollout, T)
    // Training
    PrecisionTensor* saved_inputs;   // (B, TT, T)[num_layers]
    PrefixScan* scan_bufs;           // [num_layers]
    PrecisionTensor* combined_bufs;  // (B*TT, 3*T)[num_layers]
    PrecisionTensor* wgrad_scratch;  // (3*T, T)[num_layers]
    PrecisionTensor grad_input_buf;  // (B*TT, T)
    PrecisionTensor grad_next_state; // (B, 1, T)
    FloatTensor* layer_norms;        // [num_layers] (B_TT,)
    FloatTensor l2_dots;             // (B_TT,)
    PrecisionTensor l2_grad_buf;     // (B_TT, H)
};

void mingru_activations_free(MinGRUActivations* a) {
    free(a->combined);
    free(a->saved_inputs);
    free(a->scan_bufs);
    free(a->combined_bufs);
    free(a->wgrad_scratch);
    free(a->layer_norms);
}

struct MinGRUWeights {
    int hidden, num_layers, horizon;
    bool l2norm_layers;
    PrecisionTensor* weights;  // [num_layers]
};

static PrecisionTensor mingru_state_layer(MinGRUWeights* m, PrecisionTensor& state, int i) {
    long B = state.shape[1], H = state.shape[2];
    return {.data = state.data + i * B * H, .shape = {B, H}};
}

static void mingru_init_weights(void* w, ulong* seed, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    for (int i = 0; i < m->num_layers; i++) {
        PrecisionTensor w2d = {
            .data = m->weights[i].data,
            .shape = {3 * m->hidden, m->hidden},
        };
        puf_kaiming_init(&w2d, 1.0f, (*seed)++, stream);
    }
}

static void mingru_reg_params(void* w, Allocator* alloc) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    for (int i = 0; i < m->num_layers; i++) {
        m->weights[i] = {.shape = {3 * m->hidden, m->hidden}};
        alloc_register(alloc,&m->weights[i]);
    }
}

static void mingru_reg_train(void* w, void* activations, Allocator* acts, Allocator* grads, int B_TT) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    int H = m->hidden, TT = m->horizon, B = B_TT / TT;
    a->num_layers = m->num_layers;
    a->saved_inputs = (PrecisionTensor*)calloc(m->num_layers, sizeof(PrecisionTensor));
    a->scan_bufs = (PrefixScan*)calloc(m->num_layers, sizeof(PrefixScan));
    a->combined_bufs = (PrecisionTensor*)calloc(m->num_layers, sizeof(PrecisionTensor));
    a->wgrad_scratch = (PrecisionTensor*)calloc(m->num_layers, sizeof(PrecisionTensor));
    a->grad_input_buf = {.shape = {B_TT, H}};
    a->grad_next_state = {.shape = {B, 1, H}};
    alloc_register(acts,&a->grad_input_buf);
    alloc_register(acts,&a->grad_next_state);
    for (int i = 0; i < m->num_layers; i++) {
        a->scan_bufs[i] = {
            .B = B, .T = TT, .H = H,
            .a_star =           {.shape = {B, TT + 1, H}},
            .s_vals =           {.shape = {B, TT + 1, H}},
            .log_values_buf =   {.shape = {B, TT + 1, H}},
            .out =              {.shape = {B, TT, H}},
            .next_state =       {.shape = {B, 1, H}},
            .grad_combined =    {.shape = {B, TT, 3 * H}},
            .grad_state =       {.shape = {B, 1, H}},
            .grad_input =       {.shape = {B, TT, H}},
        };
        a->saved_inputs[i]  = {.shape = {B, TT, H}};
        a->combined_bufs[i] = {.shape = {B_TT, 3 * H}};
        a->wgrad_scratch[i] = {.shape = {3 * H, H}};
        alloc_register(acts,&a->saved_inputs[i]);
        alloc_register(acts,&a->combined_bufs[i]);
        alloc_register(acts,&a->scan_bufs[i].out);
        alloc_register(acts,&a->scan_bufs[i].next_state);
        alloc_register(acts,&a->scan_bufs[i].a_star);
        alloc_register(acts,&a->scan_bufs[i].s_vals);
        alloc_register(acts,&a->scan_bufs[i].log_values_buf);
        alloc_register(acts,&a->scan_bufs[i].grad_combined);
        alloc_register(acts,&a->scan_bufs[i].grad_state);
        alloc_register(acts,&a->scan_bufs[i].grad_input);
        alloc_register(grads,&a->wgrad_scratch[i]);
    }
    if (m->l2norm_layers) {
        a->layer_norms = (FloatTensor*)calloc(m->num_layers, sizeof(FloatTensor));
        a->l2_dots = {.shape = {B_TT}};
        a->l2_grad_buf = {.shape = {B_TT, H}};
        alloc_register(acts, &a->l2_dots);
        alloc_register(acts, &a->l2_grad_buf);
        for (int i = 0; i < m->num_layers; i++) {
            a->layer_norms[i] = {.shape = {B_TT}};
            alloc_register(acts, &a->layer_norms[i]);
        }
    }
}

static void mingru_reg_rollout(void* weights, void* activations, Allocator* alloc, int B_inf) {
    MinGRUWeights* w = (MinGRUWeights*)weights;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    int H = w->hidden;
    a->num_layers = w->num_layers;
    a->combined = (PrecisionTensor*)calloc(w->num_layers, sizeof(PrecisionTensor));
    for (int i = 0; i < w->num_layers; i++) {
        a->combined[i] = {.shape = {B_inf, 3 * H}};
        alloc_register(alloc,&a->combined[i]);
    }
    a->out = {.shape = {B_inf, H}};
    a->next_state = {.shape = {B_inf, H}};
    alloc_register(alloc,&a->out);
    alloc_register(alloc,&a->next_state);
}

static void* mingru_create_weights(void* self) {
    Network* n = (Network*)self;
    MinGRUWeights* mw = (MinGRUWeights*)calloc(1, sizeof(MinGRUWeights));
    mw->hidden = n->hidden; mw->num_layers = n->num_layers; mw->horizon = n->horizon;
    mw->l2norm_layers = n->l2norm_layers;
    mw->weights = (PrecisionTensor*)calloc(n->num_layers, sizeof(PrecisionTensor));
    return mw;
}

static void mingru_free_weights(void* weights) {
    MinGRUWeights* mw = (MinGRUWeights*)weights;
    free(mw->weights);
    free(mw);
}

static void mingru_free_activations(void* activations) {
    MinGRUActivations* a = (MinGRUActivations*)activations;
    mingru_activations_free(a);
    free(a);
}

static PrecisionTensor mingru_forward(void* w, PrecisionTensor x, PrecisionTensor state,
        void* activations, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    int B = state.shape[1];
    int H = state.shape[2];
    for (int i = 0; i < m->num_layers; i++) {
        PrecisionTensor state_i = mingru_state_layer(m, state, i);
        puf_mm(&x, &m->weights[i], &a->combined[i], stream);
        mingru_gate<<<grid_size(B*H), BLOCK_SIZE, 0, stream>>>(
            a->out.data, a->next_state.data,
            a->combined[i].data, state_i.data, x.data, H, B);
        puf_copy(&state_i, &a->next_state, stream);
        if (m->l2norm_layers)
            sv2_l2norm_fwd<<<B, BLOCK_SIZE, 0, stream>>>(
                a->out.data, nullptr, a->out.data, B, H);
        x = a->out;
    }
    return x;
}

static PrecisionTensor mingru_forward_train(void* w, PrecisionTensor x, PrecisionTensor state,
        void* activations, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    int B = x.shape[0];
    for (int i = 0; i < m->num_layers; i++) {
        puf_copy(&a->saved_inputs[i], &x, stream);
        PrecisionTensor state_i = mingru_state_layer(m, state, i);
        puf_mm(&x, &m->weights[i], &a->combined_bufs[i], stream);
        a->scan_bufs[i].combined_ptr = a->combined_bufs[i].data;
        a->scan_bufs[i].state_ptr = state_i.data;
        a->scan_bufs[i].input_ptr = a->saved_inputs[i].data;
        mingru_scan_forward<<<grid_size(B*m->hidden), BLOCK_SIZE, 0, stream>>>(a->scan_bufs[i]);
        if (m->l2norm_layers) {
            int B_TT = a->scan_bufs[i].B * a->scan_bufs[i].T;
            sv2_l2norm_fwd<<<B_TT, BLOCK_SIZE, 0, stream>>>(
                a->scan_bufs[i].out.data, a->layer_norms[i].data,
                a->scan_bufs[i].out.data, B_TT, m->hidden);
        }
        x = a->scan_bufs[i].out;
    }
    return x;
}

static PrecisionTensor mingru_backward(void* w, PrecisionTensor grad, void* activations, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    for (int i = m->num_layers - 1; i >= 0; i--) {
        PrefixScan& scan = a->scan_bufs[i];
        precision_t* grad_ptr = grad.data;
        if (m->l2norm_layers) {
            int B_TT = scan.B * scan.T;
            int N = B_TT * scan.H;
            sv2_l2norm_dots<<<B_TT, BLOCK_SIZE, 0, stream>>>(
                a->l2_dots.data, grad.data, scan.out.data, B_TT, scan.H);
            sv2_l2norm_bwd<<<grid_size(N), BLOCK_SIZE, 0, stream>>>(
                a->l2_grad_buf.data, grad.data, scan.out.data,
                a->l2_dots.data, a->layer_norms[i].data, N, scan.H);
            grad_ptr = a->l2_grad_buf.data;
        }
        mingru_scan_backward<<<grid_size(scan.B*scan.H), BLOCK_SIZE, 0, stream>>>(
            scan, grad_ptr, a->grad_next_state.data);
        puf_mm_tn(&scan.grad_combined, &a->saved_inputs[i], &a->wgrad_scratch[i], stream);
        puf_mm_nn(&scan.grad_combined, &m->weights[i], &a->grad_input_buf, stream);
        int n = numel(scan.grad_input.shape);
        add_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
            a->grad_input_buf.data, scan.grad_input.data, n);
        grad = a->grad_input_buf;
    }
    return grad;
}

typedef void (*policy_post_step_fn)(void* encoder_w, void* network_w,
    precision_t* param_base, float* master_weights, cudaStream_t stream);

// SimbaV2: Hyperspherical RL Architecture

__global__ void sv2_augment_const(precision_t* dst, const precision_t* src,
        float c_shift, int num_rows, int in_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_dim = in_dim + 1;
    int total = num_rows * out_dim;
    if (idx >= total) return;
    int b = idx / out_dim, j = idx % out_dim;
    dst[idx] = (j < in_dim) ? src[b * in_dim + j] : from_float(c_shift);
}

__global__ void sv2_scale(precision_t* out, const precision_t* input,
        const precision_t* scaler, float fwd_scaler, int N, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    out[idx] = from_float(to_float(scaler[idx % H]) * fwd_scaler * to_float(input[idx]));
}

__global__ void sv2_scale_relu_eps(precision_t* out, const precision_t* input,
        const precision_t* scaler, float fwd_scaler, int N, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float v = to_float(scaler[idx % H]) * fwd_scaler * to_float(input[idx]);
    out[idx] = from_float(fmaxf(0.0f, v) + SV2_EPS);
}

__global__ void sv2_lerp(precision_t* dst, const precision_t* residual,
        const precision_t* mlp, const precision_t* alpha, float alpha_fwd,
        int N, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float r = to_float(residual[idx]);
    float m = to_float(mlp[idx]);
    float a = to_float(alpha[idx % H]) * alpha_fwd;
    dst[idx] = from_float(r + a * (m - r));
}

__global__ void sv2_scale_grad(precision_t* grad_scaler,
        const precision_t* grad_y, const precision_t* x,
        float fwd_scaler, int num_rows, int H) {
    int h = blockIdx.x;
    if (h >= H) return;
    __shared__ float sdata[BLOCK_SIZE];
    float sum = 0;
    for (int b = threadIdx.x; b < num_rows; b += blockDim.x)
        sum += to_float(grad_y[b * H + h]) * to_float(x[b * H + h]);
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) grad_scaler[h] = from_float(fwd_scaler * sdata[0]);
}

__global__ void sv2_scale_relu_bwd(precision_t* grad_input,
        const precision_t* grad_out, const precision_t* after_w1,
        const precision_t* scaler, float fwd_scaler, int N, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float s = to_float(scaler[idx % H]) * fwd_scaler;
    float pre_relu = s * to_float(after_w1[idx]);
    float g = (pre_relu > 0.0f) ? to_float(grad_out[idx]) : 0.0f;
    grad_input[idx] = from_float(s * g);
}

__global__ void sv2_scale_relu_grad(precision_t* grad_scaler,
        const precision_t* grad_out, const precision_t* after_w1,
        const precision_t* scaler, float fwd_scaler, int num_rows, int H) {
    int h = blockIdx.x;
    if (h >= H) return;
    float s = to_float(scaler[h]) * fwd_scaler;
    __shared__ float sdata[BLOCK_SIZE];
    float sum = 0;
    for (int b = threadIdx.x; b < num_rows; b += blockDim.x) {
        float pre_relu = s * to_float(after_w1[b * H + h]);
        float g = (pre_relu > 0.0f) ? to_float(grad_out[b * H + h]) : 0.0f;
        sum += g * to_float(after_w1[b * H + h]);
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (threadIdx.x < s2) sdata[threadIdx.x] += sdata[threadIdx.x + s2];
        __syncthreads();
    }
    if (threadIdx.x == 0) grad_scaler[h] = from_float(fwd_scaler * sdata[0]);
}

__global__ void sv2_lerp_bwd(precision_t* grad_mlp, precision_t* grad_res,
        const precision_t* grad_lerp, const precision_t* alpha,
        float alpha_fwd, int N, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float gl = to_float(grad_lerp[idx]);
    float a = to_float(alpha[idx % H]) * alpha_fwd;
    grad_res[idx] = from_float((1.0f - a) * gl);
    grad_mlp[idx] = from_float(a * gl);
}

__global__ void sv2_alpha_grad(precision_t* grad_alpha,
        const precision_t* grad_lerp, const precision_t* mlp_out,
        const precision_t* input, float alpha_fwd, int num_rows, int H) {
    int h = blockIdx.x;
    if (h >= H) return;
    __shared__ float sdata[BLOCK_SIZE];
    float sum = 0;
    for (int b = threadIdx.x; b < num_rows; b += blockDim.x) {
        float gl = to_float(grad_lerp[b * H + h]);
        float diff = to_float(mlp_out[b * H + h]) - to_float(input[b * H + h]);
        sum += gl * diff;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) grad_alpha[h] = from_float(alpha_fwd * sdata[0]);
}

__global__ void sv2_l2norm_weight_rows(float* W, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;
    float* r = W + row * N;
    __shared__ float sdata[BLOCK_SIZE];
    float sum = 0;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        float v = r[j]; sum += v * v;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(fmaxf(sdata[0], SV2_EPS));
    for (int j = threadIdx.x; j < N; j += blockDim.x) r[j] *= inv;
}

struct SV2EncoderWeights {
    PrecisionTensor w;          // (H, obs+1)
    PrecisionTensor scaler;     // (H,)
    int in_dim, out_dim;
    float scaler_fwd;
    float c_shift;
};

struct SV2EncoderActivations {
    PrecisionTensor augmented;  // (B, obs+1)
    PrecisionTensor proj;       // (B, H)
    PrecisionTensor out;        // (B, H)
    FloatTensor scaled_norms;   // (B,)
    PrecisionTensor wgrad;      // (H, obs+1)
    PrecisionTensor scaler_grad;// (H,)
    PrecisionTensor grad_buf;   // (B, H)
    FloatTensor dots;           // (B,)
};

static PrecisionTensor sv2_encoder_forward(void* w, void* activations,
        PrecisionTensor input, cudaStream_t stream) {
    SV2EncoderWeights* ew = (SV2EncoderWeights*)w;
    SV2EncoderActivations* a = (SV2EncoderActivations*)activations;
    int B = input.shape[0];
    int aug_dim = ew->in_dim + 1;
    int H = ew->out_dim;
    int N_aug = B * aug_dim;
    int N_h = B * H;

    sv2_augment_const<<<grid_size(N_aug), BLOCK_SIZE, 0, stream>>>(
        a->augmented.data, input.data, ew->c_shift, B, ew->in_dim);
    sv2_l2norm_fwd<<<B, BLOCK_SIZE, 0, stream>>>(
        a->augmented.data, nullptr, a->augmented.data, B, aug_dim);

    if (a->proj.data) {
        puf_mm(&a->augmented, &ew->w, &a->proj, stream);
        sv2_scale<<<grid_size(N_h), BLOCK_SIZE, 0, stream>>>(
            a->out.data, a->proj.data, ew->scaler.data, ew->scaler_fwd, N_h, H);
    } else {
        puf_mm(&a->augmented, &ew->w, &a->out, stream);
        sv2_scale<<<grid_size(N_h), BLOCK_SIZE, 0, stream>>>(
            a->out.data, a->out.data, ew->scaler.data, ew->scaler_fwd, N_h, H);
    }
    sv2_l2norm_fwd<<<B, BLOCK_SIZE, 0, stream>>>(
        a->out.data, a->scaled_norms.data, a->out.data, B, H);
    return a->out;
}

static void sv2_encoder_backward(void* w, void* activations,
        PrecisionTensor grad, cudaStream_t stream) {
    SV2EncoderWeights* ew = (SV2EncoderWeights*)w;
    SV2EncoderActivations* a = (SV2EncoderActivations*)activations;
    int B_TT = a->proj.shape[0];
    int H = ew->out_dim;
    int N = B_TT * H;

    sv2_l2norm_dots<<<B_TT, BLOCK_SIZE, 0, stream>>>(
        a->dots.data, grad.data, a->out.data, B_TT, H);
    sv2_l2norm_bwd<<<grid_size(N), BLOCK_SIZE, 0, stream>>>(
        a->grad_buf.data, grad.data, a->out.data, a->dots.data,
        a->scaled_norms.data, N, H);

    sv2_scale_grad<<<H, BLOCK_SIZE, 0, stream>>>(
        a->scaler_grad.data, a->grad_buf.data, a->proj.data,
        ew->scaler_fwd, B_TT, H);
    sv2_scale<<<grid_size(N), BLOCK_SIZE, 0, stream>>>(
        a->grad_buf.data, a->grad_buf.data, ew->scaler.data, ew->scaler_fwd, N, H);

    puf_mm_tn(&a->grad_buf, &a->augmented, &a->wgrad, stream);
}

static void sv2_encoder_init_weights(void* w, ulong* seed, cudaStream_t stream) {
    SV2EncoderWeights* ew = (SV2EncoderWeights*)w;
    puf_normal_init(&ew->w, 1.0f, (*seed)++, stream);
    float scaler_init_val = sqrtf(2.0f / ew->out_dim);
    fill_precision_kernel<<<grid_size(ew->out_dim), BLOCK_SIZE, 0, stream>>>(
        ew->scaler.data, from_float(scaler_init_val), ew->out_dim);
}

static void sv2_encoder_reg_params(void* w, Allocator* alloc) {
    SV2EncoderWeights* ew = (SV2EncoderWeights*)w;
    int aug_dim = ew->in_dim + 1;
    ew->w = {.shape = {ew->out_dim, aug_dim}};
    ew->scaler = {.shape = {ew->out_dim}};
    alloc_register(alloc, &ew->w);
    alloc_register(alloc, &ew->scaler);
}

static void sv2_encoder_reg_train(void* w, void* activations,
        Allocator* acts, Allocator* grads, int B_TT) {
    SV2EncoderWeights* ew = (SV2EncoderWeights*)w;
    SV2EncoderActivations* a = (SV2EncoderActivations*)activations;
    int aug_dim = ew->in_dim + 1;
    int H = ew->out_dim;
    *a = {};
    a->augmented =    {.shape = {B_TT, aug_dim}};
    a->proj =         {.shape = {B_TT, H}};
    a->out =          {.shape = {B_TT, H}};
    a->scaled_norms = {.shape = {B_TT}};
    a->wgrad =        {.shape = {H, aug_dim}};
    a->scaler_grad =  {.shape = {H}};
    a->grad_buf =     {.shape = {B_TT, H}};
    a->dots =         {.shape = {B_TT}};
    alloc_register(acts, &a->augmented);
    alloc_register(acts, &a->proj);
    alloc_register(acts, &a->out);
    alloc_register(acts, &a->scaled_norms);
    alloc_register(acts, &a->grad_buf);
    alloc_register(acts, &a->dots);
    alloc_register(grads, &a->wgrad);
    alloc_register(grads, &a->scaler_grad);
}

static void sv2_encoder_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    SV2EncoderWeights* ew = (SV2EncoderWeights*)w;
    SV2EncoderActivations* a = (SV2EncoderActivations*)activations;
    int aug_dim = ew->in_dim + 1;
    int H = ew->out_dim;
    *a = {};
    a->augmented = {.shape = {B, aug_dim}};
    a->out =       {.shape = {B, H}};
    a->scaled_norms = {};
    alloc_register(alloc, &a->augmented);
    alloc_register(alloc, &a->out);
}

static void* sv2_encoder_create_weights(void* self) {
    Encoder* e = (Encoder*)self;
    SV2EncoderWeights* ew = (SV2EncoderWeights*)calloc(1, sizeof(SV2EncoderWeights));
    ew->in_dim = e->in_dim;
    ew->out_dim = e->out_dim;
    ew->scaler_fwd = 1.0f;
    ew->c_shift = e->c_shift;
    return ew;
}

static void sv2_encoder_free_weights(void* w) { free(w); }
static void sv2_encoder_free_activations(void* a) { free(a); }

struct SV2NetworkWeights {
    int hidden, num_blocks, horizon, expansion;
    float mlp_scaler_fwd;
    float alpha_fwd;
    PrecisionTensor* w1;         // [num_blocks] (H*4, H)
    PrecisionTensor* mlp_scaler; // [num_blocks] (H*4,)
    PrecisionTensor* w2;         // [num_blocks] (H, H*4)
    PrecisionTensor* alpha;      // [num_blocks] (H,)
};

struct SV2NetworkActivations {
    int num_blocks;
    PrecisionTensor saved_input0; // (B_TT, H)
    PrecisionTensor* after_w1;    // [num_blocks] (B_TT, H*4)
    PrecisionTensor* after_relu;  // [num_blocks] (B_TT, H*4)
    PrecisionTensor* mlp_out;     // [num_blocks] (B_TT, H)
    FloatTensor* mlp_norms;       // [num_blocks] (B_TT,)
    PrecisionTensor* block_out;   // [num_blocks] (B_TT, H)
    FloatTensor* block_norms;     // [num_blocks] (B_TT,)
    PrecisionTensor* wgrad_w1;    // [num_blocks] (H*4, H)
    PrecisionTensor* sgrad;       // [num_blocks] (H*4,)
    PrecisionTensor* wgrad_w2;    // [num_blocks] (H, H*4)
    PrecisionTensor* agrad;       // [num_blocks] (H,)
    PrecisionTensor grad_a, grad_b; // (B_TT, H)
    PrecisionTensor grad_h4;        // (B_TT, H*4)
    FloatTensor dots;               // (B_TT,)
    PrecisionTensor temp_h4;     // (B, H*4) rollout
    PrecisionTensor temp_h;      // (B, H) rollout
    PrecisionTensor out;         // (B, H) rollout
};

static PrecisionTensor sv2_network_forward(void* w, PrecisionTensor x,
        PrecisionTensor state, void* activations, cudaStream_t stream) {
    SV2NetworkWeights* nw = (SV2NetworkWeights*)w;
    SV2NetworkActivations* a = (SV2NetworkActivations*)activations;
    int B = x.shape[0], H = nw->hidden, H4 = H * nw->expansion;
    int N_h4 = B * H4, N_h = B * H;

    PrecisionTensor cur = x; // (B, H) — encoder output
    for (int i = 0; i < nw->num_blocks; i++) {
        puf_mm(&cur, &nw->w1[i], &a->temp_h4, stream);
        sv2_scale_relu_eps<<<grid_size(N_h4), BLOCK_SIZE, 0, stream>>>(
            a->temp_h4.data, a->temp_h4.data, nw->mlp_scaler[i].data,
            nw->mlp_scaler_fwd, N_h4, H4);
        puf_mm(&a->temp_h4, &nw->w2[i], &a->temp_h, stream);
        sv2_l2norm_fwd<<<B, BLOCK_SIZE, 0, stream>>>(
            a->temp_h.data, nullptr, a->temp_h.data, B, H);
        sv2_lerp<<<grid_size(N_h), BLOCK_SIZE, 0, stream>>>(
            a->out.data, cur.data, a->temp_h.data, nw->alpha[i].data,
            nw->alpha_fwd, N_h, H);
        sv2_l2norm_fwd<<<B, BLOCK_SIZE, 0, stream>>>(
            a->out.data, nullptr, a->out.data, B, H);
        cur = a->out;
    }
    return a->out;
}

static PrecisionTensor sv2_network_forward_train(void* w, PrecisionTensor x,
        PrecisionTensor state, void* activations, cudaStream_t stream) {
    SV2NetworkWeights* nw = (SV2NetworkWeights*)w;
    SV2NetworkActivations* a = (SV2NetworkActivations*)activations;
    int B = x.shape[0], TT = x.shape[1], H = nw->hidden;
    int H4 = H * nw->expansion;
    int B_TT = B * TT;
    int N_h4 = B_TT * H4, N_h = B_TT * H;

    PrecisionTensor x_flat = {.data = x.data, .shape = {B_TT, H}};
    puf_copy(&a->saved_input0, &x_flat, stream);

    PrecisionTensor cur = x_flat;
    for (int i = 0; i < nw->num_blocks; i++) {
        puf_mm(&cur, &nw->w1[i], &a->after_w1[i], stream);
        sv2_scale_relu_eps<<<grid_size(N_h4), BLOCK_SIZE, 0, stream>>>(
            a->after_relu[i].data, a->after_w1[i].data, nw->mlp_scaler[i].data,
            nw->mlp_scaler_fwd, N_h4, H4);
        puf_mm(&a->after_relu[i], &nw->w2[i], &a->mlp_out[i], stream);
        sv2_l2norm_fwd<<<B_TT, BLOCK_SIZE, 0, stream>>>(
            a->mlp_out[i].data, a->mlp_norms[i].data, a->mlp_out[i].data, B_TT, H);
        sv2_lerp<<<grid_size(N_h), BLOCK_SIZE, 0, stream>>>(
            a->block_out[i].data, cur.data, a->mlp_out[i].data, nw->alpha[i].data,
            nw->alpha_fwd, N_h, H);
        sv2_l2norm_fwd<<<B_TT, BLOCK_SIZE, 0, stream>>>(
            a->block_out[i].data, a->block_norms[i].data, a->block_out[i].data, B_TT, H);
        cur = a->block_out[i];
    }
    PrecisionTensor result = {.data = a->block_out[nw->num_blocks - 1].data, .shape = {B, TT, H}};
    return result;
}

static PrecisionTensor sv2_network_backward(void* w, PrecisionTensor grad,
        void* activations, cudaStream_t stream) {
    SV2NetworkWeights* nw = (SV2NetworkWeights*)w;
    SV2NetworkActivations* a = (SV2NetworkActivations*)activations;
    int H = nw->hidden, H4 = H * nw->expansion;
    int B_TT = numel(grad.shape) / H;
    int N_h = B_TT * H, N_h4 = B_TT * H4;

    PrecisionTensor grad_in = grad;

    for (int i = nw->num_blocks - 1; i >= 0; i--) {
        PrecisionTensor* input_i = (i == 0) ? &a->saved_input0 : &a->block_out[i - 1];

        sv2_l2norm_dots<<<B_TT, BLOCK_SIZE, 0, stream>>>(
            a->dots.data, grad_in.data, a->block_out[i].data, B_TT, H);
        sv2_l2norm_bwd<<<grid_size(N_h), BLOCK_SIZE, 0, stream>>>(
            a->grad_a.data, grad_in.data, a->block_out[i].data,
            a->dots.data, a->block_norms[i].data, N_h, H);

        sv2_alpha_grad<<<H, BLOCK_SIZE, 0, stream>>>(
            a->agrad[i].data, a->grad_a.data, a->mlp_out[i].data,
            input_i->data, nw->alpha_fwd, B_TT, H);

        sv2_lerp_bwd<<<grid_size(N_h), BLOCK_SIZE, 0, stream>>>(
            a->grad_a.data, a->grad_b.data, a->grad_a.data,
            nw->alpha[i].data, nw->alpha_fwd, N_h, H);

        sv2_l2norm_dots<<<B_TT, BLOCK_SIZE, 0, stream>>>(
            a->dots.data, a->grad_a.data, a->mlp_out[i].data, B_TT, H);
        sv2_l2norm_bwd<<<grid_size(N_h), BLOCK_SIZE, 0, stream>>>(
            a->grad_a.data, a->grad_a.data, a->mlp_out[i].data,
            a->dots.data, a->mlp_norms[i].data, N_h, H);

        puf_mm_tn(&a->grad_a, &a->after_relu[i], &a->wgrad_w2[i], stream);
        puf_mm_nn(&a->grad_a, &nw->w2[i], &a->grad_h4, stream);

        sv2_scale_relu_grad<<<H4, BLOCK_SIZE, 0, stream>>>(
            a->sgrad[i].data, a->grad_h4.data, a->after_w1[i].data,
            nw->mlp_scaler[i].data, nw->mlp_scaler_fwd, B_TT, H4);
        sv2_scale_relu_bwd<<<grid_size(N_h4), BLOCK_SIZE, 0, stream>>>(
            a->grad_h4.data, a->grad_h4.data, a->after_w1[i].data,
            nw->mlp_scaler[i].data, nw->mlp_scaler_fwd, N_h4, H4);

        puf_mm_tn(&a->grad_h4, input_i, &a->wgrad_w1[i], stream);
        puf_mm_nn(&a->grad_h4, &nw->w1[i], &a->grad_a, stream);

        add_kernel<<<grid_size(N_h), BLOCK_SIZE, 0, stream>>>(
            a->grad_a.data, a->grad_b.data, N_h);

        grad_in = a->grad_a;
    }
    return a->grad_a;
}

static void sv2_network_init_weights(void* w, ulong* seed, cudaStream_t stream) {
    SV2NetworkWeights* nw = (SV2NetworkWeights*)w;
    int H = nw->hidden, H4 = H * nw->expansion;
    float scaler_val = sqrtf(2.0f / H) / sqrtf((float)nw->expansion);
    float alpha_val = 1.0f / sqrtf((float)H);

    for (int i = 0; i < nw->num_blocks; i++) {
        puf_normal_init(&nw->w1[i], 1.0f, (*seed)++, stream);
        puf_normal_init(&nw->w2[i], 1.0f, (*seed)++, stream);
        fill_precision_kernel<<<grid_size(H4), BLOCK_SIZE, 0, stream>>>(
            nw->mlp_scaler[i].data, from_float(scaler_val), H4);
        fill_precision_kernel<<<grid_size(H), BLOCK_SIZE, 0, stream>>>(
            nw->alpha[i].data, from_float(alpha_val), H);
    }
}

static void sv2_network_reg_params(void* w, Allocator* alloc) {
    SV2NetworkWeights* nw = (SV2NetworkWeights*)w;
    int H = nw->hidden, H4 = H * nw->expansion;
    for (int i = 0; i < nw->num_blocks; i++) {
        nw->w1[i] =         {.shape = {H4, H}};
        nw->mlp_scaler[i] = {.shape = {H4}};
        nw->w2[i] =         {.shape = {H, H4}};
        nw->alpha[i] =      {.shape = {H}};
        alloc_register(alloc, &nw->w1[i]);
        alloc_register(alloc, &nw->mlp_scaler[i]);
        alloc_register(alloc, &nw->w2[i]);
        alloc_register(alloc, &nw->alpha[i]);
    }
}

static void sv2_network_reg_train(void* w, void* activations,
        Allocator* acts, Allocator* grads, int B_TT) {
    SV2NetworkWeights* nw = (SV2NetworkWeights*)w;
    SV2NetworkActivations* a = (SV2NetworkActivations*)activations;
    int H = nw->hidden, H4 = H * nw->expansion, nb = nw->num_blocks;
    a->num_blocks = nb;

    a->after_w1 =   (PrecisionTensor*)calloc(nb, sizeof(PrecisionTensor));
    a->after_relu =  (PrecisionTensor*)calloc(nb, sizeof(PrecisionTensor));
    a->mlp_out =     (PrecisionTensor*)calloc(nb, sizeof(PrecisionTensor));
    a->mlp_norms =   (FloatTensor*)calloc(nb, sizeof(FloatTensor));
    a->block_out =   (PrecisionTensor*)calloc(nb, sizeof(PrecisionTensor));
    a->block_norms = (FloatTensor*)calloc(nb, sizeof(FloatTensor));
    a->wgrad_w1 =    (PrecisionTensor*)calloc(nb, sizeof(PrecisionTensor));
    a->sgrad =       (PrecisionTensor*)calloc(nb, sizeof(PrecisionTensor));
    a->wgrad_w2 =    (PrecisionTensor*)calloc(nb, sizeof(PrecisionTensor));
    a->agrad =       (PrecisionTensor*)calloc(nb, sizeof(PrecisionTensor));

    a->saved_input0 = {.shape = {B_TT, H}};
    a->grad_a =       {.shape = {B_TT, H}};
    a->grad_b =       {.shape = {B_TT, H}};
    a->grad_h4 =      {.shape = {B_TT, H4}};
    a->dots =          {.shape = {B_TT}};
    alloc_register(acts, &a->saved_input0);
    alloc_register(acts, &a->grad_a);
    alloc_register(acts, &a->grad_b);
    alloc_register(acts, &a->grad_h4);
    alloc_register(acts, &a->dots);

    for (int i = 0; i < nb; i++) {
        a->after_w1[i] =   {.shape = {B_TT, H4}};
        a->after_relu[i] = {.shape = {B_TT, H4}};
        a->mlp_out[i] =    {.shape = {B_TT, H}};
        a->mlp_norms[i] =  {.shape = {B_TT}};
        a->block_out[i] =  {.shape = {B_TT, H}};
        a->block_norms[i] ={.shape = {B_TT}};
        alloc_register(acts, &a->after_w1[i]);
        alloc_register(acts, &a->after_relu[i]);
        alloc_register(acts, &a->mlp_out[i]);
        alloc_register(acts, &a->mlp_norms[i]);
        alloc_register(acts, &a->block_out[i]);
        alloc_register(acts, &a->block_norms[i]);

        a->wgrad_w1[i] = {.shape = {H4, H}};
        a->sgrad[i] =    {.shape = {H4}};
        a->wgrad_w2[i] = {.shape = {H, H4}};
        a->agrad[i] =    {.shape = {H}};
        alloc_register(grads, &a->wgrad_w1[i]);
        alloc_register(grads, &a->sgrad[i]);
        alloc_register(grads, &a->wgrad_w2[i]);
        alloc_register(grads, &a->agrad[i]);
    }
}

static void sv2_network_reg_rollout(void* w, void* activations,
        Allocator* alloc, int B) {
    SV2NetworkWeights* nw = (SV2NetworkWeights*)w;
    SV2NetworkActivations* a = (SV2NetworkActivations*)activations;
    int H = nw->hidden, H4 = H * nw->expansion;
    a->num_blocks = nw->num_blocks;
    a->temp_h4 = {.shape = {B, H4}};
    a->temp_h =  {.shape = {B, H}};
    a->out =     {.shape = {B, H}};
    alloc_register(alloc, &a->temp_h4);
    alloc_register(alloc, &a->temp_h);
    alloc_register(alloc, &a->out);
}

static void* sv2_network_create_weights(void* self) {
    Network* n = (Network*)self;
    SV2NetworkWeights* nw = (SV2NetworkWeights*)calloc(1, sizeof(SV2NetworkWeights));
    nw->hidden = n->hidden;
    nw->num_blocks = n->num_layers;
    nw->horizon = n->horizon;
    nw->expansion = n->expansion;
    float scaler_init = sqrtf(2.0f / n->hidden) / sqrtf((float)n->expansion);
    float scaler_scale = scaler_init;
    nw->mlp_scaler_fwd = scaler_init / scaler_scale; // = 1.0
    float alpha_init = 1.0f / (n->num_layers + 1);
    float alpha_scale = 1.0f / sqrtf((float)n->hidden);
    nw->alpha_fwd = alpha_init / alpha_scale;
    nw->w1 =         (PrecisionTensor*)calloc(n->num_layers, sizeof(PrecisionTensor));
    nw->mlp_scaler =  (PrecisionTensor*)calloc(n->num_layers, sizeof(PrecisionTensor));
    nw->w2 =         (PrecisionTensor*)calloc(n->num_layers, sizeof(PrecisionTensor));
    nw->alpha =       (PrecisionTensor*)calloc(n->num_layers, sizeof(PrecisionTensor));
    return nw;
}

static void sv2_network_free_weights(void* w) {
    SV2NetworkWeights* nw = (SV2NetworkWeights*)w;
    free(nw->w1); free(nw->mlp_scaler); free(nw->w2); free(nw->alpha);
    free(nw);
}

static void sv2_network_free_activations(void* activations) {
    SV2NetworkActivations* a = (SV2NetworkActivations*)activations;
    free(a->after_w1); free(a->after_relu); free(a->mlp_out); free(a->mlp_norms);
    free(a->block_out); free(a->block_norms);
    free(a->wgrad_w1); free(a->sgrad); free(a->wgrad_w2); free(a->agrad);
    free(a);
}

static inline void normalize_weight(PrecisionTensor& wt, int M, int N,
        precision_t* param_base, float* master_weights, cudaStream_t stream) {
    long offset = wt.data - param_base;
    sv2_l2norm_weight_rows<<<M, BLOCK_SIZE, 0, stream>>>(master_weights + offset, M, N);
}

static void sv2_post_step(void* encoder_w, void* network_w,
        precision_t* param_base, float* master_weights, cudaStream_t stream) {
    SV2EncoderWeights* ew = (SV2EncoderWeights*)encoder_w;
    normalize_weight(ew->w, ew->out_dim, ew->in_dim + 1, param_base, master_weights, stream);
    SV2NetworkWeights* nw = (SV2NetworkWeights*)network_w;
    int H = nw->hidden, H4 = H * nw->expansion;
    for (int i = 0; i < nw->num_blocks; i++) {
        normalize_weight(nw->w1[i], H4, H, param_base, master_weights, stream);
        normalize_weight(nw->w2[i], H, H4, param_base, master_weights, stream);
    }
}

static void sv2_network_only_post_step(void* encoder_w, void* network_w,
        precision_t* param_base, float* master_weights, cudaStream_t stream) {
    SV2NetworkWeights* nw = (SV2NetworkWeights*)network_w;
    int H = nw->hidden, H4 = H * nw->expansion;
    for (int i = 0; i < nw->num_blocks; i++) {
        normalize_weight(nw->w1[i], H4, H, param_base, master_weights, stream);
        normalize_weight(nw->w2[i], H, H4, param_base, master_weights, stream);
    }
}

static void hyper_mingru_post_step(void* encoder_w, void* network_w,
        precision_t* param_base, float* master_weights, cudaStream_t stream) {
    SV2EncoderWeights* ew = (SV2EncoderWeights*)encoder_w;
    normalize_weight(ew->w, ew->out_dim, ew->in_dim + 1, param_base, master_weights, stream);
    MinGRUWeights* nw = (MinGRUWeights*)network_w;
    int H = nw->hidden;
    for (int i = 0; i < nw->num_layers; i++)
        normalize_weight(nw->weights[i], 3 * H, H, param_base, master_weights, stream);
}

struct Policy {
    Encoder encoder;
    Decoder decoder;
    Network network;
    int input_dim, hidden_dim, output_dim;
    int num_atns;
    policy_post_step_fn post_step;
};

struct PolicyActivations {
    void* encoder;
    void* decoder;
    void* network;
};

struct PolicyWeights {
    void* encoder;
    void* decoder;
    void* network;
};

static void policy_activations_free(Policy* p, PolicyActivations& a) {
    p->encoder.free_activations(a.encoder);
    p->decoder.free_activations(a.decoder);
    p->network.free_activations(a.network);
}

PrecisionTensor policy_forward(Policy* p, PolicyWeights& w, PolicyActivations& activations,
        PrecisionTensor obs, PrecisionTensor state, cudaStream_t stream) {
    PrecisionTensor enc_out = p->encoder.forward(w.encoder, activations.encoder, obs, stream);
    PrecisionTensor h = p->network.forward(w.network, enc_out, state, activations.network, stream);
    return p->decoder.forward(w.decoder, activations.decoder, h, stream);
}

PrecisionTensor policy_forward_train(Policy* p, PolicyWeights& w, PolicyActivations& activations,
        PrecisionTensor x, PrecisionTensor state, cudaStream_t stream) {
    int B = x.shape[0], TT = x.shape[1];
    PrecisionTensor h = p->encoder.forward(w.encoder, activations.encoder, *puf_squeeze(&x, 0), stream);
    h = p->network.forward_train(w.network, *puf_unsqueeze(&h, 0, B, TT), state, activations.network, stream);
    PrecisionTensor dec_out = p->decoder.forward(w.decoder, activations.decoder, *puf_squeeze(&h, 0), stream);
    return *puf_unsqueeze(&dec_out, 0, B, TT);
}

void policy_backward(Policy* p, PolicyWeights& w, PolicyActivations& activations,
        FloatTensor grad_logits, FloatTensor grad_logstd, FloatTensor grad_value, cudaStream_t stream) {
    int B = grad_logits.shape[0], TT = grad_logits.shape[1];
    PrecisionTensor grad_h = p->decoder.backward(w.decoder, activations.decoder,
        *puf_squeeze(&grad_logits, 0), grad_logstd, *puf_squeeze(&grad_value, 0), stream);
    grad_h = p->network.backward(w.network, *puf_unsqueeze(&grad_h, 0, B, TT), activations.network, stream);
    p->encoder.backward(w.encoder, activations.encoder, grad_h, stream);
}

PolicyActivations policy_reg_train(Policy* p, PolicyWeights& w,
        Allocator* acts, Allocator* grads, int B_TT) {
    PolicyActivations a;
    a.encoder = calloc(1, p->encoder.activation_size);
    a.decoder = calloc(1, p->decoder.activation_size);
    a.network = calloc(1, p->network.activation_size);
    p->encoder.reg_train(w.encoder, a.encoder, acts, grads, B_TT);
    p->decoder.reg_train(w.decoder, a.decoder, acts, grads, B_TT);
    p->network.reg_train(w.network, a.network, acts, grads, B_TT);
    return a;
}

PolicyActivations policy_reg_rollout(Policy* p, PolicyWeights& w, Allocator* acts, int B_inf) {
    PolicyActivations a;
    a.encoder = calloc(1, p->encoder.activation_size);
    a.decoder = calloc(1, p->decoder.activation_size);
    a.network = calloc(1, p->network.activation_size);
    p->encoder.reg_rollout(w.encoder, a.encoder, acts, B_inf);
    p->decoder.reg_rollout(w.decoder, a.decoder, acts, B_inf);
    p->network.reg_rollout(w.network, a.network, acts, B_inf);
    return a;
}

void policy_init_weights(Policy* p, PolicyWeights& w, uint64_t* seed, cudaStream_t stream) {
    p->encoder.init_weights(w.encoder, seed, stream);
    p->decoder.init_weights(w.decoder, seed, stream);
    p->network.init_weights(w.network, seed, stream);
}

PolicyWeights policy_weights_create(Policy* p, Allocator* params) {
    PolicyWeights w;
    w.encoder = p->encoder.create_weights(&p->encoder);
    w.decoder = p->decoder.create_weights(&p->decoder);
    w.network = p->network.create_weights(&p->network);
    p->encoder.reg_params(w.encoder, params);
    p->decoder.reg_params(w.decoder, params);
    p->network.reg_params(w.network, params);
    return w;
}

void policy_weights_free(Policy* p, PolicyWeights* w) {
    p->encoder.free_weights(w->encoder);
    p->decoder.free_weights(w->decoder);
    p->network.free_weights(w->network);
}

#endif // PUFFERLIB_MODELS_CU
