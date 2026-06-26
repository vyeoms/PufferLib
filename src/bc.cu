#ifndef PUFFERLIB_BC_CU
#define PUFFERLIB_BC_CU

#include <cstdlib>
#include <cstdio>

struct BCBuffer {
    precision_t* obs;
    precision_t* actions;
    int capacity;
    int count;
    int write_idx;
    int obs_size;
    int act_dim;
};

static BCBuffer* bc_buffer_create(int capacity, int obs_size, int act_dim) {
    BCBuffer* bc = (BCBuffer*)calloc(1, sizeof(BCBuffer));
    bc->capacity = capacity;
    bc->obs_size = obs_size;
    bc->act_dim = act_dim;
    bc->count = 0;
    bc->write_idx = 0;
    cudaMalloc(&bc->obs, (long)capacity * obs_size * sizeof(precision_t));
    cudaMalloc(&bc->actions, (long)capacity * act_dim * sizeof(precision_t));
    cudaMemset(bc->obs, 0, (long)capacity * obs_size * sizeof(precision_t));
    cudaMemset(bc->actions, 0, (long)capacity * act_dim * sizeof(precision_t));
    return bc;
}

static void bc_buffer_free(BCBuffer* bc) {
    if (bc->obs) cudaFree(bc->obs);
    if (bc->actions) cudaFree(bc->actions);
    free(bc);
}

static int bc_buffer_load(BCBuffer* bc, const char* path, cudaStream_t stream) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "bc_buffer_load: cannot open %s\n", path);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    int row_floats = bc->obs_size + bc->act_dim;
    long row_bytes = row_floats * sizeof(float);
    long num_rows = file_size / row_bytes;
    if (num_rows <= 0 || file_size % row_bytes != 0) {
        fprintf(stderr, "bc_buffer_load: invalid file size %ld (row=%ld bytes)\n",
            file_size, row_bytes);
        fclose(f);
        return -1;
    }
    if (num_rows > bc->capacity) num_rows = bc->capacity;

    float* host_buf = (float*)malloc(num_rows * row_bytes);
    size_t nread = fread(host_buf, 1, num_rows * row_bytes, f);
    fclose(f);
    if ((long)nread != num_rows * row_bytes) {
        fprintf(stderr, "bc_buffer_load: short read\n");
        free(host_buf);
        return -1;
    }

    float* host_obs = (float*)malloc(num_rows * bc->obs_size * sizeof(float));
    float* host_act = (float*)malloc(num_rows * bc->act_dim * sizeof(float));
    for (long i = 0; i < num_rows; i++) {
        memcpy(host_obs + i * bc->obs_size,
            host_buf + i * row_floats, bc->obs_size * sizeof(float));
        memcpy(host_act + i * bc->act_dim,
            host_buf + i * row_floats + bc->obs_size, bc->act_dim * sizeof(float));
    }
    free(host_buf);

    float* d_obs_f;
    float* d_act_f;
    cudaMalloc(&d_obs_f, num_rows * bc->obs_size * sizeof(float));
    cudaMalloc(&d_act_f, num_rows * bc->act_dim * sizeof(float));
    cudaMemcpyAsync(d_obs_f, host_obs, num_rows * bc->obs_size * sizeof(float),
        cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_act_f, host_act, num_rows * bc->act_dim * sizeof(float),
        cudaMemcpyHostToDevice, stream);
    free(host_obs);
    free(host_act);

    int n_obs = (int)(num_rows * bc->obs_size);
    int n_act = (int)(num_rows * bc->act_dim);
    cast<<<grid_size(n_obs), BLOCK_SIZE, 0, stream>>>(bc->obs, d_obs_f, n_obs);
    cast<<<grid_size(n_act), BLOCK_SIZE, 0, stream>>>(bc->actions, d_act_f, n_act);
    cudaStreamSynchronize(stream);
    cudaFree(d_obs_f);
    cudaFree(d_act_f);

    bc->count = (int)num_rows;
    bc->write_idx = (int)(num_rows % bc->capacity);
    printf("bc_buffer_load: loaded %ld samples from %s\n", num_rows, path);
    return (int)num_rows;
}

static int bc_buffer_save(BCBuffer* bc, const char* path, cudaStream_t stream) {
    if (bc->count <= 0) return 0;

    long n_obs = (long)bc->count * bc->obs_size;
    long n_act = (long)bc->count * bc->act_dim;

    float* h_obs = (float*)malloc(n_obs * sizeof(float));
    float* h_act = (float*)malloc(n_act * sizeof(float));

    if (USE_BF16) {
        float* d_obs_f;
        float* d_act_f;
        cudaMalloc(&d_obs_f, n_obs * sizeof(float));
        cudaMalloc(&d_act_f, n_act * sizeof(float));
        cast<<<grid_size((int)n_obs), BLOCK_SIZE, 0, stream>>>(d_obs_f, bc->obs, (int)n_obs);
        cast<<<grid_size((int)n_act), BLOCK_SIZE, 0, stream>>>(d_act_f, bc->actions, (int)n_act);
        cudaStreamSynchronize(stream);
        cudaMemcpy(h_obs, d_obs_f, n_obs * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_act, d_act_f, n_act * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_obs_f);
        cudaFree(d_act_f);
    } else {
        cudaMemcpy(h_obs, bc->obs, n_obs * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_act, bc->actions, n_act * sizeof(float), cudaMemcpyDeviceToHost);
    }

    int row_floats = bc->obs_size + bc->act_dim;
    float* h_buf = (float*)malloc((long)bc->count * row_floats * sizeof(float));
    for (int i = 0; i < bc->count; i++) {
        memcpy(h_buf + (long)i * row_floats,
            h_obs + (long)i * bc->obs_size, bc->obs_size * sizeof(float));
        memcpy(h_buf + (long)i * row_floats + bc->obs_size,
            h_act + (long)i * bc->act_dim, bc->act_dim * sizeof(float));
    }
    free(h_obs);
    free(h_act);

    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "bc_buffer_save: cannot open %s for writing\n", path);
        free(h_buf);
        return -1;
    }
    fwrite(h_buf, sizeof(float), (long)bc->count * row_floats, f);
    fclose(f);
    free(h_buf);

    printf("bc_buffer_save: saved %d samples to %s\n", bc->count, path);
    return bc->count;
}

__global__ void bc_collect_kernel(
        precision_t* __restrict__ dst,
        const precision_t* __restrict__ src,
        int T, int B, int feature_size,
        int dst_start, int capacity) {
    int bt = blockIdx.x;
    int b = bt / T;
    int t = bt % T;
    int tid = threadIdx.x;

    long src_idx = (long)t * B + b;
    long dst_idx = ((long)dst_start + (long)b * T + t) % capacity;

    for (int i = tid; i < feature_size; i += blockDim.x)
        dst[dst_idx * feature_size + i] = src[src_idx * feature_size + i];
}

__global__ void bc_gather_kernel(
        precision_t* __restrict__ dst_obs,
        precision_t* __restrict__ dst_actions,
        const precision_t* __restrict__ src_obs,
        const precision_t* __restrict__ src_actions,
        const int* __restrict__ start_indices,
        int obs_size, int act_dim, int horizon, int capacity) {
    int seg = blockIdx.x;
    int t = blockIdx.y;
    int tid = threadIdx.x;

    int src_idx = (start_indices[seg] + t) % capacity;
    long dst_offset = ((long)seg * horizon + t);

    for (int i = tid; i < obs_size; i += blockDim.x)
        dst_obs[dst_offset * obs_size + i] = src_obs[(long)src_idx * obs_size + i];

    for (int i = tid; i < act_dim; i += blockDim.x)
        dst_actions[dst_offset * act_dim + i] = src_actions[(long)src_idx * act_dim + i];
}

__global__ void bc_mse_grad_kernel(
        float* __restrict__ grad_logits,
        float* __restrict__ loss_out,
        const precision_t* __restrict__ dec_out,
        const precision_t* __restrict__ expert_actions,
        int B_TT, int act_dim, int fused_cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B_TT * act_dim;
    if (idx >= total) return;

    int bt = idx / act_dim;
    int h = idx % act_dim;

    float pred = to_float(dec_out[bt * fused_cols + h]);
    float target = to_float(expert_actions[bt * act_dim + h]);
    float diff = pred - target;

    float inv_total = 1.0f / (float)total;
    grad_logits[bt * act_dim + h] = 2.0f * diff * inv_total;
    atomicAdd(loss_out, diff * diff * inv_total);
}

__global__ void bc_nll_grad_kernel(
        float* __restrict__ grad_logits,
        float* __restrict__ grad_logstd,
        float* __restrict__ loss_out,
        const precision_t* __restrict__ dec_out,
        const precision_t* __restrict__ expert_actions,
        const precision_t* __restrict__ logstd,
        int B_TT, int act_dim, int fused_cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B_TT * act_dim;
    if (idx >= total) return;

    int bt = idx / act_dim;
    int h = idx % act_dim;

    float pred = to_float(dec_out[bt * fused_cols + h]);
    float target = to_float(expert_actions[bt * act_dim + h]);
    float ls = to_float(logstd[h]);
    float var = expf(2.0f * ls);
    float inv_var = 1.0f / var;
    float diff = pred - target;

    float inv_total = 1.0f / (float)total;
    grad_logits[bt * act_dim + h] = diff * inv_var * inv_total;
    grad_logstd[bt * act_dim + h] = (1.0f - diff * diff * inv_var) * inv_total;
    atomicAdd(loss_out, 0.5f * (diff * diff * inv_var + 2.0f * ls) * inv_total);
}

__global__ void bc_ce_grad_kernel(
        float* __restrict__ grad_logits,
        float* __restrict__ loss_out,
        const precision_t* __restrict__ dec_out,
        const precision_t* __restrict__ expert_actions,
        const int* __restrict__ act_sizes,
        int B_TT, int num_heads, int fused_cols) {
    int bt = blockIdx.x * blockDim.x + threadIdx.x;
    if (bt >= B_TT) return;

    int A_total = fused_cols - 1;
    float inv_total = 1.0f / (float)B_TT;
    int logit_offset = 0;
    for (int h = 0; h < num_heads; h++) {
        int A = act_sizes[h];
        int target = (int)to_float(expert_actions[bt * num_heads + h]);
        if (target < 0) target = 0;
        if (target >= A) target = A - 1;

        float max_val = -1e30f;
        for (int j = 0; j < A; j++) {
            float v = to_float(dec_out[bt * fused_cols + logit_offset + j]);
            if (v > max_val) max_val = v;
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < A; j++) {
            float v = to_float(dec_out[bt * fused_cols + logit_offset + j]);
            sum_exp += expf(v - max_val);
        }
        float log_sum_exp = max_val + logf(sum_exp);

        float target_logit = to_float(dec_out[bt * fused_cols + logit_offset + target]);
        atomicAdd(loss_out, (log_sum_exp - target_logit) * inv_total);

        for (int j = 0; j < A; j++) {
            float v = to_float(dec_out[bt * fused_cols + logit_offset + j]);
            float p = expf(v - log_sum_exp);
            float grad = (p - (j == target ? 1.0f : 0.0f)) * inv_total;
            grad_logits[bt * A_total + logit_offset + j] = grad;
        }
        logit_offset += A;
    }
}

#endif // PUFFERLIB_BC_CU
