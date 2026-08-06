#ifndef PUFFERLIB_BC_CU
#define PUFFERLIB_BC_CU

// Behavior cloning (`puffer clone ENV`). Demos come from clone.data_path, or
// are collected by rolling out an expert loaded from base.load_model_path.
// The student (random init, or clone.init_model_path) then imitates them.
// Actions stay float32 like RolloutBuf (bf16 cannot represent large discrete
// action IDs exactly). The ring holds whole per-agent horizon windows so
// minibatch sampling never stitches two agents' streams together.

// (T, B, F) rollout layout -> agent-major ring: slot dst_start + b*T + t.
__global__ void bc_collect_kernel(
        precision_t* __restrict__ dst_obs, float* __restrict__ dst_act,
        precision_t* __restrict__ dst_term,
        const precision_t* __restrict__ src_obs, const float* __restrict__ src_act,
        const precision_t* __restrict__ src_term,
        int T, int B, int obs_size, int act_dim, int dst_start, int capacity) {
    int b = blockIdx.x / T;
    int t = blockIdx.x % T;
    long src_idx = (long)t * B + b;
    long dst_idx = ((long)dst_start + (long)b * T + t) % capacity;

    for (int i = threadIdx.x; i < obs_size; i += blockDim.x)
        dst_obs[dst_idx * obs_size + i] = src_obs[src_idx * obs_size + i];
    for (int i = threadIdx.x; i < act_dim; i += blockDim.x)
        dst_act[dst_idx * act_dim + i] = src_act[src_idx * act_dim + i];
    if (threadIdx.x == 0)
        dst_term[dst_idx] = src_term[src_idx];
}

__global__ void bc_gather_kernel(
        precision_t* __restrict__ dst_obs, float* __restrict__ dst_actions,
        precision_t* __restrict__ dst_terminals,
        const precision_t* __restrict__ src_obs, const float* __restrict__ src_actions,
        const precision_t* __restrict__ src_terminals,
        const int* __restrict__ start_indices,
        int obs_size, int act_dim, int horizon, int capacity) {
    int seg = blockIdx.x;
    int t = blockIdx.y;
    int src_idx = (start_indices[seg] + t) % capacity;
    long dst_offset = (long)seg * horizon + t;

    for (int i = threadIdx.x; i < obs_size; i += blockDim.x)
        dst_obs[dst_offset * obs_size + i] = src_obs[(long)src_idx * obs_size + i];
    for (int i = threadIdx.x; i < act_dim; i += blockDim.x)
        dst_actions[dst_offset * act_dim + i] = src_actions[(long)src_idx * act_dim + i];
    if (threadIdx.x == 0)
        dst_terminals[dst_offset] = src_terminals[src_idx];
}

// Continuous: Gaussian NLL vs expert actions, shared learned logstd.
// d/dmean = diff/var, d/dlogstd = 1 - diff^2/var (both mean-reduced).
__global__ void bc_nll_grad_kernel(
        float* __restrict__ grad_logits, float* __restrict__ grad_logstd,
        float* __restrict__ loss_out,
        const precision_t* __restrict__ dec_out,
        const float* __restrict__ expert_actions,
        const precision_t* __restrict__ logstd,
        int B_TT, int act_dim, int fused_cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B_TT * act_dim;
    if (idx >= total) return;

    int bt = idx / act_dim;
    int h = idx % act_dim;
    float pred = to_float(dec_out[bt * fused_cols + h]);
    float target = expert_actions[bt * act_dim + h];
    float ls = to_float(logstd[h]);
    float inv_var = expf(-2.0f * ls);
    float diff = pred - target;

    float inv_total = 1.0f / (float)total;
    grad_logits[bt * act_dim + h] = diff * inv_var * inv_total;
    grad_logstd[bt * act_dim + h] = (1.0f - diff * diff * inv_var) * inv_total;
    atomicAdd(loss_out, 0.5f * (diff * diff * inv_var + 2.0f * ls) * inv_total);
}

// Discrete: independent softmax cross-entropy per action head.
__global__ void bc_ce_grad_kernel(
        float* __restrict__ grad_logits, float* __restrict__ loss_out,
        const precision_t* __restrict__ dec_out,
        const float* __restrict__ expert_actions,
        const int* __restrict__ act_sizes,
        int B_TT, int num_heads, int fused_cols) {
    int bt = blockIdx.x * blockDim.x + threadIdx.x;
    if (bt >= B_TT) return;

    int A_total = fused_cols - 1;
    float inv_total = 1.0f / (float)B_TT;
    int logit_offset = 0;
    for (int h = 0; h < num_heads; h++) {
        int A = act_sizes[h];
        int target = (int)expert_actions[bt * num_heads + h];
        target = target < 0 ? 0 : (target >= A ? A - 1 : target);

        float max_val = -1e30f;
        for (int j = 0; j < A; j++) {
            float v = to_float(dec_out[bt * fused_cols + logit_offset + j]);
            if (v > max_val) max_val = v;
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < A; j++) {
            sum_exp += expf(to_float(dec_out[bt * fused_cols + logit_offset + j]) - max_val);
        }
        float log_sum_exp = max_val + logf(sum_exp);
        atomicAdd(loss_out, (log_sum_exp - to_float(dec_out[bt * fused_cols + logit_offset + target])) * inv_total);

        for (int j = 0; j < A; j++) {
            float prob = expf(to_float(dec_out[bt * fused_cols + logit_offset + j]) - log_sum_exp);
            grad_logits[bt * A_total + logit_offset + j] = (prob - (j == target ? 1.0f : 0.0f)) * inv_total;
        }
        logit_offset += A;
    }
}

void run_clone(Ini* ini, TrainContext* ctx) {
    // Collection reads p->rollouts directly: sync, single trainable policy.
    puf_ini_put(ini, "base.async", "0");
    puf_ini_put(ini, "vec.num_policies", "1");
    puf_ini_put(ini, "vec.hist_policy_percent", "0");
    puf_ini_put(ini, "selfplay.enabled", "0");

    char run_id[64];
    const char* configured_run_id = puf_ini_get_str(ini, "base", "run_id");
    if (!configured_run_id[0] || strcmp(configured_run_id, "None") == 0) {
        snprintf(run_id, sizeof(run_id), "%ld", (long)(1000.0 * wall_clock()));
        puf_ini_put(ini, "base.run_id", run_id);
    } else {
        snprintf(run_id, sizeof(run_id), "%s", configured_run_id);
    }
    char checkpoint_dir[2048];
    snprintf(checkpoint_dir, sizeof(checkpoint_dir), "%s/%s/%s", puf_ini_get_str(ini, "base", "checkpoint_dir"), puf_ini_get_str(ini, "base", "env_name"), run_id);
    mkdir_p(checkpoint_dir);

    long bc_steps = puf_ini_get(ini, "clone", "bc_steps");
    long capacity = puf_ini_get(ini, "clone", "capacity");
    long collect_epochs = puf_ini_get(ini, "clone", "collect_epochs");

    PuffeRL* p = create_pufferl(ini, ctx);
    Hypers* hypers = &p->hypers;
    Policy* primary = &p->policies[0];
    TrainGraph* graph = &p->train_buf;
    int horizon = hypers->horizon;
    int obs_size = OBS_SIZE;
    int act_dim = NUM_ATNS;
    int minibatch_segments = hypers->minibatch_size / horizon;
    int B_TT = minibatch_segments * horizon;
    long row_bytes = (obs_size + act_dim + 1) * sizeof(float);
    unsigned int rng = (unsigned int)p->seed;

    capacity = (capacity / horizon) * horizon;
    assert(capacity > 0 && "clone: clone.capacity must be >= train.horizon");
    precision_t* bc_obs;
    float* bc_act;
    precision_t* bc_term;
    float* d_loss;
    cudaMalloc((void**)&bc_obs, capacity * obs_size * sizeof(precision_t));
    cudaMalloc((void**)&bc_act, capacity * act_dim * sizeof(float));
    cudaMalloc((void**)&bc_term, capacity * sizeof(precision_t));
    cudaMalloc((void**)&d_loss, sizeof(float));
    long bc_count = 0;
    long write_idx = 0;

    // Snapshot the student start so expert weights used for collection don't
    // leak into training.
    const char* init_path = puf_ini_get_str(ini, "clone", "init_model_path");
    if (strcmp(init_path, "None") != 0) {
        pufferl_load_policy(p, 0, init_path);
    }
    Float master = primary->master_weights;
    long n_master = numel(master.shape);
    float* student = (float*)malloc(n_master * sizeof(float));
    cudaMemcpy(student, master.data, n_master * sizeof(float), cudaMemcpyDeviceToHost);

    // Demo file rows: [obs_size + act_dim + 1] float32, terminal last.
    const char* data_path = puf_ini_get_str(ini, "clone", "data_path");
    if (strcmp(data_path, "None") != 0) {
        FILE* f = fopen(data_path, "rb");
        assert(f && "clone: cannot open clone.data_path");
        fseek(f, 0, SEEK_END);
        long rows = ftell(f) / row_bytes;
        fseek(f, 0, SEEK_SET);
        rows = rows > capacity ? capacity : (rows / horizon) * horizon;
        assert(rows > 0 && "clone: demo file smaller than one horizon");
        float* h_rows = (float*)malloc(rows * row_bytes);
        assert(fread(h_rows, 1, rows * row_bytes, f) == (size_t)(rows * row_bytes) && "clone: short read");
        fclose(f);

        float* d_scratch;
        cudaMalloc((void**)&d_scratch, rows * obs_size * sizeof(float));
        cudaMemcpy2D(d_scratch, obs_size * sizeof(float), h_rows, row_bytes, obs_size * sizeof(float), rows, cudaMemcpyHostToDevice);
        cast<<<grid_size((int)(rows * obs_size)), BLOCK_SIZE>>>(bc_obs, d_scratch, (int)(rows * obs_size));
        cudaMemcpy2D(bc_act, act_dim * sizeof(float), h_rows + obs_size, row_bytes, act_dim * sizeof(float), rows, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        cudaMemcpy2D(d_scratch, sizeof(float), h_rows + obs_size + act_dim, row_bytes, sizeof(float), rows, cudaMemcpyHostToDevice);
        cast<<<grid_size((int)rows), BLOCK_SIZE>>>(bc_term, d_scratch, (int)rows);
        cudaDeviceSynchronize();
        cudaFree(d_scratch);
        free(h_rows);
        bc_count = rows;
        write_idx = rows % capacity;
        printf("clone: loaded %ld samples from %s\n", rows, data_path);
    } else {
        char ebuf[4096];
        const char* expert = puf_checkpoint_path_key(ini, "load_model_path", ebuf, sizeof(ebuf));
        assert(expert && "clone: set base.load_model_path (expert) or clone.data_path");
        pufferl_load_policy(p, 0, expert);
        RolloutBuf* r = &p->rollouts;
        int T = (int)r->observations.shape[0];
        int B = (int)r->observations.shape[1];
        for (long e = 0; e < collect_epochs; e++) {
            rollouts(p);
            cudaDeviceSynchronize();  // rollout buffers fill on per-buffer streams
            bc_collect_kernel<<<T * B, BLOCK_SIZE>>>(bc_obs, bc_act, bc_term, r->observations.data, r->actions.data, r->terminals.data, T, B, obs_size, act_dim, (int)write_idx, (int)capacity);
            cudaDeviceSynchronize();
            write_idx = (write_idx + (long)T * B) % capacity;
            bc_count = bc_count + (long)T * B > capacity ? capacity : bc_count + (long)T * B;
        }
        printf("clone: collected %ld expert samples\n", bc_count);
    }

    const char* save_path = puf_ini_get_str(ini, "clone", "save_data_path");
    if (strcmp(save_path, "None") != 0) {
        float* h_rows = (float*)malloc(bc_count * row_bytes);
        float* d_scratch;
        cudaMalloc((void**)&d_scratch, bc_count * obs_size * sizeof(float));
        cast<<<grid_size((int)(bc_count * obs_size)), BLOCK_SIZE>>>(d_scratch, bc_obs, (int)(bc_count * obs_size));
        cudaDeviceSynchronize();
        cudaMemcpy2D(h_rows, row_bytes, d_scratch, obs_size * sizeof(float), obs_size * sizeof(float), bc_count, cudaMemcpyDeviceToHost);
        cudaMemcpy2D(h_rows + obs_size, row_bytes, bc_act, act_dim * sizeof(float), act_dim * sizeof(float), bc_count, cudaMemcpyDeviceToHost);
        cast<<<grid_size((int)bc_count), BLOCK_SIZE>>>(d_scratch, bc_term, (int)bc_count);
        cudaDeviceSynchronize();
        cudaMemcpy2D(h_rows + obs_size + act_dim, row_bytes, d_scratch, sizeof(float), sizeof(float), bc_count, cudaMemcpyDeviceToHost);
        cudaFree(d_scratch);

        FILE* f = fopen(save_path, "wb");
        assert(f && "clone: cannot open clone.save_data_path for writing");
        fwrite(h_rows, 1, bc_count * row_bytes, f);
        fclose(f);
        free(h_rows);
        printf("clone: saved %ld samples to %s\n", bc_count, save_path);
    }

    cudaMemcpy(master.data, student, n_master * sizeof(float), cudaMemcpyHostToDevice);
    free(student);
    if (USE_BF16) {
        cast<<<grid_size((int)numel(primary->param.shape)), BLOCK_SIZE>>>(primary->param.data, master.data, (int)numel(primary->param.shape));
    }
    cudaDeviceSynchronize();

    // BC minibatch steps: gather horizon-aligned expert windows, forward with
    // stored terminals, imitation grads (value head untouched), backward, muon.
    cudaStream_t stream = p->train_stream;
    cudaMemsetAsync(p->ppo_bufs.grad_values.data, 0, numel(p->ppo_bufs.grad_values.shape) * sizeof(float), stream);
    cudaMemsetAsync(d_loss, 0, sizeof(float), stream);
    int* h_indices = (int*)malloc(minibatch_segments * sizeof(int));
    long num_windows = bc_count / horizon;
    for (long step = 0; step < bc_steps; step++) {
        for (int i = 0; i < minibatch_segments; i++) {
            h_indices[i] = (int)(rand_r(&rng) % (unsigned int)num_windows) * horizon;
        }
        cudaMemcpyAsync(p->prio_bufs.idx.data, h_indices, minibatch_segments * sizeof(int), cudaMemcpyHostToDevice, stream);
        dim3 grid_gather(minibatch_segments, horizon);
        bc_gather_kernel<<<grid_gather, BLOCK_SIZE, 0, stream>>>(graph->mb_obs.data, graph->mb_actions.data, graph->mb_terminals.data, bc_obs, bc_act, bc_term, p->prio_bufs.idx.data, obs_size, act_dim, horizon, (int)capacity);

        cudaMemsetAsync(graph->mb_state.data, 0, numel(graph->mb_state.shape) * sizeof(precision_t), stream);
        Prec dec = arch_forward_train(&primary->arch, primary->weights, p->train_activs, graph->mb_obs, graph->mb_state, graph->mb_terminals, stream);
        int fused_cols = (int)dec.shape[2];

        // dec is (B, TT, fused) contiguous; kernels index it as (B_TT, fused).
        Float grad_logstd = p->is_continuous ? p->ppo_bufs.grad_logstd : Float();
        if (p->is_continuous) {
            DecoderWeights* dw = (DecoderWeights*)primary->weights.decoder;
            bc_nll_grad_kernel<<<grid_size(B_TT * act_dim), BLOCK_SIZE, 0, stream>>>(p->ppo_bufs.grad_logits.data, grad_logstd.data, d_loss, dec.data, graph->mb_actions.data, dw->logstd.data, B_TT, act_dim, fused_cols);
        } else {
            bc_ce_grad_kernel<<<grid_size(B_TT), BLOCK_SIZE, 0, stream>>>(p->ppo_bufs.grad_logits.data, d_loss, dec.data, graph->mb_actions.data, p->act_sizes, B_TT, act_dim, fused_cols);
        }

        arch_backward(&primary->arch, primary->weights, p->train_activs, p->ppo_bufs.grad_logits, grad_logstd, p->ppo_bufs.grad_values, stream);
        muon_step(&p->muon, primary->master_weights, p->grad, hypers->max_grad_norm, stream);
        if (USE_BF16) {
            cast<<<grid_size((int)numel(primary->param.shape)), BLOCK_SIZE, 0, stream>>>(primary->param.data, master.data, (int)numel(primary->param.shape));
        }

        if ((step + 1) % 100 == 0 || step + 1 == bc_steps) {
            float loss;
            cudaMemcpyAsync(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            printf("clone: step %ld/%ld loss %f\n", step + 1, bc_steps, loss / (float)(step % 100 + 1));
            cudaMemsetAsync(d_loss, 0, sizeof(float), stream);
        }
    }
    free(h_indices);

    char out_path[4096];
    snprintf(out_path, sizeof(out_path), "%s/%016ld.bin", checkpoint_dir, p->global_step);
    puf_save_weights(p, out_path);
    printf("clone: saved %s\n", out_path);

    cudaFree(bc_obs);
    cudaFree(bc_act);
    cudaFree(bc_term);
    cudaFree(d_loss);
    close_pufferl(p);
}

#endif // PUFFERLIB_BC_CU
