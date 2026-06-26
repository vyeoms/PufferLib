#ifndef PUFFERLIB_BC_TRAIN_CU
#define PUFFERLIB_BC_TRAIN_CU

static void bc_collect(BCBuffer* bc, RolloutBuf& rollouts,
        int horizon, int total_agents, cudaStream_t stream) {
    long num_samples = (long)horizon * total_agents;
    int wi = bc->write_idx;

    int num_blocks = horizon * total_agents;
    bc_collect_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        bc->obs, rollouts.observations.data,
        horizon, total_agents, bc->obs_size, wi, bc->capacity);
    bc_collect_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        bc->actions, rollouts.actions.data,
        horizon, total_agents, bc->act_dim, wi, bc->capacity);

    bc->write_idx = (int)((wi + num_samples) % bc->capacity);
    bc->count = (bc->count + (int)num_samples > bc->capacity)
        ? bc->capacity : bc->count + (int)num_samples;
}

static float bc_train_step_gaussian(PuffeRL& pufferl, BCBuffer& bc,
        int* h_indices, cudaStream_t stream) {
    HypersT& hypers = pufferl.hypers;
    int minibatch_segments = hypers.minibatch_size / hypers.horizon;
    int horizon = hypers.horizon;
    int B_TT = minibatch_segments * horizon;
    TrainGraph& graph = pufferl.train_buf;

    int max_start = bc.count - horizon;
    if (max_start < 1) max_start = 1;
    for (int i = 0; i < minibatch_segments; i++)
        h_indices[i] = (int)(rand() % max_start);

    cudaMemcpyAsync(pufferl.prio_bufs.idx.data, h_indices,
        minibatch_segments * sizeof(int), cudaMemcpyHostToDevice, stream);

    dim3 grid_gather(minibatch_segments, horizon);
    bc_gather_kernel<<<grid_gather, BLOCK_SIZE, 0, stream>>>(
        graph.mb_obs.data, graph.mb_actions.data,
        bc.obs, bc.actions,
        pufferl.prio_bufs.idx.data,
        bc.obs_size, bc.act_dim, horizon, bc.capacity);

    puf_zero(&graph.mb_state, stream);
    PrecisionTensor obs_puf = graph.mb_obs;
    PrecisionTensor state_puf = graph.mb_state;
    PrecisionTensor dec_puf = policy_forward_train(
        &pufferl.policy, pufferl.weights, pufferl.train_activations,
        obs_puf, state_puf, stream);

    int fused_cols = dec_puf.shape[2];

    FloatTensor grad_logits = pufferl.ppo_bufs_puf.grad_logits;
    FloatTensor loss_scalar = pufferl.ppo_bufs_puf.loss_output;
    cudaMemsetAsync(loss_scalar.data, 0, sizeof(float), stream);
    cudaMemsetAsync(grad_logits.data, 0,
        numel(grad_logits.shape) * sizeof(float), stream);

    PrecisionTensor dec_flat = dec_puf;
    puf_squeeze(&dec_flat, 0);

    FloatTensor grad_values = pufferl.ppo_bufs_puf.grad_values;
    cudaMemsetAsync(grad_values.data, 0,
        numel(grad_values.shape) * sizeof(float), stream);

    FloatTensor grad_logstd = pufferl.is_continuous
        ? pufferl.ppo_bufs_puf.grad_logstd : FloatTensor();

    if (pufferl.is_continuous) {
        int act_dim = pufferl.policy.num_atns;
        int total_elems = B_TT * act_dim;
        DecoderWeights* dw = (DecoderWeights*)pufferl.weights.decoder;
        cudaMemsetAsync(grad_logstd.data, 0,
            numel(grad_logstd.shape) * sizeof(float), stream);
        bc_nll_grad_kernel<<<grid_size(total_elems), BLOCK_SIZE, 0, stream>>>(
            grad_logits.data, grad_logstd.data, loss_scalar.data,
            dec_flat.data, graph.mb_actions.data, dw->logstd.data,
            B_TT, act_dim, fused_cols);
    } else {
        int num_heads = numel(pufferl.act_sizes_puf.shape);
        bc_ce_grad_kernel<<<grid_size(B_TT), BLOCK_SIZE, 0, stream>>>(
            grad_logits.data, loss_scalar.data,
            dec_flat.data, graph.mb_actions.data,
            pufferl.act_sizes_puf.data,
            B_TT, num_heads, fused_cols);
        if (grad_logstd.data)
            cudaMemsetAsync(grad_logstd.data, 0,
                numel(grad_logstd.shape) * sizeof(float), stream);
    }

    float loss_val = 0.0f;
    cudaMemcpyAsync(&loss_val, loss_scalar.data, sizeof(float),
        cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    policy_backward(&pufferl.policy, pufferl.weights, pufferl.train_activations,
        grad_logits, grad_logstd, grad_values, stream);

    muon_step(&pufferl.muon, pufferl.master_weights, pufferl.grad_puf,
        hypers.max_grad_norm, stream);
    if (USE_BF16) {
        int n = numel(pufferl.param_puf.shape);
        cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
            pufferl.param_puf.data, pufferl.master_weights.data, n);
    }

    return loss_val;
}

static float bc_train(PuffeRL& pufferl, BCBuffer& bc, int num_steps) {
    if (bc.count < pufferl.hypers.horizon) {
        fprintf(stderr, "bc_train: need at least %d samples (have %d)\n",
            pufferl.hypers.horizon, bc.count);
        return -1.0f;
    }

    int minibatch_segments = pufferl.hypers.minibatch_size / pufferl.hypers.horizon;
    int* h_indices = (int*)malloc(minibatch_segments * sizeof(int));
    cudaStream_t stream = pufferl.default_stream;

    float total_loss = 0.0f;
    for (int step = 0; step < num_steps; step++) {
        float loss = bc_train_step_gaussian(pufferl, bc, h_indices, stream);
        total_loss += loss;
    }

    free(h_indices);
    return total_loss / (float)num_steps;
}

#endif // PUFFERLIB_BC_TRAIN_CU
