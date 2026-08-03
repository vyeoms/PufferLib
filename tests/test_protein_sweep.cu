// test_protein_sweep.cu -- End-to-end sweep test for protein.cu public API
//
// Build:
//   nvcc -o test_protein_sweep tests/test_protein_sweep.cu -I src/ -lcublas -lcusolver -lcurand

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BLOCK_SIZE 256

__device__ __forceinline__ void block_reduce_sum(
        float *smem, float *out, int tid, int nthreads, int nchan) {
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int c = 0; c < nchan; c++) {
                smem[c * nthreads + tid] += smem[c * nthreads + tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        for (int c = 0; c < nchan; c++) {
            out[c] = smem[c * nthreads];
        }
    }
}

#include "protein.cu"

#define DIM 21
#define COST_IDX 0
#define LR_IDX 2

static SweepSpace *build_default_search_space(void) {
    SweepSpace *hp = (SweepSpace *)calloc(1, sizeof(SweepSpace));
    hp->spaces = (Space *)calloc(DIM, sizeof(Space));
    hp->num = DIM;
    hp->cost_idx = COST_IDX;
    hp->optimize_direction = 1;

    float ts_scale = 1.0f / (log2f(1e11f) - log2f(3e7f));
    hp->spaces[0]  = (Space){SPACE_LOG,    3e7f,    1e11f,    ts_scale, 0};
    hp->spaces[1]  = (Space){SPACE_POW2,   8.0f,    1024.0f,  0.5f, 1};
    hp->spaces[2]  = (Space){SPACE_LOG,    1e-5f,   0.1f,     0.5f, 0};
    hp->spaces[3]  = (Space){SPACE_LOG,    1e-5f,   0.2f,     0.5f, 0};
    hp->spaces[4]  = (Space){SPACE_LOGIT,  0.8f,    0.9999f,  0.5f, 0};
    hp->spaces[5]  = (Space){SPACE_LOGIT,  0.2f,    0.995f,   0.5f, 0};
    hp->spaces[6]  = (Space){SPACE_LINEAR, 0.1f,    5.0f,     0.5f, 0};
    hp->spaces[7]  = (Space){SPACE_LINEAR, 0.1f,    5.0f,     0.5f, 0};
    hp->spaces[8]  = (Space){SPACE_LINEAR, 0.25f,   4.0f,     0.5f, 0};
    hp->spaces[9]  = (Space){SPACE_LINEAR, 0.01f,   1.0f,     0.5f, 0};
    hp->spaces[10] = (Space){SPACE_LINEAR, 0.01f,   5.0f,     0.5f, 0};
    hp->spaces[11] = (Space){SPACE_LINEAR, 0.1f,    5.0f,     0.5f, 0};
    hp->spaces[12] = (Space){SPACE_LINEAR, 0.1f,    5.0f,     0.5f, 0};
    hp->spaces[13] = (Space){SPACE_LOGIT,  0.5f,    0.999f,   0.5f, 0};
    hp->spaces[14] = (Space){SPACE_LOGIT,  0.9f,    0.99999f, 0.5f, 0};
    hp->spaces[15] = (Space){SPACE_LOG,    1e-14f,  1e-4f,    0.5f, 0};
    hp->spaces[16] = (Space){SPACE_LINEAR, 0.0f,    1.0f,     0.5f, 0};
    hp->spaces[17] = (Space){SPACE_LINEAR, 0.0f,    1.0f,     0.5f, 0};
    hp->spaces[18] = (Space){SPACE_POW2,   32.0f,   1024.0f,  0.5f, 1};
    hp->spaces[19] = (Space){SPACE_LINEAR, 1.0f,    8.0f,     0.5f, 0};
    hp->spaces[20] = (Space){SPACE_LINEAR, 1.0f,    8.0f,     0.5f, 0};
    return hp;
}

static void synthetic_linear(float learning_rate, float total_timesteps,
        float *out_score, float *out_cost) {
    float basic = expf(-powf(log10f(learning_rate) + 3.0f, 2.0f));
    *out_cost = total_timesteps / 5e7f;
    *out_score = basic * (*out_cost);
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    srand((unsigned)time(NULL));
    printf("=== Protein Sweep Test (synthetic linear, 21D) ===\n\n");

    SweepSpace *hp = build_default_search_space();

    int num_runs = 200;
    int downsample = 5;
    int max_obs = num_runs * downsample + 10;

    ProteinSweep *sw = protein_sweep_create((ProteinSweep){
        .space = hp,
        .num_random_samples = 10,
        .suggestions_per_pareto = 256,
        .gp_training_iter = 50,
        .gp_learning_rate = 0.001f,
        .optimizer_reset_frequency = 50,
        .gp_max_obs = 750,
        .infer_batch_size = 4096,
        .use_success_prob = 0,
        .prune_pareto = 1,
        .use_logit = 0,
        .global_search_scale = 1.0f,
        .max_suggestion_cost = 3600.0f,
        .expansion_rate = 1.0f,
        .cost_random_suggestion = -0.8f,
        .early_stop_quantile = 0.3f,
        .success_cap = max_obs,
        .failure_cap = max_obs / 10,
        .top_k = 5,
        .rng_seed = 42ULL,
    });

    float suggestion[DIM];
    float random_best = -FLT_MAX, guided_best = -FLT_MAX;

    for (int run = 0; run < num_runs; run++) {
        ProteinSweepInfo info = protein_sweep_suggest(sw, suggestion, NAN);

        for (int ds = 1; ds <= downsample; ds++) {
            float real_lr = space_unnormalize(&hp->spaces[LR_IDX], suggestion[LR_IDX]);
            float real_ts = space_unnormalize(&hp->spaces[COST_IDX], suggestion[COST_IDX]);
            float frac_ts = real_ts * (float)ds / (float)downsample;

            float score, cost;
            synthetic_linear(real_lr, frac_ts, &score, &cost);

            float obs_p[DIM];
            memcpy(obs_p, suggestion, DIM * sizeof(float));
            obs_p[COST_IDX] = space_normalize(&hp->spaces[COST_IDX], frac_ts);
            protein_sweep_observe(sw, obs_p, score, cost, 0);

            if (ds == downsample) {
                if (info.is_random) {
                    if (score > random_best) random_best = score;
                } else {
                    if (score > guided_best) guided_best = score;
                }
                if (run % 50 == 0 || run == num_runs - 1) {
                    printf("  [%3d] %s score=%.4f cost=%.2f "
                           "pareto=%d gp_obs=%d cands=%d\n",
                        run, info.is_random ? "rand" : "  GP",
                        score, cost, info.n_pareto, info.n_gp_obs, info.n_candidates);
                }
            }
        }
    }

    printf("\n=== Results ===\n");
    printf("  Random best:     %.4f\n", random_best);
    printf("  GP-guided best:  %.4f\n", guided_best);

    int pass = (guided_best >= random_best * 0.9f);
    printf("\n=== %s ===\n", pass ? "PASS" : "FAIL");

    protein_sweep_destroy(sw);
    free(hp->spaces);
    free(hp);

    // Leak check: create/destroy cycle should not accumulate GPU memory
    size_t free_before, free_after, total;
    cudaMemGetInfo(&free_before, &total);
    for (int i = 0; i < 5; i++) {
        SweepSpace *hp2 = build_default_search_space();
        ProteinSweep *sw2 = protein_sweep_create((ProteinSweep){
            .space = hp2,
            .num_random_samples = 5,
            .suggestions_per_pareto = 64,
            .gp_training_iter = 10,
            .gp_learning_rate = 0.001f,
            .optimizer_reset_frequency = 0,
            .gp_max_obs = 100,
            .infer_batch_size = 256,
            .success_cap = 200,
            .failure_cap = 50,
            .top_k = 3,
            .rng_seed = 42ULL + i,
        });
        float buf[DIM];
        protein_sweep_suggest(sw2, buf, NAN);
        protein_sweep_destroy(sw2);
        free(hp2->spaces);
        free(hp2);
    }
    cudaMemGetInfo(&free_after, &total);
    long leaked_mb = ((long)free_before - (long)free_after) / (1024 * 1024);
    printf("\n  Leak check: %ld MB leaked across 5 create/destroy cycles\n", leaked_mb);
    if (leaked_mb > 2) {
        printf("  FAIL: significant GPU memory leak detected\n");
        pass = 0;
    } else {
        printf("  PASS: no significant GPU memory leak\n");
    }

    return pass ? 0 : 1;
}
