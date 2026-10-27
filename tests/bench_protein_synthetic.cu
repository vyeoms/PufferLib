// bench_protein_synthetic.cu -- Sweep-quality benchmark for protein.cu on synthetic tasks
//
// C port of the tasks in tests/test_sweep.py (linear / log / percentile cost-score
// relationships, 20% cost noise on log/percentile), lifted to the 16-dim default.ini
// sweep space. Only a few dims matter (one interaction); the rest are distractors.
// Protein is configured as in run_sweep (pufferl.cu).
//
// Metric: frontier regret. With the noiseless objective f(u, c) and the best achievable
// frontier opt(c) = f(u*, c), regret = mean over a log cost grid of
// (opt(c) - best true score of any run with cost <= c) / opt(c). Lower is better.
//
// Build:
//   nvcc -O2 -o bench_protein_synthetic tests/bench_protein_synthetic.cu \
//        -I src/ -lcublas -lcusolver -lcurand
// Point -I at a directory holding another protein.cu to A/B versions.
//
// Usage:
//   ./bench_protein_synthetic [--task linear|log|percentile|all] [--seeds N] [--runs N]
//                             [--parallel P] [--curve]
//   --parallel P: suggest P runs before observing any (pufferl runs trials concurrently)
//   --curve: observe each learning-curve point at its own timestep params instead of
//            reusing the run's sample (pufferl currently reuses it; observe dedupes)
//   --expansion E, --cost-init C: expansion_rate and cost_random_suggestion overrides
//   --cost-scale S: candidate perturbation scale of the cost dim (default.ini "time" = ~0.085;
//                   breakout.ini "auto" = 0.5)
//   --diag: per run-bin means of score-GP params (sigma_f, other non-lengthscale params,
//           cost lengthscale, noise), dmu_score/dmu_cost = change in each GP mean when the
//           best run's normalized cost is raised by DIAG_STEP, x*_cost = that run's
//           normalized cost, pick_cost = suggested normalized cost, dmu_lin = the linear
//           kernel term's share of dmu_score (exact; the rest is the Matern term), dsig =
//           change in the score GP's predictive std there (variants with gp_predict_var),
//           dsig_other = same for a DIAG_STEP move along DIAG_OTHER_DIM instead of cost,
//           ell_q / ell_dis = lengthscale geo-mean over quality / distractor dims, rho_q =
//           Spearman(mu, true quality) over N_PROBE probes around the best run at its cost,
//           floor = score-GP noise floor (measured from replicates when --replicates)
//   --log-score: observe log(score) (regret still on true scores)
//   --noise-prior W: score-GP noise prior log-sd shrunk by sqrt(1 + W)
//   --score-noise S: observed score *= 1 + S * N(0,1), one draw per run (seed noise);
//                    regret stays on true scores
//   --score-noise-add A: observed score += A * opt(top of cost grid) * N(0,1), one draw per
//                        run (constant-variance seed noise)
//   --replicates R: Protein re-runs the first run R times to measure the score noise
//   --fisher N, --noise-floor F: Fisher steps and score-GP noise floor (only with a protein.cu
//           that defines GP_FISHER_DAMP); --fisher-hutch M: 0 exact, 1 Hutchinson Fisher matrix
//           + exact gradient, 2 Hutchinson for both; --ucb B, --noise-prior W: UCB beta and noise-prior
//           weight (GP_LAPLACE_RIDGE)

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

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

#define DIM 16
#define COST_IDX 0
#define DOWNSAMPLE 5
#define COST_UNIT 3e7f
#define N_GRID 32
#define N_BAND 4
#define N_CHECK 4
#define MAX_PAR 64
#define MAX_RUNS 1024
#define MAX_SEEDS 64
#define N_DIAG 16
#define N_PROBE 256
#ifndef DIAG_OTHER_DIM
#define DIAG_OTHER_DIM 12
#endif
#define DIAG_STEP 0.25f

enum { TASK_LINEAR, TASK_LOG, TASK_PERCENTILE, N_TASKS };
static const char *TASK_NAMES[N_TASKS] = {"linear", "log", "percentile"};

static float g_cost_scale = 0.0f;

// default.ini sweep space, total_timesteps first
static void make_space(SweepSpace *sp) {
    static Space s[DIM];
    float t_scale = 1.0f / (log2f(1e11f) - log2f(3e7f));
    s[0] = (Space){.type = SPACE_LOG, .min = 3e7f, .max = 1e11f, .scale = t_scale};
    s[1] = (Space){.type = SPACE_POW2, .min = 32, .max = 1024, .scale = 0.5f, .is_integer = 1};
    s[2] = (Space){.type = SPACE_LINEAR, .min = 1, .max = 8, .scale = 0.5f};
    s[3] = (Space){.type = SPACE_POW2, .min = 256, .max = 16384, .scale = 0.5f, .is_integer = 1};
    s[4] = (Space){.type = SPACE_POW2, .min = 32, .max = 512, .scale = 0.5f, .is_integer = 1};
    s[5] = (Space){.type = SPACE_POW2, .min = 4096, .max = 65536, .scale = 0.5f, .is_integer = 1};
    s[6] = (Space){.type = SPACE_LOG, .min = 1e-4f, .max = 1.0f, .scale = 0.5f};
    s[7] = (Space){.type = SPACE_LOG, .min = 1e-5f, .max = 0.1f, .scale = 0.5f};
    s[8] = (Space){.type = SPACE_LOGIT, .min = 0.8f, .max = 0.9999f, .scale = 0.5f};
    s[9] = (Space){.type = SPACE_LOGIT, .min = 0.2f, .max = 0.995f, .scale = 0.5f};
    s[10] = (Space){.type = SPACE_LINEAR, .min = 0.25f, .max = 4.0f, .scale = 0.5f};
    s[11] = (Space){.type = SPACE_LINEAR, .min = 0.1f, .max = 0.9f, .scale = 0.5f};
    s[12] = (Space){.type = SPACE_LINEAR, .min = 0.01f, .max = 5.0f, .scale = 0.5f};
    s[13] = (Space){.type = SPACE_LINEAR, .min = 0.1f, .max = 4.0f, .scale = 0.5f};
    s[14] = (Space){.type = SPACE_LINEAR, .min = 0.1f, .max = 1.5f, .scale = 0.5f};
    s[15] = (Space){.type = SPACE_LOGIT, .min = 0.5f, .max = 0.999f, .scale = 0.5f};
    if (g_cost_scale > 0.0f) {
        s[COST_IDX].scale = g_cost_scale;
    }
    *sp = (SweepSpace){.spaces = s, .num = DIM, .cost_idx = COST_IDX, .optimize_direction = 1};
}

// Normalized-space quality in (0, 1]. lr (6) keeps test_sweep.py's
// exp(-(log10(lr)+3)^2), with its optimum shifted by minibatch size (5). gamma (8),
// gae_lambda (9), ent_coef (7) and hidden_size (1) matter less; the rest are distractors.
static float base_quality(const float *u) {
    float log10_lr = -4.0f + 2.0f * (u[6] + 1.0f);
    float lr_opt = -3.0f + 0.5f * u[5];
    float e = (log10_lr - lr_opt) * (log10_lr - lr_opt);
    e += 2.0f * (u[8] - 0.4f) * (u[8] - 0.4f);
    e += 1.0f * (u[9] - 0.5f) * (u[9] - 0.5f);
    e += 0.5f * (u[7] + 0.3f) * (u[7] + 0.3f);
    e += 0.25f * (u[1] - 0.2f) * (u[1] - 0.2f);
    return expf(-e);
}

static float task_score(int task, float q, float c) {
    if (task == TASK_LINEAR) {
        return q * c;
    }
    c = fminf(fmaxf(c, 1.0f), 200.0f);
    if (task == TASK_LOG) {
        return q * log10f(c);
    }
    return q / (1.0f + expf(-c / 10.0f));
}

static uint64_t g_rng;
static float urand(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (g_rng >> 40) * (1.0f / 16777216.0f);
}
static float nrand(void) {
    float u1 = urand() + 1e-7f, u2 = urand();
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

static float noisy_cost(int task, float c) {
    if (task == TASK_LOG) {
        return c + 0.2f * nrand() * c;
    }
    if (task == TASK_PERCENTILE) {
        return c - 0.2f * fabsf(nrand()) * c;
    }
    return c;
}

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

typedef struct {
    float regret[N_CHECK], best_q[N_CHECK];
    float band_regret[N_BAND], band_frac[N_BAND], compute;
    double suggest_ms;
    float diag[N_CHECK][N_DIAG];
} SeedResult;

static const char *DIAG_NAMES[N_DIAG] = {"sigma_f", "extra2", "extra3", "ell_cost", "noise",
    "dmu_score", "dmu_cost", "x*_cost", "pick_cost", "dmu_lin", "dsig", "dsig_other", "ell_q", "ell_dis", "rho_q", "floor"};

static const int CHECKS[N_CHECK] = {25, 50, 100, 200};

// band_regret: regret over each quarter of the cost grid, cheap to expensive
static float frontier_regret(int task, const float *costs, const float *true_s, int n,
        const float *grid, float *band_regret) {
    float acc = 0.0f;
    for (int b = 0; b < N_BAND; b++) {
        band_regret[b] = 0.0f;
    }
    for (int g = 0; g < N_GRID; g++) {
        float opt = task_score(task, 1.0f, grid[g]), best = 0.0f;
        for (int i = 0; i < n; i++) {
            if (costs[i] <= grid[g]) {
                best = fmaxf(best, true_s[i]);
            }
        }
        acc += (opt - best) / opt;
        band_regret[g * N_BAND / N_GRID] += (opt - best) / opt / (N_GRID / N_BAND);
    }
    return acc / N_GRID;
}

static const float *g_rank_v;
static int rank_cmp(const void *a, const void *b) {
    float va = g_rank_v[*(const int *)a], vb = g_rank_v[*(const int *)b];
    return (va > vb) - (va < vb);
}

static float spearman(const float *a, const float *b, int n) {
    int idx[N_PROBE];
    float ra[N_PROBE], rb[N_PROBE];
    for (int pass = 0; pass < 2; pass++) {
        const float *v = pass ? b : a;
        float *rk = pass ? rb : ra;
        for (int i = 0; i < n; i++) {
            idx[i] = i;
        }
        g_rank_v = v;
        qsort(idx, n, sizeof(int), rank_cmp);
        for (int i = 0; i < n; i++) {
            rk[idx[i]] = i;
        }
    }
    double m = 0.5 * (n - 1), num = 0.0, da = 0.0, db = 0.0;
    for (int i = 0; i < n; i++) {
        num += (ra[i] - m) * (rb[i] - m);
        da += (ra[i] - m) * (ra[i] - m);
        db += (rb[i] - m) * (rb[i] - m);
    }
    return num / sqrt(da * db + 1e-12);
}

static float g_expansion = 0.1f, g_cost_init = -0.8f, g_noise_floor = 0.0f, g_ucb = 0.0f;
static float g_noise_prior = 0.0f, g_score_noise = 0.0f, g_score_noise_add = 0.0f;
static int g_replicates = 0, g_fisher_hutch = 0;
static int g_fisher = 0, g_log_score = 0;
static int g_diag = 0;

// Score-GP state after a suggest: kernel params, and how both GP means change when the best
// observed run's cost coordinate is pushed DIAG_STEP past it (the extrapolation that decides
// whether expensive candidates look attractive).
static void diag_sample(ProteinSweep *sw, const float *pick, float *d_x2, float *d_o,
        float *out) {
    GaussianProcess *gs = &sw->gp_score, *gc = &sw->gp_cost;
    int np = gs->kernel->n_params;
    for (int k = 0; k < 3; k++) {
        out[k] = DIM + k < np ? softplus(gs->kernel->raw_params[DIM + k]) : 0.0f;
    }
    out[3] = softplus(gs->kernel->raw_params[COST_IDX]);
    out[4] = softplus(gs->raw_noise);
    float x2[2 * DIM], o[2];
    memcpy(x2, &sw->succ_params[sw->top_idx[0] * DIM], DIM * sizeof(float));
    memcpy(x2 + DIM, x2, DIM * sizeof(float));
    x2[DIM + COST_IDX] = fminf(1.0f, x2[COST_IDX] + DIAG_STEP);
    cudaMemcpy(d_x2, x2, sizeof(x2), cudaMemcpyHostToDevice);
    gp_predict(gs, d_x2, d_o, 2, sw->stream);
    cudaMemcpy(o, d_o, sizeof(o), cudaMemcpyDeviceToHost);
    out[5] = o[1] - o[0];
    gp_predict(gc, d_x2, d_o, 2, sw->stream);
    cudaMemcpy(o, d_o, sizeof(o), cudaMemcpyDeviceToHost);
    out[6] = o[1] - o[0];
    out[7] = x2[COST_IDX];
    out[8] = pick[COST_IDX];
    // Linear-kernel share of dmu_score: sigma_f * step * sum_j alpha_j x_j[cost]
    static float alpha[1024], X[1024 * DIM];
    int n = gs->n;
    cudaMemcpy(alpha, gs->d_alpha, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(X, gs->d_X, n * DIM * sizeof(float), cudaMemcpyDeviceToHost);
    float acc = 0.0f;
    for (int j = 0; j < n; j++) {
        acc += alpha[j] * X[j * DIM + COST_IDX];
    }
    float lin_w = np > DIM + 2 ? out[2] : out[0];
    out[9] = lin_w * (x2[DIM + COST_IDX] - x2[COST_IDX]) * acc;
    out[10] = out[11] = 0.0f;
#ifdef GP_LAPLACE_RIDGE
    float v[2], xo = x2[DIAG_OTHER_DIM];
    gp_predict(gs, d_x2, d_o, 2, sw->stream);
    gp_predict_var(gs, d_x2, d_o, 2, sw->stream);
    cudaMemcpy(v, d_o, sizeof(v), cudaMemcpyDeviceToHost);
    out[10] = sqrtf(v[1]) - sqrtf(v[0]);
    // same step along a distractor dim (away from the boundary) instead of cost
    x2[DIM + COST_IDX] = x2[COST_IDX];
    x2[DIM + DIAG_OTHER_DIM] = xo + (xo > 0.0f ? -DIAG_STEP : DIAG_STEP);
    cudaMemcpy(d_x2, x2, sizeof(x2), cudaMemcpyHostToDevice);
    gp_predict(gs, d_x2, d_o, 2, sw->stream);
    gp_predict_var(gs, d_x2, d_o, 2, sw->stream);
    cudaMemcpy(v, d_o, sizeof(v), cudaMemcpyDeviceToHost);
    out[11] = sqrtf(v[1]) - sqrtf(v[0]);
#endif
    // Lengthscale geo-means over quality dims (1, 5-9) and distractors (2-4, 10-15)
    static const int QDIMS[6] = {1, 5, 6, 7, 8, 9}, DDIMS[9] = {2, 3, 4, 10, 11, 12, 13, 14, 15};
    double lq = 0.0, ld = 0.0;
    for (int k = 0; k < 6; k++) {
        lq += log(softplus(gs->kernel->raw_params[QDIMS[k]])) / 6;
    }
    for (int k = 0; k < 9; k++) {
        ld += log(softplus(gs->kernel->raw_params[DDIMS[k]])) / 9;
    }
    out[12] = exp(lq);
    out[13] = exp(ld);
    // rho_q: Spearman(mu, true quality) over probes around the best run at its cost, each
    // non-cost dim moved uniformly by +-0.5 (the candidate sampler's scale). Own RNG.
    static float *d_probe, *d_pmu, probe[N_PROBE * DIM], pmu[N_PROBE], pq[N_PROBE];
    if (!d_probe) {
        cudaMalloc(&d_probe, sizeof(probe));
        cudaMalloc(&d_pmu, sizeof(pmu));
    }
    uint64_t r = 0x2545F4914F6CDD1DULL ^ (uint64_t)gs->n;
    for (int i = 0; i < N_PROBE; i++) {
        for (int k = 0; k < DIM; k++) {
            r ^= r << 13;
            r ^= r >> 7;
            r ^= r << 17;
            float u = (r >> 40) * (1.0f / 16777216.0f);
            float v0 = x2[k] + (k == COST_IDX ? 0.0f : 0.5f * (2.0f * u - 1.0f));
            probe[i * DIM + k] = fmaxf(-1.0f, fminf(1.0f, v0));
        }
        pq[i] = base_quality(&probe[i * DIM]);
    }
    cudaMemcpy(d_probe, probe, sizeof(probe), cudaMemcpyHostToDevice);
    gp_predict(gs, d_probe, d_pmu, N_PROBE, sw->stream);
    cudaMemcpy(pmu, d_pmu, sizeof(pmu), cudaMemcpyDeviceToHost);
    out[14] = spearman(pmu, pq, N_PROBE);
    out[15] = 0.0f;
#ifdef GP_FISHER_DAMP
    out[15] = gs->noise_floor;
#endif
}

static SeedResult run_seed(int task, int seed, int runs, int parallel, int curve) {
    SweepSpace space;
    make_space(&space);
    ProteinSweep *sw = protein_sweep_create((ProteinSweep){
        .space = &space,
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
        .max_suggestion_cost = task == TASK_LOG ? 1000.0f : 500.0f, // top of the cost grid
        .expansion_rate = g_expansion,
        .cost_random_suggestion = g_cost_init,
        .early_stop_quantile = 0.3f,
        .success_cap = 8192,
        .failure_cap = 1024,
        .top_k = 5,
        .rng_seed = 73ULL + seed,
#ifdef GP_FISHER_DAMP
        .gp_fisher_steps = g_fisher,
        .gp_fisher_hutch = g_fisher_hutch,
        .gp_noise_floor = g_noise_floor,
#endif
#ifdef GP_LAPLACE_RIDGE
        .ucb_beta = g_ucb,
        .gp_noise_prior_weight = g_noise_prior,
#endif
#ifdef PROTEIN_MAX_REPLICATES
        .noise_replicates = g_replicates,
#endif
    });
    srand(1000 + seed);
    g_rng = 0x9E3779B97F4A7C15ULL ^ (uint64_t)(seed + 1) * 0xBF58476D1CE4E5B9ULL;

    // grid spans the cheapest to ~1000x the cheapest run (c = timesteps / 3e7)
    float grid[N_GRID];
    for (int g = 0; g < N_GRID; g++) {
        grid[g] = (task == TASK_LOG ? 2.0f : 1.0f) * powf(500.0f, (g + 0.5f) / N_GRID);
    }
    static float costs[MAX_RUNS], true_s[MAX_RUNS], qual[MAX_RUNS];
    float batch[MAX_PAR][DIM];
    int is_rep[MAX_PAR];
    SeedResult r = {0};
    double t_suggest = 0.0;
    int done = 0, ci = 0;
    float *d_x2, *d_o;
    cudaMalloc(&d_x2, 2 * DIM * sizeof(float));
    cudaMalloc(&d_o, 2 * sizeof(float));
    int diag_cnt[N_CHECK] = {0};
    while (done < runs) {
        int nb = parallel < runs - done ? parallel : runs - done;
        for (int b = 0; b < nb; b++) {
            double t0 = now_s();
            ProteinSweepInfo info = protein_sweep_suggest(sw, batch[b], NAN);
            t_suggest += now_s() - t0;
            is_rep[b] = 0;
#ifdef PROTEIN_MAX_REPLICATES
            is_rep[b] = info.is_replicate;
#endif
            int bin = 0;
            while (bin < N_CHECK - 1 && done + b >= CHECKS[bin]) {
                bin++;
            }
            if (g_diag && !info.is_random && !is_rep[b]) {
                float dv[N_DIAG];
                diag_sample(sw, batch[b], d_x2, d_o, dv);
                for (int k = 0; k < N_DIAG; k++) {
                    r.diag[bin][k] += dv[k];
                }
                diag_cnt[bin]++;
            }
        }
        for (int b = 0; b < nb; b++) {
            float *u = batch[b];
            float c = space_unnormalize(&space.spaces[COST_IDX], u[COST_IDX]) / COST_UNIT;
            float q = base_quality(u);
            float nc = noisy_cost(task, c);
            float eps = g_score_noise > 0.0f ? g_score_noise * nrand() : 0.0f, s = 0.0f;
            float add = g_score_noise_add > 0.0f ? g_score_noise_add * task_score(task, 1.0f, grid[N_GRID - 1]) * nrand() : 0.0f;
            for (int p = 1; p <= DOWNSAMPLE; p++) {
                float cp = c * p / DOWNSAMPLE;
                s = task_score(task, q, nc * p / DOWNSAMPLE) * (1.0f + eps) + add;
                float up[DIM];
                memcpy(up, u, sizeof(up));
                if (curve) {
                    up[COST_IDX] = space_normalize(&space.spaces[COST_IDX], cp * COST_UNIT);
                }
                protein_sweep_observe(sw, up, g_log_score ? logf(s) : s, cp, 0);
            }
#ifdef PROTEIN_MAX_REPLICATES
            if (done == 0 || is_rep[b]) {
                protein_sweep_observe_replicate(sw, g_log_score ? logf(s) : s);
            }
#endif
            costs[done] = c;
            true_s[done] = task_score(task, q, c);
            qual[done] = q;
            done++;
            while (ci < N_CHECK && done == CHECKS[ci]) {
                float bq = 0.0f;
                for (int i = 0; i < done; i++) {
                    bq = fmaxf(bq, qual[i]);
                }
                r.regret[ci] = frontier_regret(task, costs, true_s, done, grid,
                    r.band_regret);
                r.best_q[ci] = bq;
                ci++;
            }
        }
    }
    r.suggest_ms = 1e3 * t_suggest / runs;
    for (int c = 0; c < N_CHECK; c++) {
        for (int k = 0; k < N_DIAG; k++) {
            r.diag[c][k] /= diag_cnt[c] > 0 ? diag_cnt[c] : 1;
        }
    }
    // fraction of runs per band, bands split at the grid quarter boundaries
    for (int i = 0; i < runs; i++) {
        int b = 0;
        while (b < N_BAND - 1 && costs[i] > grid[(b + 1) * N_GRID / N_BAND]) {
            b++;
        }
        r.band_frac[b] += 1.0f / runs;
        r.compute += costs[i];
    }
    return r;
}

int main(int argc, char **argv) {
    int task_lo = 0, task_hi = N_TASKS, seeds = 8, runs = 200, parallel = 1, curve = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--task") == 0 && i + 1 < argc) {
            i++;
            for (int t = 0; t < N_TASKS; t++) {
                if (strcmp(argv[i], TASK_NAMES[t]) == 0) {
                    task_lo = t;
                    task_hi = t + 1;
                }
            }
        } else if (strcmp(argv[i], "--seeds") == 0 && i + 1 < argc) {
            seeds = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            runs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--parallel") == 0 && i + 1 < argc) {
            parallel = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--expansion") == 0 && i + 1 < argc) {
            g_expansion = atof(argv[++i]);
        } else if (strcmp(argv[i], "--cost-init") == 0 && i + 1 < argc) {
            g_cost_init = atof(argv[++i]);
        } else if (strcmp(argv[i], "--cost-scale") == 0 && i + 1 < argc) {
            g_cost_scale = atof(argv[++i]);
        } else if (strcmp(argv[i], "--fisher") == 0 && i + 1 < argc) {
            g_fisher = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--fisher-hutch") == 0 && i + 1 < argc) {
            g_fisher_hutch = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--noise-floor") == 0 && i + 1 < argc) {
            g_noise_floor = atof(argv[++i]);
        } else if (strcmp(argv[i], "--replicates") == 0 && i + 1 < argc) {
            g_replicates = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--score-noise") == 0 && i + 1 < argc) {
            g_score_noise = atof(argv[++i]);
        } else if (strcmp(argv[i], "--score-noise-add") == 0 && i + 1 < argc) {
            g_score_noise_add = atof(argv[++i]);
        } else if (strcmp(argv[i], "--noise-prior") == 0 && i + 1 < argc) {
            g_noise_prior = atof(argv[++i]);
        } else if (strcmp(argv[i], "--ucb") == 0 && i + 1 < argc) {
            g_ucb = atof(argv[++i]);
        } else if (strcmp(argv[i], "--log-score") == 0) {
            g_log_score = 1;
        } else if (strcmp(argv[i], "--diag") == 0) {
            g_diag = 1;
        } else if (strcmp(argv[i], "--curve") == 0) {
            curve = 1;
        }
    }
    parallel = parallel < MAX_PAR ? parallel : MAX_PAR;
    runs = runs < MAX_RUNS ? runs : MAX_RUNS;
    seeds = seeds < MAX_SEEDS ? seeds : MAX_SEEDS;
    printf("seeds=%d runs=%d parallel=%d curve=%d expansion=%g cost_init=%g cost_scale=%g"
        " fisher=%d noise_floor=%g ucb=%g log_score=%d score_noise=%g score_noise_add=%g"
        " replicates=%d fisher_hutch=%d\n", seeds,
        runs, parallel, curve, g_expansion, g_cost_init, g_cost_scale, g_fisher, g_noise_floor,
        g_ucb, g_log_score, g_score_noise, g_score_noise_add, g_replicates, g_fisher_hutch);
    printf("%-11s %6s %18s %18s %10s\n", "task", "runs", "regret (mean±se)",
        "best_q (mean±se)", "ms/suggest");
    for (int task = task_lo; task < task_hi; task++) {
        SeedResult res[MAX_SEEDS];
        double ms = 0.0;
        // One process per seed: each sweep allocates at its process start and the OS
        // reclaims it at exit (~400MB of d_Ks per sweep would not fit 24x on one GPU)
        for (int s = 0; s < seeds; s++) {
            int fd[2];
            assert(pipe(fd) == 0);
            if (fork() == 0) {
                SeedResult r = run_seed(task, s, runs, parallel, curve);
                assert(write(fd[1], &r, sizeof(r)) == sizeof(r));
                _exit(0);
            }
            close(fd[1]);
            assert(read(fd[0], &res[s], sizeof(SeedResult)) == sizeof(SeedResult));
            close(fd[0]);
            wait(NULL);
            ms += res[s].suggest_ms / seeds;
        }
        for (int c = 0; c < N_CHECK && CHECKS[c] <= runs; c++) {
            double m = 0, m2 = 0, q = 0, q2 = 0;
            for (int s = 0; s < seeds; s++) {
                m += res[s].regret[c];
                m2 += res[s].regret[c] * res[s].regret[c];
                q += res[s].best_q[c];
                q2 += res[s].best_q[c] * res[s].best_q[c];
            }
            m /= seeds;
            q /= seeds;
            double se = seeds > 1 ? sqrt(fmax(m2 / seeds - m * m, 0) / (seeds - 1)) : 0;
            double qse = seeds > 1 ? sqrt(fmax(q2 / seeds - q * q, 0) / (seeds - 1)) : 0;
            printf("%-11s %6d %10.4f±%.4f %10.4f±%.4f %10.2f\n", TASK_NAMES[task],
                CHECKS[c], m, se, q, qse, ms);
        }
        printf("%-11s  band regret (cheap->expensive):", TASK_NAMES[task]);
        for (int b = 0; b < N_BAND; b++) {
            double m = 0, m2 = 0;
            for (int s = 0; s < seeds; s++) {
                m += res[s].band_regret[b] / seeds;
                m2 += res[s].band_regret[b] * res[s].band_regret[b] / seeds;
            }
            printf(" %.3f±%.3f", m, seeds > 1 ? sqrt(fmax(m2 - m * m, 0) / (seeds - 1)) : 0);
        }
        double comp = 0;
        for (int s = 0; s < seeds; s++) {
            comp += res[s].compute / seeds;
        }
        printf("\n%-11s  total compute (cheapest-run units): %.0f", TASK_NAMES[task], comp);
        printf("\n%-11s  run share per band:            ", TASK_NAMES[task]);
        for (int b = 0; b < N_BAND; b++) {
            double f = 0;
            for (int s = 0; s < seeds; s++) {
                f += res[s].band_frac[b] / seeds;
            }
            printf(" %.3f      ", f);
        }
        printf("\n");
        if (!g_diag) {
            continue;
        }
        printf("%-11s  diag (mean over seeds, suggestions up to run):\n%-11s %6s",
            TASK_NAMES[task], "", "runs");
        for (int k = 0; k < N_DIAG; k++) {
            printf(" %10s", DIAG_NAMES[k]);
        }
        printf("\n");
        for (int c = 0; c < N_CHECK && CHECKS[c] <= runs; c++) {
            printf("%-11s %6d", "", CHECKS[c]);
            for (int k = 0; k < N_DIAG; k++) {
                double m = 0;
                for (int s = 0; s < seeds; s++) {
                    m += res[s].diag[c][k] / seeds;
                }
                printf(" %10.4f", m);
            }
            printf("\n");
        }
    }
    return 0;
}
