// replay_protein_gp.cu -- Offline calibration check of protein.cu's GPs on logged sweeps
//
// Refits the score and cost GPs on PufferLib sweep logs (logs/<env>/sweep_*.ini) and scores
// held-out predictions, so GP-side changes can be compared without running sweeps.
//   seq:   replay each logged sweep in completion order. Before run t (t > 10) fit on runs
//          [0, t) the way protein_sweep_suggest does (warm-started Adam, 50 iters per
//          suggestion, optimizer reset every 50 suggestions) and predict run t.
//   split: pool every run in the directory, fit on random subsets of --ntrain runs with the
//          same per-suggestion training schedule, predict the remaining runs.
// Metrics are in the normalized units protein trains on (score min-max, log-cost
// min/q97). NLL is the mean Gaussian negative log likelihood of y under (mu, sigma^2 +
// noise). z^2 = mean squared z-score (1 if calibrated); c1/c2 = fraction with |z| < 1 / 2
// (0.683 / 0.954 if calibrated). rho = Spearman(mu, y). rho_se = Spearman(sigma, |y - mu|),
// i.e. whether sigma ranks the errors.
//
// Build:
//   nvcc -O2 -o replay tests/replay_protein_gp.cu -I src/ -lcublas -lcusolver -lcurand
// Point -I at a directory holding another protein.cu to A/B versions.
// Usage:
//   ./replay logs/breakout [--mode seq|split] [--ntrain N] [--splits K] [--seed S]
//                          [--lr LR] [--iters N]   (run_sweep: 0.001, 50)
//   --fisher N / --noise-floor F (protein.cu defining GP_FISHER_DAMP): Fisher-scoring steps
//   per suggestion instead of Adam; F added to the score GP's noise variance.
//   --fisher-hutch M: 0 exact Fisher, 1 Hutchinson Fisher matrix + exact gradient,
//   2 Hutchinson for both (current candidate only)
//   --laplace (GP_LAPLACE_RIDGE): Laplace hyperparameter marginalization of the variance;
//   --noise-prior W: score-GP noise prior log-sd NOISE_PRIOR_SIGMA / sqrt(1 + W)

#include <cuda_runtime.h>
#include <dirent.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

#define MAX_RUNS 4096
#define MAX_DIM 32
#define MAX_KV 1024
#define GP_CAP 750
#define N_RANDOM 10
#define RESET_FREQ 50
#define STATS_CAP 65536

typedef struct {
    char *sec, *key, *val;
} KV;

typedef struct {
    KV kv[MAX_KV];
    int n;
} Ini;

static void ini_load(Ini *ini, const char *path) {
    FILE *f = fopen(path, "rb");
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(size + 1);
    size = fread(buf, 1, size, f);
    fclose(f);
    buf[size] = 0;
    ini->n = 0;
    char *sec = buf + size;
    for (char *line = strtok(buf, "\n"); line; line = strtok(NULL, "\n")) {
        if (line[0] == '[') {
            *strchr(line, ']') = 0;
            sec = line + 1;
            continue;
        }
        char *eq = strstr(line, " = ");
        if (line[0] == '#' || !eq || ini->n >= MAX_KV) {
            continue;
        }
        *eq = 0;
        ini->kv[ini->n++] = (KV){.sec = sec, .key = line, .val = eq + 3};
    }
}

static const char *ini_get(const Ini *ini, const char *sec, const char *key) {
    for (int i = 0; i < ini->n; i++) {
        if (!strcmp(ini->kv[i].sec, sec) && !strcmp(ini->kv[i].key, key)) {
            return ini->kv[i].val;
        }
    }
    return NULL;
}

static float last_value(const char *csv) {
    const char *c = strrchr(csv, ',');
    return atof(c ? c + 1 : csv);
}

typedef struct {
    char sec[64], key[64];
    Space space;
} Param;

static int load_space(const Ini *ini, Param *params, int *cost_idx) {
    int n = 0;
    *cost_idx = -1;
    for (int i = 0; i < ini->n; i++) {
        const KV *kv = &ini->kv[i];
        if (strncmp(kv->sec, "sweep.", 6) || strcmp(kv->key, "distribution")) {
            continue;
        }
        Param *p = &params[n];
        const char *name = kv->sec + 6;
        const char *dot = strchr(name, '.');
        snprintf(p->sec, sizeof(p->sec), "%.*s", (int)(dot - name), name);
        snprintf(p->key, sizeof(p->key), "%s", dot + 1);
        const char *dist = kv->val;
        SpaceType type = SPACE_LINEAR;
        if (!strcmp(dist, "uniform_pow2")) {
            type = SPACE_POW2;
        } else if (!strcmp(dist, "log_normal")) {
            type = SPACE_LOG;
        } else if (!strcmp(dist, "logit_normal")) {
            type = SPACE_LOGIT;
        }
        p->space = (Space){.type = type,
            .min = (float)atof(ini_get(ini, kv->sec, "min")),
            .max = (float)atof(ini_get(ini, kv->sec, "max")), .scale = 0.5f,
            .is_integer = !strcmp(dist, "int_uniform") || type == SPACE_POW2};
        if (!strcmp(kv->sec, "sweep.train.total_timesteps")) {
            *cost_idx = n;
        }
        n++;
    }
    return n;
}

typedef struct {
    float x[MAX_DIM], score, cost;
    int sweep;
} Run;

static int name_cmp(const void *a, const void *b) {
    return strcmp(*(char *const *)a, *(char *const *)b);
}

// Loads every run whose sweep space matches the newest file's. Returns run count.
static int load_runs(const char *dir, Run *runs, Param *params, int *dim, int *cost_idx) {
    DIR *dp = opendir(dir);
    char *names[MAX_RUNS];
    int n_files = 0;
    for (struct dirent *e = readdir(dp); e && n_files < MAX_RUNS; e = readdir(dp)) {
        if (!strncmp(e->d_name, "sweep_", 6) && strstr(e->d_name, ".ini")) {
            names[n_files++] = strdup(e->d_name);
        }
    }
    closedir(dp);
    qsort(names, n_files, sizeof(char *), name_cmp);
    static Ini ini;
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dir, names[n_files - 1]);
    ini_load(&ini, path);
    *dim = load_space(&ini, params, cost_idx);
    Param p2[MAX_DIM];
    int n = 0, sweep = -1, prev_idx = 1 << 30, skipped = 0;
    char metric_key[128];
    for (int f = 0; f < n_files; f++) {
        snprintf(path, sizeof(path), "%s/%s", dir, names[f]);
        ini_load(&ini, path);
        long long ts;
        int idx;
        sscanf(names[f], "sweep_%lld_%d.ini", &ts, &idx);
        if (idx <= prev_idx) {
            sweep++;
        }
        prev_idx = idx;
        int ci2;
        int d2 = load_space(&ini, p2, &ci2);
        int match = d2 == *dim;
        for (int i = 0; match && i < d2; i++) {
            match = p2[i].space.type == params[i].space.type
                && p2[i].space.min == params[i].space.min
                && p2[i].space.max == params[i].space.max;
        }
        const char *metric = ini_get(&ini, "sweep", "metric");
        snprintf(metric_key, sizeof(metric_key), strchr(metric, '/') ? "%s" : "env/%s", metric);
        const char *score = ini_get(&ini, "metrics", metric_key);
        const char *cost = ini_get(&ini, "metrics", "uptime");
        if (!match || !score || !cost) {
            skipped++;
            continue;
        }
        Run *r = &runs[n++];
        r->sweep = sweep;
        r->score = last_value(score);
        r->cost = last_value(cost);
        for (int i = 0; i < *dim; i++) {
            r->x[i] = space_normalize(&params[i].space,
                (float)atof(ini_get(&ini, params[i].sec, params[i].key)));
        }
    }
    printf("%s: %d runs, %d sweeps, %d skipped, dim %d\n", dir, n, sweep + 1, skipped, *dim);
    return n;
}

typedef struct {
    GaussianProcess gp;
    float *m, *v, *vmax, *init_raw;
    float init_noise, init_mean;
    int t, n_opt;
} Fit;

static void fit_init(Fit *f, int dim, int fisher_steps, int fisher_hutch) {
#if defined(GP_FISHER_DAMP) && !defined(GP_LAPLACE_RIDGE)
    gp_init(&f->gp, dim, GP_CAP, MAX_RUNS, 0.01f, fisher_steps, fisher_hutch);
#else
    gp_init(&f->gp, dim, GP_CAP, MAX_RUNS, 0.01f);
#endif
    int np = f->gp.kernel->n_params;
    f->n_opt = np + 2;
    f->m = (float *)calloc(3 * f->n_opt + np, sizeof(float));
    f->v = f->m + f->n_opt;
    f->vmax = f->v + f->n_opt;
    f->init_raw = f->vmax + f->n_opt;
    memcpy(f->init_raw, f->gp.kernel->raw_params, np * sizeof(float));
    f->init_noise = f->gp.raw_noise;
    f->init_mean = f->gp.mean;
}

static void fit_reset(Fit *f) {
    memcpy(f->gp.kernel->raw_params, f->init_raw, f->gp.kernel->n_params * sizeof(float));
    f->gp.raw_noise = f->init_noise;
    f->gp.mean = f->init_mean;
    memset(f->m, 0, 3 * f->n_opt * sizeof(float));
    f->t = 0;
}

static int train_iters = 50;
static float train_lr = 0.001f;

// One protein_sweep_suggest worth of training; call_idx is the suggestion index.
static void fit_step(Fit *f, const float *X, const float *y, int n, int call_idx,
        cudaStream_t s) {
    gp_train(&f->gp, f->m, f->v, f->vmax, &f->t, train_lr, X, y, n, train_iters, s);
    if (call_idx % RESET_FREQ == 0) {
        memset(f->m, 0, 3 * f->n_opt * sizeof(float));
        f->t = 0;
    }
}

static float *d_Xte, *d_mu, *d_Kss, *d_V, *h_V, *h_Kss;

// Posterior mean and predictive variance (latent + noise) at m test points.
static void fit_predict(Fit *f, const float *X, int m, float *mu, float *var,
        cudaStream_t s) {
    GaussianProcess *gp = &f->gp;
    int n = gp->n, d = gp->dim;
    cudaMemcpyAsync(d_Xte, X, m * d * sizeof(float), cudaMemcpyHostToDevice, s);
    gp_predict(gp, d_Xte, d_mu, m, s);
    matern32lin_build_K(gp, d_Xte, d_Xte, m, m, 0.0f, d_Kss, s);
    cudaMemcpyAsync(d_V, gp->d_Ks, n * m * sizeof(float), cudaMemcpyDeviceToDevice, s);
    float one = 1.0f;
    cublasSetStream(gp->cublas, s);
    cublasStrsm(gp->cublas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT, n, m, &one, gp->d_L, n, d_V, n);
    cudaMemcpyAsync(mu, d_mu, m * sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(h_V, d_V, n * m * sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(h_Kss, d_Kss, m * m * sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    float noise = softplus(gp->raw_noise);
#ifdef GP_FISHER_DAMP
    noise += gp->noise_floor;
#endif
    for (int j = 0; j < m; j++) {
        double q = 0.0;
        for (int i = 0; i < n; i++) {
            q += (double)h_V[j * n + i] * h_V[j * n + i];
        }
        var[j] = fmaxf(h_Kss[j * m + j] - (float)q, 1e-8f) + noise;
    }
#ifdef GP_LAPLACE_RIDGE
    if (gp->laplace) {
        gp_predict(gp, d_Xte, d_mu, m, s);
        gp_predict_var(gp, d_Xte, d_Kss, m, s);
        cudaMemcpy(var, d_Kss, m * sizeof(float), cudaMemcpyDeviceToHost);
        for (int j = 0; j < m; j++) {
            var[j] = fmaxf(var[j], 1e-8f) + noise;
        }
    }
#endif
}

typedef struct {
    double nll, z2, se, sig;
    int c1, c2, n;
    float *mu, *y, *s;
} Stats;

static void stats_add(Stats *st, float mu, float var, float y) {
    float z = (y - mu) / sqrtf(var);
    st->nll += 0.5 * log(2.0 * M_PI * var) + 0.5 * z * z;
    st->z2 += z * z;
    st->se += (y - mu) * (y - mu);
    st->sig += sqrtf(var);
    st->c1 += fabsf(z) < 1.0f;
    st->c2 += fabsf(z) < 2.0f;
    if (st->n < STATS_CAP) {
        st->mu[st->n] = mu;
        st->y[st->n] = y;
        st->s[st->n] = sqrtf(var);
    }
    st->n++;
}

static const float *g_rank_v;
static int rank_cmp(const void *a, const void *b) {
    float va = g_rank_v[*(const int *)a], vb = g_rank_v[*(const int *)b];
    return (va > vb) - (va < vb);
}

static void ranks(const float *v, int n, float *out, int *idx) {
    for (int i = 0; i < n; i++) {
        idx[i] = i;
    }
    g_rank_v = v;
    qsort(idx, n, sizeof(int), rank_cmp);
    for (int i = 0; i < n; i++) {
        out[idx[i]] = i;
    }
}

static float spearman(const float *a, const float *b, int n) {
    static float ra[STATS_CAP], rb[STATS_CAP];
    static int idx[STATS_CAP];
    ranks(a, n, ra, idx);
    ranks(b, n, rb, idx);
    double m = 0.5 * (n - 1), num = 0.0, da = 0.0, db = 0.0;
    for (int i = 0; i < n; i++) {
        num += (ra[i] - m) * (rb[i] - m);
        da += (ra[i] - m) * (ra[i] - m);
        db += (rb[i] - m) * (rb[i] - m);
    }
    return num / sqrt(da * db + 1e-12);
}

static void stats_print(const char *label, Stats *st) {
    int n = st->n < STATS_CAP ? st->n : STATS_CAP;
    static float err[STATS_CAP];
    for (int i = 0; i < n; i++) {
        err[i] = fabsf(st->y[i] - st->mu[i]);
    }
    printf("  %-14s n=%5d  nll %7.3f  z2 %7.2f  c1 %.3f  c2 %.3f  rmse %.3f  sig %.3f"
        "  rho %+.3f  rho_se %+.3f\n", label, st->n, st->nll / st->n, st->z2 / st->n,
        (float)st->c1 / st->n, (float)st->c2 / st->n, sqrt(st->se / st->n),
        st->sig / st->n, spearman(st->mu, st->y, n), spearman(st->s, err, n));
}

static void stats_init(Stats *st) {
    memset(st, 0, sizeof(*st));
    st->mu = (float *)malloc(3 * STATS_CAP * sizeof(float));
    st->y = st->mu + STATS_CAP;
    st->s = st->y + STATS_CAP;
}

static void print_hypers(const char *label, Fit *f) {
    GaussianProcess *gp = &f->gp;
    int d = gp->dim, np = gp->kernel->n_params;
    double lg = 0.0;
    float lmin = FLT_MAX;
    for (int i = 0; i < d; i++) {
        float l = softplus(gp->kernel->raw_params[i]);
        lg += log(l);
        lmin = fminf(lmin, l);
    }
    printf("    %s: ell geo %.3f min %.3f  noise %.4f  mean %+.3f  extra", label,
        exp(lg / d), lmin, softplus(gp->raw_noise), gp->mean);
    for (int i = d; i < np; i++) {
        printf(" %.4f", softplus(gp->kernel->raw_params[i]));
    }
    printf("\n");
}

static float *Xb, *yb, *cb, *Xt, *yt, *ct, *mu_b, *var_b;

// Protein's normalization of training targets; test targets mapped with the same constants.
static void normalize(const Run *runs, const int *tr, int n_tr, const int *te, int n_te,
        int dim) {
    float smin = FLT_MAX, smax = -FLT_MAX;
    static float lc[MAX_RUNS];
    for (int i = 0; i < n_tr; i++) {
        smin = fminf(smin, runs[tr[i]].score);
        smax = fmaxf(smax, runs[tr[i]].score);
        lc[i] = logf(fmaxf(runs[tr[i]].cost, PROTEIN_EPSILON));
    }
    qsort(lc, n_tr, sizeof(float), float_cmp);
    float qidx = PROTEIN_COST_QUANTILE * (n_tr - 1);
    int qlo = (int)qidx, qhi = qlo + 1 < n_tr ? qlo + 1 : n_tr - 1;
    float lcmin = lc[0], lcmax = lc[qlo] * (1.0f - (qidx - qlo)) + lc[qhi] * (qidx - qlo);
    float srange = fabsf(smax - smin) + PROTEIN_EPSILON;
    float crange = lcmax - lcmin + PROTEIN_EPSILON;
    for (int i = 0; i < n_tr + n_te; i++) {
        const Run *r = &runs[i < n_tr ? tr[i] : te[i - n_tr]];
        float *X = i < n_tr ? &Xb[i * dim] : &Xt[(i - n_tr) * dim];
        memcpy(X, r->x, dim * sizeof(float));
        float y = (r->score - smin) / srange;
        float c = (logf(fmaxf(r->cost, PROTEIN_EPSILON)) - lcmin) / crange;
        if (i < n_tr) {
            yb[i] = y;
            cb[i] = c;
        } else {
            yt[i - n_tr] = y;
            ct[i - n_tr] = c;
        }
    }
}

int main(int argc, char **argv) {
    const char *dir = argv[1];
    const char *mode = "seq";
    int ntrain = 100, splits = 5, fisher_steps = 0, fisher_hutch = 0, laplace = 0;
    float noise_floor = 0.0f, noise_prior_w = 0.0f;
    unsigned seed = 1;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--mode")) {
            mode = argv[++i];
        } else if (!strcmp(argv[i], "--ntrain")) {
            ntrain = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--splits")) {
            splits = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--lr")) {
            train_lr = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--iters")) {
            train_iters = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--fisher")) {
            fisher_steps = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--fisher-hutch")) {
            fisher_hutch = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--laplace")) {
            laplace = 1;
        } else if (!strcmp(argv[i], "--noise-floor")) {
            noise_floor = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--noise-prior")) {
            noise_prior_w = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--seed")) {
            seed = atoi(argv[++i]);
        }
    }
    static Run runs[MAX_RUNS];
    static Param params[MAX_DIM];
    int dim, cost_idx;
    int n = load_runs(dir, runs, params, &dim, &cost_idx);
    Xb = (float *)malloc(2 * MAX_RUNS * (dim + 4) * sizeof(float));
    yb = Xb + MAX_RUNS * dim;
    cb = yb + MAX_RUNS;
    Xt = cb + MAX_RUNS;
    yt = Xt + MAX_RUNS * dim;
    ct = yt + MAX_RUNS;
    mu_b = ct + MAX_RUNS;
    var_b = mu_b + MAX_RUNS;
    cudaMalloc(&d_Xte, MAX_RUNS * dim * sizeof(float));
    cudaMalloc(&d_mu, MAX_RUNS * sizeof(float));
    cudaMalloc(&d_Kss, MAX_RUNS * MAX_RUNS * sizeof(float));
    cudaMalloc(&d_V, GP_CAP * MAX_RUNS * sizeof(float));
    h_V = (float *)malloc(GP_CAP * MAX_RUNS * sizeof(float));
    h_Kss = (float *)malloc(MAX_RUNS * MAX_RUNS * sizeof(float));
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    Fit fs, fc;
    fit_init(&fs, dim, fisher_steps, fisher_hutch);
    fit_init(&fc, dim, fisher_steps, fisher_hutch);
#ifdef GP_FISHER_DAMP
    fs.gp.fisher_steps = fc.gp.fisher_steps = fisher_steps;
    fs.gp.noise_floor = noise_floor;
#endif
#ifdef GP_LAPLACE_RIDGE
    fs.gp.laplace = fc.gp.laplace = laplace;
    fs.gp.noise_prior_sigma = NOISE_PRIOR_SIGMA / sqrtf(1.0f + noise_prior_w);
#endif
    static int tr[MAX_RUNS], te[MAX_RUNS];
    Stats ss[4], sc[4];
    for (int b = 0; b < 4; b++) {
        stats_init(&ss[b]);
        stats_init(&sc[b]);
    }
    const char *bands[4] = {"all", "t<30", "30<=t<80", "t>=80"};

    if (!strcmp(mode, "seq")) {
        for (int lo = 0; lo < n;) {
            int hi = lo;
            while (hi < n && runs[hi].sweep == runs[lo].sweep) {
                hi++;
            }
            fit_reset(&fs);
            fit_reset(&fc);
            for (int t = N_RANDOM + 1; t < hi - lo; t++) {
                int n_tr = t < GP_CAP ? t : GP_CAP;
                for (int i = 0; i < n_tr; i++) {
                    tr[i] = lo + t - n_tr + i;
                }
                te[0] = lo + t;
                normalize(runs, tr, n_tr, te, 1, dim);
                fit_step(&fs, Xb, yb, n_tr, t, stream);
                fit_step(&fc, Xb, cb, n_tr, t, stream);
                int band = t < 30 ? 1 : t < 80 ? 2 : 3;
                fit_predict(&fs, Xt, 1, mu_b, var_b, stream);
                stats_add(&ss[0], mu_b[0], var_b[0], yt[0]);
                stats_add(&ss[band], mu_b[0], var_b[0], yt[0]);
                fit_predict(&fc, Xt, 1, mu_b, var_b, stream);
                stats_add(&sc[0], mu_b[0], var_b[0], ct[0]);
                stats_add(&sc[band], mu_b[0], var_b[0], ct[0]);
                if (t == 30 || t == 79 || t == hi - lo - 1) {
                    char label[64];
                    snprintf(label, sizeof(label), "sweep %d t=%d score", runs[lo].sweep, t);
                    print_hypers(label, &fs);
                }
            }
            lo = hi;
        }
    } else {
        srand(seed);
        static int perm[MAX_RUNS];
        for (int sp = 0; sp < splits; sp++) {
            for (int i = 0; i < n; i++) {
                perm[i] = i;
            }
            for (int i = n - 1; i > 0; i--) {
                int j = rand() % (i + 1);
                int tmp = perm[i];
                perm[i] = perm[j];
                perm[j] = tmp;
            }
            int n_tr = ntrain, n_te = n - ntrain;
            memcpy(tr, perm, n_tr * sizeof(int));
            memcpy(te, perm + n_tr, n_te * sizeof(int));
            normalize(runs, tr, n_tr, te, n_te, dim);
            fit_reset(&fs);
            fit_reset(&fc);
            for (int call = N_RANDOM + 1; call <= n_tr; call++) {
                fit_step(&fs, Xb, yb, n_tr, call, stream);
                fit_step(&fc, Xb, cb, n_tr, call, stream);
            }
            fit_predict(&fs, Xt, n_te, mu_b, var_b, stream);
            for (int i = 0; i < n_te; i++) {
                stats_add(&ss[0], mu_b[i], var_b[i], yt[i]);
            }
            fit_predict(&fc, Xt, n_te, mu_b, var_b, stream);
            for (int i = 0; i < n_te; i++) {
                stats_add(&sc[0], mu_b[i], var_b[i], ct[i]);
            }
            char label[64];
            snprintf(label, sizeof(label), "split %d score", sp);
            print_hypers(label, &fs);
        }
    }
    int nb = strcmp(mode, "seq") ? 1 : 4;
    printf("score GP (%s)\n", mode);
    for (int b = 0; b < nb; b++) {
        stats_print(bands[b], &ss[b]);
    }
    printf("cost GP (%s)\n", mode);
    for (int b = 0; b < nb; b++) {
        stats_print(bands[b], &sc[b]);
    }
    return 0;
}
