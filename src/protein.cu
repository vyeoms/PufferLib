// protein.cu -- Protein hyperparameter optimizer

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <curand.h>
#include <curand_kernel.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PROTEIN_EPSILON 1e-6f
#define PROTEIN_NUM_COST_RATIOS 6
#define PROTEIN_ACQ_MAX_CAP 65536
#define PROTEIN_COST_QUANTILE 0.97f
#define PROTEIN_MIN_OBS_NO_FAIL 100
#define PROTEIN_CLF_ITERS 100
#define PROTEIN_STOP_MIN_RUNS 8
#define NOISE_PRIOR_MU (-4.60517f)
#define NOISE_PRIOR_SIGMA 0.5f
#define SP_LB 1e-4
#define LIN_IDX(np) ((np)-3)
#define SF_IDX(np) ((np)-2)
#define OFF_IDX(np) ((np)-1)
#define GP_BLOCK 16
#define GP_HUTCH_PROBES 8
#define KD_LEAF 16
#define GP_MAX_PARAMS 64
#define GP_FISHER_DAMP 1e-3f
#define GP_FISHER_MAX_STEP 1.0f
#define GP_FISHER_STEPS 3
#define GP_NOISE_FLOOR 0.01f
#define GP_LIN_MEDIAN 0.1f

typedef enum {
    SPACE_LINEAR,
    SPACE_LOG,
    SPACE_POW2,
    SPACE_LOGIT,
} SpaceType;

typedef struct {
    SpaceType type;
    float min, max, scale;
    int is_integer;
} Space;

typedef struct {
    Space *spaces;
    int num;
    int cost_idx;
    int optimize_direction;
} SweepSpace;

// Map raw hyperparam <-> interpolation domain (identity / log10 / log2 / logit).
static inline float space_fwd(SpaceType type, float v) {
    if (type == SPACE_LOG) return log10f(v);
    if (type == SPACE_POW2) return log2f(v);
    if (type == SPACE_LOGIT) return log10f(1.0f - v);
    return v;
}

static inline float space_normalize(const Space *s, float value) {
    if (s->type == SPACE_LOGIT) {
        value = fmaxf(s->min, fminf(value, s->max));
    }
    float t = space_fwd(s->type, value);
    float lo = space_fwd(s->type, s->min);
    float hi = space_fwd(s->type, s->max);
    return 2.0f * ((t - lo) / (hi - lo)) - 1.0f;
}

static inline float space_unnormalize(const Space *s, float norm) {
    float u = (norm + 1.0f) * 0.5f;
    float lo = space_fwd(s->type, s->min);
    float hi = space_fwd(s->type, s->max);
    float t = u * (hi - lo) + lo;
    if (s->type == SPACE_POW2) {
        t = roundf(t);
    }
    float val = t;
    if (s->type == SPACE_LOG) val = powf(10.0f, t);
    if (s->type == SPACE_POW2) val = powf(2.0f, t);
    if (s->type == SPACE_LOGIT) val = 1.0f - powf(10.0f, t);
    if (s->is_integer) {
        val = roundf(val);
    }
    return val;
}

// Softplus + Matern32+linear kernel
__host__ __device__ __forceinline__ float softplus(float x) {
    return (x > 20.0 ? x : log1p(exp(x))) + SP_LB;
}
__host__ __device__ __forceinline__ float inv_softplus(float x) {
    float v = x - SP_LB;
    return v > 20.0 ? v : log(expm1(v));
}
__host__ __device__ __forceinline__ float softplus_grad(float x) {
    return 1.0 / (1.0 + exp(-x));
}

typedef struct {
    int n_params;
    float *raw_params;
} GPKernel;

static inline int gp_grid(int n) {
    return (n + GP_BLOCK - 1) / GP_BLOCK;
}

// Gram / cross-covariance: K = sigma_f * (x·x' + offset + Matern32(r)), or with lin_dim >= 0
// K = lin * x_d·x'_d + sigma_f * (offset + Matern32(r)) for d = lin_dim
__global__ void matern32lin_k_kernel(const float *__restrict__ X1,
        const float *__restrict__ X2, float *__restrict__ K,
        const float *__restrict__ inv_ells, int n, int m, int d, float sigma_f,
        float diag_noise, float offset, float lin, int lin_dim) {
    int col = blockIdx.x * GP_BLOCK + threadIdx.x;
    int row = blockIdx.y * GP_BLOCK + threadIdx.y;
    if (row >= n || col >= m) {
        return;
    }
    float dot = 0.0, r2 = 0.0;
    for (int k = 0; k < d; k++) {
        float xr = X1[row * d + k], xc = X2[col * d + k];
        dot += lin_dim < 0 || k == lin_dim ? xr * xc : 0.0f;
        float diff = (xr - xc) * inv_ells[k];
        r2 += diff * diff;
    }
    float u = sqrt(3.0) * sqrt(r2);
    float lin_dot = lin_dim < 0 ? 0.0f : dot;
    float val = lin * lin_dot + sigma_f * (dot - lin_dot + offset + (1.0 + u) * exp(-u));
    if (diag_noise != 0.0 && row == col) {
        val += diag_noise;
    }
    K[col * n + row] = val;
}

// Fill Z[0..total) with ±1 Rademacher probes, resampled each Fisher step via counter
__global__ void gp_fill_rademacher(float *Z, int total,
        unsigned long long seed, unsigned long long counter) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= total) {
        return;
    }
    curandStatePhilox4_32_10_t st;
    curand_init(seed, i, counter, &st);
    Z[i] = (curand_uniform(&st) < 0.5f) ? -1.0f : 1.0f;
}

// Fused MLL kernel grads: per (i,j) W = alpha alpha^T - (K^-1)_ij terms for d/dell, d/dlin,
// d/dsigma_f, d/doffset (lin_dim < 0 folds the linear term into sigma_f). Summing
// W_ij (dK/dtheta)_ij gives alpha^T (dK/dtheta) alpha - tr(K^-1 dK/dtheta). Kinv holds the lower
// triangle of K^-1. Block partials via block_reduce_sum into out_partials[bid * (d+3) + .]; the
// host folds them in fixed block order. Dynamic smem: (d+3) * GP_BLOCK^2 floats.
__global__ void matern32lin_mll_grad_fuse(const float *__restrict__ X,
        const float *__restrict__ alpha, const float *__restrict__ Kinv,
        const float *__restrict__ inv_ells, int n, int d, float sigma_f, float offset, int lin_dim,
        float *__restrict__ out_partials) {
    extern __shared__ float sh[];
    int col = blockIdx.x * GP_BLOCK + threadIdx.x;
    int row = blockIdx.y * GP_BLOCK + threadIdx.y;
    int tid = threadIdx.y * GP_BLOCK + threadIdx.x;
    int nthreads = GP_BLOCK * GP_BLOCK;
    int nsum = d + 3;

    for (int c = 0; c < nsum; c++) {
        sh[c * nthreads + tid] = 0.0f;
    }
    if (row < n && col < n) {
        float dot = 0.0f, r2 = 0.0f;
        for (int k = 0; k < d; k++) {
            float xr = X[row * d + k], xc = X[col * d + k];
            dot += lin_dim < 0 || k == lin_dim ? xr * xc : 0.0f;
            float t = (xr - xc) * inv_ells[k];
            r2 += t * t;
        }
        float u = sqrtf(3.0f) * sqrtf(r2);
        float e = expf(-u), lin_dot = lin_dim < 0 ? 0.0f : dot;
        float W = alpha[row] * alpha[col] - Kinv[min(row, col) * n + max(row, col)];
        for (int dd = 0; dd < d; dd++) {
            float diff = X[row * d + dd] - X[col * d + dd];
            float inv = inv_ells[dd];
            sh[dd * nthreads + tid] =
                W * (sigma_f * 3.0f * diff * diff * inv * inv * inv * e);
        }
        sh[d * nthreads + tid] = W * lin_dot;
        sh[(d + 1) * nthreads + tid] = W * (dot - lin_dot + offset + (1.0f + u) * e);
        sh[(d + 2) * nthreads + tid] = W;
    }
    int bid = blockIdx.y * gridDim.x + blockIdx.x;
    block_reduce_sum(sh, out_partials + bid * nsum, tid, nthreads, nsum);
}

typedef struct {
    int dim, n, cap;
    float *d_X, *d_y, *d_L, *d_alpha, *d_work;
    int lwork;
    int *d_info;
    float raw_noise;
    float mean;
    cublasHandle_t cublas;
    cusolverDnHandle_t cusolver;
    GPKernel *kernel;
    float *d_inv_ells, *d_partials, *d_Ks, *d_ones;
    float *d_Z, *d_U, *d_G, *d_Kinv, *d_u, *d_spg, *d_F;
    float noise_floor, lin_mu, lin_sigma;
    int lin_dim;
    unsigned long long hutch_seed, hutch_t;
    float *h_partials;
} GaussianProcess;

static void matern32lin_build_K(GaussianProcess *gp, const float *d_X1,
        const float *d_X2, int n, int m, float diag_noise, float *d_K,
        cudaStream_t stream) {
    int d = gp->dim, np = gp->kernel->n_params;
    float h_inv[d];
    for (int i = 0; i < d; i++) {
        h_inv[i] = 1.0f / softplus(gp->kernel->raw_params[i]);
    }
    cudaMemcpyAsync(gp->d_inv_ells, h_inv, d * sizeof(float),
        cudaMemcpyHostToDevice, stream);
    float sigma_f = softplus(gp->kernel->raw_params[SF_IDX(np)]);
    float offset = softplus(gp->kernel->raw_params[OFF_IDX(np)]);
    float lin = softplus(gp->kernel->raw_params[LIN_IDX(np)]);
    dim3 block(GP_BLOCK, GP_BLOCK), grid(gp_grid(m), gp_grid(n));
    matern32lin_k_kernel<<<grid, block, 0, stream>>>(
        d_X1, d_X2, d_K, gp->d_inv_ells, n, m, d, sigma_f, diag_noise, offset, lin, gp->lin_dim);
}

static void gp_init(GaussianProcess *gp, int dim, int cap, int pred_m, float noise) {
    memset(gp, 0, sizeof(*gp));
    gp->lin_dim = -1;
    gp->dim = dim;
    gp->cap = cap;
    int n_params = dim + 3;
    int nsum = dim + 3;
    int nblocks = gp_grid(cap) * gp_grid(cap);
    GPKernel *k = (GPKernel *)malloc(sizeof(GPKernel) + n_params * sizeof(float));
    k->n_params = n_params;
    k->raw_params = (float *)(k + 1);
    for (int i = 0; i < dim; i++) {
        k->raw_params[i] = inv_softplus(1.0f);
    }
    k->raw_params[LIN_IDX(n_params)] = inv_softplus(1.0f);
    k->raw_params[SF_IDX(n_params)] = inv_softplus(1.0f);
    k->raw_params[OFF_IDX(n_params)] = inv_softplus(1.0f);
    gp->kernel = k;
    gp->raw_noise = inv_softplus(noise);
    gp->mean = 0.0f;
    cudaMalloc(&gp->d_X, cap * dim * sizeof(float));
    cudaMalloc(&gp->d_y, cap * sizeof(float));
    cudaMalloc(&gp->d_L, cap * cap * sizeof(float));
    cudaMalloc(&gp->d_alpha, cap * sizeof(float));
    cudaMalloc(&gp->d_info, sizeof(int));
    cudaMalloc(&gp->d_inv_ells, dim * sizeof(float));
    cudaMalloc(&gp->d_Z, cap * GP_HUTCH_PROBES * sizeof(float));
    cudaMalloc(&gp->d_U, cap * GP_HUTCH_PROBES * sizeof(float));
    gp->hutch_seed = 0x9E3779B97F4A7C15ULL;
    cudaMalloc(&gp->d_partials, nblocks * nsum * sizeof(float));
    // Dominant device buffer: cap × acq_cap floats (×2 GPs)
    assert(cudaMalloc(&gp->d_Ks, cap * pred_m * sizeof(float)) == cudaSuccess);
    cudaMalloc(&gp->d_ones, sizeof(float));
    float one = 1.0f;
    cudaMemcpy(gp->d_ones, &one, sizeof(float), cudaMemcpyHostToDevice);
    gp->h_partials = (float *)malloc(nblocks * nsum * sizeof(float));
    int nc = dim + 4;
    cudaMalloc(&gp->d_spg, (nc + nc * nc + cap) * sizeof(float));
    gp->d_F = gp->d_spg + nc;
    gp->d_u = gp->d_F + nc * nc;
    cudaMalloc(&gp->d_G, (size_t)nc * GP_HUTCH_PROBES * cap * sizeof(float));
    cudaMalloc(&gp->d_Kinv, cap * cap * sizeof(float));
    cublasCreate(&gp->cublas);
    cusolverDnCreate(&gp->cusolver);
    int lwork;
    cusolverDnSpotrf_bufferSize(
        gp->cusolver, CUBLAS_FILL_MODE_LOWER, cap, gp->d_L, cap, &gp->lwork);
    cusolverDnSpotri_bufferSize(gp->cusolver, CUBLAS_FILL_MODE_LOWER, cap, gp->d_L, cap, &lwork);
    gp->lwork = max(gp->lwork, lwork);
    cudaMalloc(&gp->d_work, gp->lwork * sizeof(float));
}

static int run_potrf(GaussianProcess *gp, float *d_K, int n, cudaStream_t stream) {
    cusolverDnSetStream(gp->cusolver, stream);
    cusolverDnSpotrf(gp->cusolver, CUBLAS_FILL_MODE_LOWER,
        n, d_K, n, gp->d_work, gp->lwork, gp->d_info);
    int h_info;
    cudaMemcpyAsync(&h_info, gp->d_info, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return h_info;
}

static int gp_recompute(GaussianProcess *gp, cudaStream_t stream) {
    int n = gp->n;
    float noise = softplus(gp->raw_noise) + gp->noise_floor;
    matern32lin_build_K(gp, gp->d_X, gp->d_X, n, n, noise, gp->d_L, stream);
    int info = run_potrf(gp, gp->d_L, n, stream);
    if (info != 0) {
        matern32lin_build_K(gp, gp->d_X, gp->d_X, n, n, noise + 1e-8f, gp->d_L, stream);
        info = run_potrf(gp, gp->d_L, n, stream);
    }
    if (info != 0) {
        return -2;
    }
    // alpha = K^{-1}(y - mean)
    cudaMemcpyAsync(gp->d_alpha, gp->d_y, n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cublasSetStream(gp->cublas, stream);
    float neg_mean = -gp->mean;
    cublasSaxpy(gp->cublas, n, &neg_mean, gp->d_ones, 0, gp->d_alpha, 1);
    cublasStrsv(gp->cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
        n, gp->d_L, n, gp->d_alpha, 1);
    cublasStrsv(gp->cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
        n, gp->d_L, n, gp->d_alpha, 1);
    return 0;
}

static void gp_predict(GaussianProcess *gp, const float *d_Xte, float *d_means, int m,
        cudaStream_t stream) {
    int n = gp->n;
    const float one = 1.0, zero = 0.0;
    cublasSetStream(gp->cublas, stream);
    matern32lin_build_K(gp, gp->d_X, d_Xte, n, m, 0.0f, gp->d_Ks, stream);
    cublasSgemv(gp->cublas, CUBLAS_OP_T, n, m, &one, gp->d_Ks, n, gp->d_alpha, 1,
        &zero, d_means, 1);
    cublasSaxpy(gp->cublas, m, &gp->mean, gp->d_ones, 0, d_means, 1);
}

// G[:, i*P + p] = dK_i V[:, p] for p < P, i < d + 4, with dK_i = dK/draw_i for raw
// [ell_0..ell_{d-1}, lin, sigma_f, offset, noise] computed on the fly
__global__ void matern32lin_dK_matvec(const float *X, const float *V, const float *inv_ells,
        const float *spg, float *G, int n, int d, int P, float sigma_f, float offset,
        int lin_dim) {
    int a = blockIdx.x * BLOCK_SIZE + threadIdx.x, p = blockIdx.y;
    if (a >= n) {
        return;
    }
    const float *v = V + p * n;
    float acc[GP_MAX_PARAMS];
    for (int k = 0; k < d + 3; k++) {
        acc[k] = 0.0f;
    }
    for (int j = 0; j < n; j++) {
        float dot = 0.0f, r2 = 0.0f;
        for (int k = 0; k < d; k++) {
            float xa = X[a * d + k], xj = X[j * d + k];
            dot += lin_dim < 0 || k == lin_dim ? xa * xj : 0.0f;
            float t = (xa - xj) * inv_ells[k];
            r2 += t * t;
        }
        float u = sqrtf(3.0f * r2), e = expf(-u), vj = v[j], lin_dot = lin_dim < 0 ? 0.0f : dot;
        for (int k = 0; k < d; k++) {
            float diff = X[a * d + k] - X[j * d + k], inv = inv_ells[k];
            acc[k] += vj * 3.0f * diff * diff * inv * inv * inv * e;
        }
        acc[d] += vj * lin_dot;
        acc[d + 1] += vj * (dot - lin_dot + offset + (1.0f + u) * e);
        acc[d + 2] += vj;
    }
    int nc = d + 4;
    for (int k = 0; k < nc; k++) {
        float w = k == d || k == d + 1 ? acc[k] : k < nc - 1 ? sigma_f * acc[k] : v[a];
        G[(size_t)(k * P + p) * n + a] = w * spg[k];
    }
}

// Fit kernel hyperparameters by damped Fisher scoring on the log posterior over raw
// [ells, lin, sigma_f, offset, noise, mean]. Fisher_ij = 1/2 tr(S_i S_j) with S_i = L^-1 dK_i L^-T,
// estimated from P Rademacher probes as 1/2 E[(S_i z)^T S_j z] (a Gram matrix, so PSD); the mean
// block is 1^T K^-1 1 (mean and covariance params are Fisher-orthogonal). The gradient is exact
// from K^-1. Priors add Gauss-Newton terms: a dimension-scaled log-normal on each lengthscale
// (Hvarfner et al., ICML 2024: mean shifted by +1/2 log D), log-normals on the noise and, with
// lin_dim >= 0, on the separate linear weight.
static void gp_train(GaussianProcess *gp, const float *X, const float *y, int n,
        cudaStream_t stream) {
    cudaMemcpyAsync(gp->d_X, X, n * gp->dim * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(gp->d_y, y, n * sizeof(float), cudaMemcpyHostToDevice, stream);
    gp->n = n;
    if (gp_recompute(gp, stream) != 0) {
        return;
    }
    int d = gp->dim, n_kp = gp->kernel->n_params, n_opt = n_kp + 2, nc = n_kp + 1, nsum = d + 3;
    int P = GP_HUTCH_PROBES, nn = n * n;
    float ls_mu = 1.4142135f + 0.5f * logf(d), ls_sigma = 1.7320508f;
    float *raw = gp->kernel->raw_params;
    const float one = 1.0f, zero = 0.0f, half_p = 0.5f / P;
    dim3 b(GP_BLOCK, GP_BLOCK), g(gp_grid(n), gp_grid(n));
    int nblocks = g.x * g.y;
    for (int s = 0; s < GP_FISHER_STEPS; s++) {
        float sigma_f = softplus(raw[SF_IDX(n_kp)]), offset = softplus(raw[OFF_IDX(n_kp)]);
        float inv_ells[d], spg[nc];
        for (int i = 0; i < d; i++) {
            inv_ells[i] = 1.0f / softplus(raw[i]);
        }
        for (int i = 0; i < nc - 1; i++) {
            spg[i] = softplus_grad(raw[i]);
        }
        spg[nc - 1] = softplus_grad(gp->raw_noise);
        cudaMemcpyAsync(gp->d_inv_ells, inv_ells, d * sizeof(float), cudaMemcpyHostToDevice,
            stream);
        cudaMemcpyAsync(gp->d_spg, spg, nc * sizeof(float), cudaMemcpyHostToDevice, stream);
        cublasSetStream(gp->cublas, stream);
        cusolverDnSetStream(gp->cusolver, stream);
        // u = L^-1 1 for the mean block
        cudaMemsetAsync(gp->d_u, 0, n * sizeof(float), stream);
        cublasSaxpy(gp->cublas, n, &one, gp->d_ones, 0, gp->d_u, 1);
        cublasStrsv(gp->cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n,
            gp->d_L, n, gp->d_u, 1);
        // Fisher: V = L^-T Z, G[:, i*P + p] = S_i z_p
        gp_fill_rademacher<<<(n * P + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            gp->d_Z, n * P, gp->hutch_seed, gp->hutch_t++);
        cudaMemcpyAsync(gp->d_U, gp->d_Z, n * P * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cublasStrsm(gp->cublas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
            CUBLAS_DIAG_NON_UNIT, n, P, &one, gp->d_L, n, gp->d_U, n);
        dim3 mg((n + BLOCK_SIZE - 1) / BLOCK_SIZE, P);
        matern32lin_dK_matvec<<<mg, BLOCK_SIZE, 0, stream>>>(gp->d_X, gp->d_U, gp->d_inv_ells,
            gp->d_spg, gp->d_G, n, d, P, sigma_f, offset, gp->lin_dim);
        cublasStrsm(gp->cublas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
            CUBLAS_DIAG_NON_UNIT, n, nc * P, &one, gp->d_L, n, gp->d_G, n);
        cublasSgemm(gp->cublas, CUBLAS_OP_T, CUBLAS_OP_N, nc, nc, n * P, &half_p, gp->d_G, n * P,
            gp->d_G, n * P, &zero, gp->d_F, nc);
        // Gradient: alpha^T dK_i alpha - tr(K^-1 dK_i) with K^-1 from potri
        cudaMemcpyAsync(gp->d_Kinv, gp->d_L, nn * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cusolverDnSpotri(gp->cusolver, CUBLAS_FILL_MODE_LOWER, n, gp->d_Kinv, n, gp->d_work,
            gp->lwork, gp->d_info);
        matern32lin_mll_grad_fuse<<<g, b, nsum * GP_BLOCK * GP_BLOCK * sizeof(float), stream>>>(
            gp->d_X, gp->d_alpha, gp->d_Kinv, gp->d_inv_ells, n, d, sigma_f, offset, gp->lin_dim,
            gp->d_partials);
        cudaMemcpyAsync(gp->h_partials, gp->d_partials, nblocks * nsum * sizeof(float),
            cudaMemcpyDeviceToHost, stream);
        float F[nc * nc], gr[nc], aa, tk, sum_u, sum_alpha;
        cublasSdot(gp->cublas, n, gp->d_alpha, 1, gp->d_alpha, 1, &aa);
        cublasSdot(gp->cublas, n, gp->d_Kinv, n + 1, gp->d_ones, 0, &tk);
        cublasSdot(gp->cublas, n, gp->d_u, 1, gp->d_u, 1, &sum_u);
        cublasSdot(gp->cublas, n, gp->d_alpha, 1, gp->d_ones, 0, &sum_alpha);
        cudaMemcpyAsync(F, gp->d_F, nc * nc * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        for (int c = 0; c < nsum; c++) {
            gr[c] = 0.0f;
            for (int bi = 0; bi < nblocks; bi++) {
                gr[c] += gp->h_partials[bi * nsum + c];
            }
            gr[c] *= 0.5f * spg[c] * (c == nsum - 1 ? sigma_f : 1.0f);
        }
        gr[nc - 1] = 0.5f * (aa - tk) * spg[nc - 1];

        float H[n_opt * n_opt], step[n_opt];
        memset(H, 0, sizeof(H));
        for (int i = 0; i < nc; i++) {
            for (int j = 0; j < nc; j++) {
                H[i * n_opt + j] = F[i * nc + j];
            }
            step[i] = gr[i];
        }
        H[n_opt * n_opt - 1] = sum_u;
        step[n_opt - 1] = sum_alpha;
        for (int k = 0; k < nc; k++) {
            int lin = k == LIN_IDX(n_kp) && gp->lin_dim >= 0;
            if (k >= d && k < nc - 1 && !lin) {
                continue;
            }
            float v = k < nc - 1 ? softplus(raw[k]) : softplus(gp->raw_noise) + gp->noise_floor;
            float mu = k < d ? ls_mu : lin ? gp->lin_mu : NOISE_PRIOR_MU;
            float sg = k < d ? ls_sigma : lin ? gp->lin_sigma : NOISE_PRIOR_SIGMA;
            float jac = spg[k] / v, z = (logf(v) - mu) / sg;
            step[k] += (-z / sg - 1.0f) * jac;
            H[k * n_opt + k] += jac * jac / (sg * sg);
        }

        // step = (H damped)^-1 grad via in-place row-major Cholesky
        for (int i = 0; i < n_opt; i++) {
            H[i * n_opt + i] = H[i * n_opt + i] * (1.0f + GP_FISHER_DAMP) + 1e-6f;
        }
        for (int j = 0; j < n_opt; j++) {
            for (int i = j; i < n_opt; i++) {
                float acc = H[i * n_opt + j];
                for (int k = 0; k < j; k++) {
                    acc -= H[i * n_opt + k] * H[j * n_opt + k];
                }
                H[i * n_opt + j] = i == j ? sqrtf(acc) : acc / H[j * n_opt + j];
            }
        }
        for (int i = 0; i < n_opt; i++) {
            for (int k = 0; k < i; k++) {
                step[i] -= H[i * n_opt + k] * step[k];
            }
            step[i] /= H[i * n_opt + i];
        }
        for (int i = n_opt - 1; i >= 0; i--) {
            for (int k = i + 1; k < n_opt; k++) {
                step[i] -= H[k * n_opt + i] * step[k];
            }
            step[i] /= H[i * n_opt + i];
        }
        for (int i = 0; i < n_opt; i++) {
            float *p = i < n_kp ? &raw[i] : i == n_kp ? &gp->raw_noise : &gp->mean;
            *p += fmaxf(-GP_FISHER_MAX_STEP, fminf(GP_FISHER_MAX_STEP, step[i]));
        }
        if (gp_recompute(gp, stream) != 0) {
            break;
        }
    }
}

// Candidate sampling (score / argmax run on host after GP predict)
__global__ void protein_k_init_rng(curandStatePhilox4_32_10_t *states, int n,
        unsigned long long seed) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) {
        curand_init(seed, i, 0, &states[i]);
    }
}

__global__ void protein_k_sample(float *__restrict__ candidates,
        const float *__restrict__ centers, const float *__restrict__ scales,
        curandStatePhilox4_32_10_t *__restrict__ rng, int n_candidates,
        int n_centers, int dim, float global_scale, int cost_dim,
        float fixed_cost) {
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= n_candidates) {
        return;
    }
    curandStatePhilox4_32_10_t state = rng[i];
    int center = ((int)(curand_uniform(&state) * n_centers)) % n_centers;
    for (int d = 0; d < dim; d++) {
        float s = scales[d] * global_scale;
        float val = s * (2.0f * curand_uniform(&state) - 1.0f)
            + centers[center * dim + d];
        val = fmaxf(-1.0f, fminf(1.0f, val));
        candidates[i * dim + d] = val;
    }
    if (cost_dim >= 0 && !isnan(fixed_cost)) {
        candidates[i * dim + cost_dim] = fixed_cost;
    }
    rng[i] = state;
}

typedef struct {
    float predicted_score, predicted_cost, rating;
    int n_pareto, n_gp_obs, n_candidates, is_random;
} ProteinSweepInfo;

typedef struct {
    int lo, hi, left, right, split_dim;
    float split_val;
} KDNode;

typedef struct {
    // Caller populates config variables
    SweepSpace *space;
    int num_random_samples;
    int suggestions_per_pareto;
    int gp_max_obs;
    int infer_batch_size;
    int use_success_prob;
    int prune_pareto;
    int use_logit;
    float global_search_scale;
    float max_suggestion_cost;
    float expansion_rate;
    float cost_random_suggestion;
    float early_stop_frac;
    int success_cap;
    int failure_cap;
    int top_k;
    unsigned long long rng_seed;
    // > 0: separate score-GP linear weight over the cost dim only, log-normal prior with this
    // log-sd. 0: linear over all dims, tied to sigma_f
    float gp_lin_sigma;
    // Internal
    float *succ_params, *succ_scores, *succ_costs;
    float *fail_params, *fail_scores, *fail_costs;
    float *check_scores, *check_costs, *check_finals, *check_front;
    int succ_n, fail_n, check_n, dim, cost_dim;
    int *top_idx;
    float min_score, max_score, log_c_min, log_c_max;
    int acq_cap;
    float *d_scales, *d_candidates, *d_pred_y, *d_pred_c;
    curandStatePhilox4_32_10_t *d_rng;
    float *clf_w, clf_bias;
    int clf_fitted;
    GaussianProcess gp_score, gp_cost;
    curandGenerator_t sobol;
    cudaStream_t stream;
    int suggestion_idx;
    float ratio_pool[PROTEIN_NUM_COST_RATIOS];
    int pool_remaining;
    float *gp_train_params, *gp_train_y, *gp_train_c;
    int *pareto_buf, *pruned_buf;
    float *centers_buf;
    // suggest scratch (sized to caps; ext_buf/kept_buf also classifier)
    float *log_c_buf, *ext_buf, *h_cands, *h_pred;
    int *kept_buf, *kd_idx, *kd_keep;
    KDNode *kd_nodes;
    float *d_centers;
} ProteinSweep;

ProteinSweep *protein_sweep_create(ProteinSweep init) {
    ProteinSweep *sw = (ProteinSweep *)malloc(sizeof(ProteinSweep));
    *sw = init;
    int dim = sw->space->num;
    sw->dim = dim;
    sw->cost_dim = sw->space->cost_idx;
    sw->succ_n = sw->fail_n = sw->check_n = sw->suggestion_idx = 0;
    sw->clf_fitted = sw->pool_remaining = 0;
    sw->min_score = FLT_MAX;
    sw->max_score = -FLT_MAX;
    sw->log_c_min = FLT_MAX;
    sw->log_c_max = -FLT_MAX;
    sw->clf_bias = 0.0f;

    static const float default_ratios[PROTEIN_NUM_COST_RATIOS] = {
        0.16f, 0.32f, 0.48f, 0.64f, 0.8f, 1.0f};
    memcpy(sw->ratio_pool, default_ratios, sizeof(default_ratios));

    sw->succ_params = (float *)calloc(sw->success_cap * (dim + 2), sizeof(float));
    sw->succ_scores = sw->succ_params + sw->success_cap * dim;
    sw->succ_costs = sw->succ_scores + sw->success_cap;
    sw->fail_params = (float *)calloc(sw->failure_cap * (dim + 2), sizeof(float));
    sw->fail_scores = sw->fail_params + sw->failure_cap * dim;
    sw->fail_costs = sw->fail_scores + sw->failure_cap;
    sw->check_scores = (float *)malloc(4 * sw->success_cap * sizeof(float));
    sw->check_costs = sw->check_scores + sw->success_cap;
    sw->check_finals = sw->check_costs + sw->success_cap;
    sw->check_front = sw->check_finals + sw->success_cap;
    sw->top_idx = (int *)malloc(sw->top_k * sizeof(int));

    int max_centers = sw->success_cap + 3 * sw->top_k;
    int acq_cap = max_centers * sw->suggestions_per_pareto;
    if (acq_cap < sw->infer_batch_size) acq_cap = sw->infer_batch_size;
    if (acq_cap > PROTEIN_ACQ_MAX_CAP) acq_cap = PROTEIN_ACQ_MAX_CAP;
    sw->acq_cap = acq_cap;
    int max_ext = sw->success_cap + sw->failure_cap;
    int kd_cap = max_ext > acq_cap ? max_ext : acq_cap;
    float scales[dim];
    for (int i = 0; i < dim; i++) {
        scales[i] = sw->space->spaces[i].scale;
    }
    cudaMalloc(&sw->d_scales, dim * sizeof(float));
    cudaMemcpy(sw->d_scales, scales, dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&sw->d_candidates, acq_cap * dim * sizeof(float));
    cudaMalloc(&sw->d_pred_y, 2 * acq_cap * sizeof(float));
    sw->d_pred_c = sw->d_pred_y + acq_cap;
    cudaMalloc(&sw->d_rng, acq_cap * sizeof(curandStatePhilox4_32_10_t));
    cudaMalloc(&sw->d_centers, max_centers * dim * sizeof(float));
    protein_k_init_rng<<<(acq_cap + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        sw->d_rng, acq_cap, sw->rng_seed);

    sw->clf_w = (float *)calloc(dim, sizeof(float));
    gp_init(&sw->gp_score, dim, sw->gp_max_obs, acq_cap, 0.01f);
    gp_init(&sw->gp_cost, dim, sw->gp_max_obs, acq_cap, 0.01f);
    sw->gp_score.noise_floor = GP_NOISE_FLOOR;
    sw->gp_score.lin_mu = logf(GP_LIN_MEDIAN);
    sw->gp_score.lin_sigma = sw->gp_lin_sigma;
    sw->gp_score.lin_dim = sw->gp_lin_sigma > 0 ? sw->cost_dim : -1;

    curandCreateGeneratorHost(&sw->sobol, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32);
    curandSetQuasiRandomGeneratorDimensions(sw->sobol, dim);
    curandSetGeneratorOffset(sw->sobol, sw->rng_seed);

    sw->gp_train_params = (float *)malloc(sw->gp_max_obs * (dim + 2) * sizeof(float));
    sw->gp_train_y = sw->gp_train_params + sw->gp_max_obs * dim;
    sw->gp_train_c = sw->gp_train_y + sw->gp_max_obs;
    sw->pareto_buf = (int *)malloc(2 * sw->success_cap * sizeof(int));
    sw->pruned_buf = sw->pareto_buf + sw->success_cap;
    sw->centers_buf = (float *)malloc(max_centers * dim * sizeof(float));
    sw->log_c_buf = (float *)malloc(sw->success_cap * sizeof(float));
    sw->ext_buf = (float *)malloc(max_ext * (dim + 2) * sizeof(float));
    sw->kept_buf = (int *)malloc(kd_cap * sizeof(int));
    sw->h_cands = (float *)malloc(acq_cap * dim * sizeof(float));
    sw->h_pred = (float *)malloc(2 * acq_cap * sizeof(float));
    sw->kd_idx = (int *)malloc(kd_cap * sizeof(int));
    sw->kd_nodes = (KDNode *)malloc(2 * kd_cap * sizeof(KDNode));
    sw->kd_keep = (int *)malloc(kd_cap * sizeof(int));
    cudaStreamCreate(&sw->stream);
    return sw;
}

static float logit_transform(float value) {
    value = fmaxf(1e-9f, fminf(1.0f - 1e-9f, value));
    return fmaxf(-5.0f, logf(value / (1.0f - value)));
}

void protein_sweep_observe(ProteinSweep *sw, const float *norm_params,
        float score, float cost, int is_failure) {
    if (sw->use_logit) {
        score = logit_transform(score);
    }
    score *= sw->space->optimize_direction;  // stored so that higher is better
    int dim = sw->dim;
    if (is_failure || !isfinite(score)) {
        if (sw->fail_n >= sw->failure_cap) {
            return;
        }
        int i = sw->fail_n++;
        memcpy(&sw->fail_params[i * dim], norm_params, dim * sizeof(float));
        sw->fail_scores[i] = score;
        sw->fail_costs[i] = cost;
        return;
    }
    for (int i = 0; i < sw->succ_n; i++) {
        float dist2 = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = norm_params[d] - sw->succ_params[i * dim + d];
            dist2 += diff * diff;
        }
        if (dist2 < PROTEIN_EPSILON * PROTEIN_EPSILON) {
            memcpy(&sw->succ_params[i * dim], norm_params, dim * sizeof(float));
            sw->succ_scores[i] = score;
            sw->succ_costs[i] = cost;
            return;
        }
    }
    if (sw->succ_n >= sw->success_cap) {
        return;
    }
    int idx = sw->succ_n++;
    memcpy(&sw->succ_params[idx * dim], norm_params, dim * sizeof(float));
    sw->succ_scores[idx] = score;
    sw->succ_costs[idx] = cost;
}

static float noise01(void) {
    float u1 = (rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = (rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    return 0.1f * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

static void sobol_fallback(ProteinSweep *sw, float *out, int is_fixed_cost,
        float fixed_cost_norm) {
    int dim = sw->dim, cost_dim = sw->cost_dim;
    curandGenerateUniform(sw->sobol, out, dim);
    for (int d = 0; d < dim; d++) {
        out[d] = 2.0f * out[d] - 1.0f;
    }
    if (cost_dim < 0) {
        return;
    }
    if (is_fixed_cost) {
        out[cost_dim] = fixed_cost_norm;
    } else {
        out[cost_dim] = fmaxf(-1.0f,
            fminf(1.0f, sw->cost_random_suggestion + noise01()));
    }
}

// KD Tree near-duplicate filter
typedef struct {
    const float *X;
    int dim, n_nodes;
    int *idx;
    KDNode *nodes;
} KDTree;

static int g_kd_split_dim, g_kd_d;
static const float *g_kd_X;

static int kd_cmp_fn(const void *a, const void *b) {
    float va = g_kd_X[*(const int *)a * g_kd_d + g_kd_split_dim];
    float vb = g_kd_X[*(const int *)b * g_kd_d + g_kd_split_dim];
    return (va > vb) - (va < vb);
}

static void kd_build(KDTree *t, int node, int lo, int hi) {
    KDNode *nd = &t->nodes[node];
    nd->lo = lo;
    nd->hi = hi;
    nd->left = nd->right = -1;
    nd->split_dim = -1;
    if (hi - lo <= KD_LEAF) {
        return;
    }
    int best = 0;
    float spread = -1.0;
    for (int k = 0; k < t->dim; k++) {
        float mn = t->X[t->idx[lo] * t->dim + k], mx = mn;
        for (int i = lo + 1; i < hi; i++) {
            float v = t->X[t->idx[i] * t->dim + k];
            mn = fminf(mn, v);
            mx = fmaxf(mx, v);
        }
        if (mx - mn > spread) {
            spread = mx - mn;
            best = k;
        }
    }
    int mid = (lo + hi) / 2;
    g_kd_split_dim = best;
    g_kd_d = t->dim;
    g_kd_X = t->X;
    qsort(t->idx + lo, hi - lo, sizeof(int), kd_cmp_fn);
    nd->split_dim = best;
    nd->split_val = t->X[t->idx[mid] * t->dim + best];
    int ln = t->n_nodes++, rn = t->n_nodes++;
    nd->left = ln;
    nd->right = rn;
    kd_build(t, ln, lo, mid);
    kd_build(t, rn, mid, hi);
}

static void kd_remove_near(const KDTree *t, int node, const float *q, float r,
        float r2, int self, int *keep) {
    const KDNode *nd = &t->nodes[node];
    if (nd->split_dim < 0) {
        for (int i = nd->lo; i < nd->hi; i++) {
            int p = t->idx[i];
            float sq = 0.0;
            for (int k = 0; k < t->dim; k++) {
                float df = q[k] - t->X[p * t->dim + k];
                sq += df * df;
            }
            if (p != self && sq <= r2) {
                keep[p] = 0;
            }
        }
        return;
    }
    float dv = q[nd->split_dim] - nd->split_val;
    if (dv <= r) {
        kd_remove_near(t, nd->left, q, r, r2, self, keep);
    }
    if (dv >= -r) {
        kd_remove_near(t, nd->right, q, r, r2, self, keep);
    }
}

static int filter_near_duplicates(ProteinSweep *sw, const float *X,
        int n, int dim, float threshold, int *kept_indices) {
    KDTree tree = {
        .X = X, .dim = dim, .n_nodes = 1, .idx = sw->kd_idx, .nodes = sw->kd_nodes};
    for (int i = 0; i < n; i++) {
        sw->kd_idx[i] = i;
    }
    kd_build(&tree, 0, 0, n);
    float r2 = threshold * threshold;
    for (int i = 0; i < n; i++) {
        sw->kd_keep[i] = 1;
    }
    for (int i = n - 1; i >= 0; i--) {
        if (sw->kd_keep[i]) {
            kd_remove_near(&tree, 0, &X[i * dim], threshold, r2, i, sw->kd_keep);
        }
    }
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (sw->kd_keep[i]) {
            kept_indices[count++] = i;
        }
    }
    return count;
}

static int float_cmp(const void *a, const void *b) {
    float fa = *(const float *)a, fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

// Score a run must reach by early_stop_frac of its steps to continue: the lowest check of the runs
// within a factor of 2 of its cost that went on to improve the front, so a run is stopped only when
// no similar-cost run that looked this bad ever mattered. -FLT_MAX never stops.
float protein_sweep_stop_score(ProteinSweep *sw, float cost) {
    float th = FLT_MAX;
    int n = 0;
    for (int i = 0; i < sw->check_n; i++) {
        if (fabsf(logf(sw->check_costs[i] / cost)) < (float)M_LN2) {
            n++;
            th = sw->check_front[i] ? fminf(th, sw->check_scores[i]) : th;
        }
    }
    if (n < PROTEIN_STOP_MIN_RUNS || th == FLT_MAX || sw->early_stop_frac <= 0.0f) {
        return -FLT_MAX;
    }
    return th;
}

// Final score for a stopped run: that of the finished similar-cost run with the same rank at the
// check, since a partial score would understate the config to the GP.
float protein_sweep_stop_estimate(ProteinSweep *sw, float check, float cost) {
    float *finals = sw->log_c_buf;
    int n = 0, below = 0;
    for (int i = 0; i < sw->check_n; i++) {
        int near = fabsf(logf(sw->check_costs[i] / cost)) < (float)M_LN2;
        if (near && isfinite(sw->check_finals[i])) {
            finals[n++] = sw->check_finals[i];
            below += sw->check_scores[i] < check;
        }
    }
    if (!n) {
        return check;
    }
    qsort(finals, n, sizeof(float), float_cmp);
    return finals[min(below, n - 1)];
}

// final is NAN for a stopped run
void protein_sweep_observe_check(ProteinSweep *sw, float check, float cost, float final) {
    int front = isfinite(final);
    for (int i = 0; i < sw->check_n && front; i++) {
        front = !(sw->check_costs[i] <= cost && sw->check_finals[i] >= final);
    }
    sw->check_front[sw->check_n] = front;
    sw->check_scores[sw->check_n] = check;
    sw->check_costs[sw->check_n] = cost;
    sw->check_finals[sw->check_n++] = final;
}

static const float *g_pareto_costs;
static int pareto_cmp(const void *a, const void *b) {
    float ca = g_pareto_costs[*(const int *)a];
    float cb = g_pareto_costs[*(const int *)b];
    return (ca > cb) - (ca < cb);
}

ProteinSweepInfo protein_sweep_suggest(ProteinSweep *sw,
        float *out, float fixed_cost_norm) {
    ProteinSweepInfo info = {0};
    sw->suggestion_idx++;
    int dim = sw->dim, cost_dim = sw->cost_dim;
    int is_fixed_cost = !isnan(fixed_cost_norm);

    if (sw->suggestion_idx <= sw->num_random_samples || sw->succ_n == 0) {
        sobol_fallback(sw, out, is_fixed_cost, fixed_cost_norm);
        info.is_random = 1;
        return info;
    }

    // sample observations for GP
    sw->min_score = sw->max_score = sw->succ_scores[0];
    sw->log_c_min = logf(fmaxf(sw->succ_costs[0], PROTEIN_EPSILON));
    sw->log_c_buf[0] = sw->log_c_min;
    for (int i = 1; i < sw->succ_n; i++) {
        float s = sw->succ_scores[i];
        sw->min_score = fminf(sw->min_score, s);
        sw->max_score = fmaxf(sw->max_score, s);
        sw->log_c_buf[i] = logf(fmaxf(sw->succ_costs[i], PROTEIN_EPSILON));
        sw->log_c_min = fminf(sw->log_c_min, sw->log_c_buf[i]);
    }
    qsort(sw->log_c_buf, sw->succ_n, sizeof(float), float_cmp);
    float qidx = PROTEIN_COST_QUANTILE * (sw->succ_n - 1);
    int qlo = (int)qidx, qhi = qlo + 1 < sw->succ_n ? qlo + 1 : sw->succ_n - 1;
    float qfrac = qidx - qlo;
    sw->log_c_max = sw->log_c_buf[qlo] * (1.0f - qfrac) + sw->log_c_buf[qhi] * qfrac;

    int use_failures = (sw->succ_n < PROTEIN_MIN_OBS_NO_FAIL && sw->fail_n > 0);
    int combined_n = use_failures ? sw->fail_n + sw->succ_n : sw->succ_n;
    int ext_dim = dim + 2;
    int ci = 0;
    if (use_failures) {
        for (int i = 0; i < sw->fail_n; i++) {
            memcpy(&sw->ext_buf[ci * ext_dim], &sw->fail_params[i * dim],
                dim * sizeof(float));
            sw->ext_buf[ci * ext_dim + dim] = sw->min_score;
            sw->ext_buf[ci * ext_dim + dim + 1] = sw->fail_costs[i];
            ci++;
        }
    }
    for (int i = 0; i < sw->succ_n; i++) {
        memcpy(&sw->ext_buf[ci * ext_dim], &sw->succ_params[i * dim],
            dim * sizeof(float));
        sw->ext_buf[ci * ext_dim + dim] = sw->succ_scores[i];
        sw->ext_buf[ci * ext_dim + dim + 1] = sw->succ_costs[i];
        ci++;
    }
    int n_kept = filter_near_duplicates(
        sw, sw->ext_buf, combined_n, ext_dim, PROTEIN_EPSILON, sw->kept_buf);
    if (n_kept > sw->gp_max_obs) {
        int recent_size = (int)(0.5f * sw->gp_max_obs);
        int older_size = n_kept - recent_size;
        int num_sample = sw->gp_max_obs - recent_size;
        for (int i = 0; i < num_sample; i++) {
            int j = i + rand() % (older_size - i);
            int tmp = sw->kept_buf[i];
            sw->kept_buf[i] = sw->kept_buf[j];
            sw->kept_buf[j] = tmp;
        }
        memmove(sw->kept_buf + num_sample, sw->kept_buf + older_size,
            recent_size * sizeof(int));
        n_kept = sw->gp_max_obs;
    }
    for (int i = 0; i < n_kept; i++) {
        int idx = sw->kept_buf[i];
        memcpy(&sw->gp_train_params[i * dim], &sw->ext_buf[idx * ext_dim],
            dim * sizeof(float));
        sw->gp_train_y[i] = sw->ext_buf[idx * ext_dim + dim];
        sw->gp_train_c[i] = sw->ext_buf[idx * ext_dim + dim + 1];
    }
    int n_gp = n_kept;

    float min_score = sw->min_score, max_score = sw->max_score;
    float log_c_min = sw->log_c_min, log_c_max = sw->log_c_max;
    float score_range = fabsf(max_score - min_score) + PROTEIN_EPSILON;
    float cost_range = log_c_max - log_c_min + PROTEIN_EPSILON;
    for (int i = 0; i < n_gp; i++) {
        sw->gp_train_y[i] = (sw->gp_train_y[i] - min_score) / score_range;
        float lc = logf(fmaxf(sw->gp_train_c[i], PROTEIN_EPSILON));
        sw->gp_train_c[i] = (lc - log_c_min) / cost_range;
    }

    // Train GPs
    gp_train(&sw->gp_score, sw->gp_train_params, sw->gp_train_y, n_gp, sw->stream);
    gp_train(&sw->gp_cost, sw->gp_train_params, sw->gp_train_c, n_gp, sw->stream);

    // Pareto front + prune
    for (int i = 0; i < sw->succ_n; i++) {
        sw->pruned_buf[i] = i;
    }
    g_pareto_costs = sw->succ_costs;
    qsort(sw->pruned_buf, sw->succ_n, sizeof(int), pareto_cmp);
    int n_pareto = 0;
    float max_ps = -FLT_MAX;
    for (int i = 0; i < sw->succ_n; i++) {
        int idx = sw->pruned_buf[i];
        if (sw->succ_scores[idx] > max_ps + PROTEIN_EPSILON) {
            sw->pareto_buf[n_pareto++] = idx;
            max_ps = sw->succ_scores[idx];
        }
    }
    memcpy(sw->pruned_buf, sw->pareto_buf, n_pareto * sizeof(int));
    int n_pruned = n_pareto;
    if (n_pruned >= 2) {
        float s_min = sw->succ_scores[sw->pruned_buf[0]], s_max = s_min;
        float c_min = sw->succ_costs[sw->pruned_buf[0]], c_max = c_min;
        for (int i = 1; i < n_pruned; i++) {
            float ss = sw->succ_scores[sw->pruned_buf[i]];
            float cc = sw->succ_costs[sw->pruned_buf[i]];
            s_min = fminf(s_min, ss);
            s_max = fmaxf(s_max, ss);
            c_min = fminf(c_min, cc);
            c_max = fmaxf(c_max, cc);
        }
        float s_range = fmaxf(s_max - s_min, PROTEIN_EPSILON);
        float c_range2 = fmaxf(c_max - c_min, PROTEIN_EPSILON);
        float max_pareto_s = sw->succ_scores[sw->pruned_buf[n_pruned - 1]];
        for (int i = n_pruned - 1; i > 1; i--) {
            float prev = sw->succ_scores[sw->pruned_buf[i - 1]];
            if (max_pareto_s - prev > 0.02f * fabsf(max_pareto_s)) {
                break;
            }
            float ng = (sw->succ_scores[sw->pruned_buf[i]] - prev) / s_range;
            float nc = (sw->succ_costs[sw->pruned_buf[i]]
                - sw->succ_costs[sw->pruned_buf[i - 1]]) / c_range2;
            if (ng / (nc + PROTEIN_EPSILON) >= 0.5f) {
                break;
            }
            n_pruned--;
        }
    }

    int *use_pareto = sw->prune_pareto ? sw->pruned_buf : sw->pareto_buf;
    int n_use = sw->prune_pareto ? n_pruned : n_pareto;

    // Top-k by score
    int n_top = 0;
    for (int i = 0; i < sw->succ_n; i++) {
        int j = n_top;
        while (j > 0 && sw->succ_scores[i] > sw->succ_scores[sw->top_idx[j - 1]]) {
            j--;
        }
        if (j == sw->top_k) {
            continue;
        }
        n_top = min(n_top + 1, sw->top_k);
        memmove(&sw->top_idx[j + 1], &sw->top_idx[j], (n_top - 1 - j) * sizeof(int));
        sw->top_idx[j] = i;
    }

    // Search centers
    int n_centers = 0;
    for (int i = 0; i < n_use; i++) {
        memcpy(&sw->centers_buf[n_centers++ * dim],
            &sw->succ_params[use_pareto[i] * dim], dim * sizeof(float));
    }
    float divisors[] = {2.0f, 3.0f};
    int n_shift = (cost_dim >= 0) ? 2 : 1;
    for (int r = 0; r < n_shift; r++) {
        for (int i = 0; i < n_top; i++) {
            const float *src = &sw->succ_params[sw->top_idx[i] * dim];
            float *dst = &sw->centers_buf[n_centers * dim];
            memcpy(dst, src, dim * sizeof(float));
            if (cost_dim >= 0) {
                float orig = src[cost_dim];
                dst[cost_dim] = orig - (orig + 1.0f) / divisors[r];
            }
            n_centers++;
        }
    }

    // Target cost
    float target_cost = 0.0f;
    if (!is_fixed_cost) {
        if (sw->pool_remaining <= 0) {
            for (int i = PROTEIN_NUM_COST_RATIOS - 1; i > 0; i--) {
                int j = rand() % (i + 1);
                float tmp = sw->ratio_pool[i];
                sw->ratio_pool[i] = sw->ratio_pool[j];
                sw->ratio_pool[j] = tmp;
            }
            sw->pool_remaining = PROTEIN_NUM_COST_RATIOS;
        }
        float ratio = sw->ratio_pool[--sw->pool_remaining];
        ratio = fmaxf(0.0f, fminf(1.0f, ratio + noise01()));
        target_cost = (1.0f + sw->expansion_rate) * ratio;
    }

    // Sample candidates
    int n_total = n_centers * sw->suggestions_per_pareto;
    if (n_total > sw->acq_cap) {
        n_total = sw->acq_cap;
    }
    cudaMemcpyAsync(sw->d_centers, sw->centers_buf, n_centers * dim * sizeof(float),
        cudaMemcpyHostToDevice, sw->stream);
    protein_k_sample<<<(n_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, sw->stream>>>(
        sw->d_candidates, sw->d_centers, sw->d_scales, sw->d_rng, n_total,
        n_centers, dim, sw->global_search_scale,
        cost_dim, is_fixed_cost ? fixed_cost_norm : NAN);

    cudaMemcpyAsync(sw->h_cands, sw->d_candidates, n_total * dim * sizeof(float),
        cudaMemcpyDeviceToHost, sw->stream);
    cudaStreamSynchronize(sw->stream);
    int n_cands = filter_near_duplicates(
        sw, sw->h_cands, n_total, dim, PROTEIN_EPSILON, sw->kept_buf);
    if (n_cands < n_total) {
        for (int i = 0; i < n_cands; i++) {
            memmove(&sw->h_cands[i * dim], &sw->h_cands[sw->kept_buf[i] * dim],
                dim * sizeof(float));
        }
        cudaMemcpyAsync(sw->d_candidates, sw->h_cands, n_cands * dim * sizeof(float),
            cudaMemcpyHostToDevice, sw->stream);
    }

    if (n_cands == 0) {
        sobol_fallback(sw, out, is_fixed_cost, fixed_cost_norm);
        info.is_random = 1;
        return info;
    }

    // Success classifier
    sw->clf_fitted = 0;
    if (sw->use_success_prob && sw->succ_n > 9 && sw->fail_n > 9) {
        int n_s = sw->succ_n, n_f = sw->fail_n, n_clf = n_s + n_f;
        float *X = sw->ext_buf;
        int *yy = sw->kept_buf;
        memcpy(X, sw->succ_params, n_s * dim * sizeof(float));
        memcpy(X + n_s * dim, sw->fail_params, n_f * dim * sizeof(float));
        for (int i = 0; i < n_s; i++) {
            yy[i] = 1;
        }
        for (int i = 0; i < n_f; i++) {
            yy[n_s + i] = 0;
        }
        int n_pos = 0;
        for (int i = 0; i < n_clf; i++) {
            n_pos += yy[i];
        }
        int n_neg = n_clf - n_pos;
        if (n_pos > 0 && n_neg > 0) {
            float w_pos = n_clf / (2.0f * n_pos);
            float w_neg = n_clf / (2.0f * n_neg);
            memset(sw->clf_w, 0, dim * sizeof(float));
            sw->clf_bias = 0.0f;
            float grad[dim];
            for (int iter = 0; iter < PROTEIN_CLF_ITERS; iter++) {
                float lr = 0.1f / (1.0f + 0.01f * iter);
                for (int d = 0; d < dim; d++) {
                    grad[d] = 0.0f;
                }
                float grad_b = 0.0f;
                for (int i = 0; i < n_clf; i++) {
                    float z = sw->clf_bias;
                    for (int d = 0; d < dim; d++) {
                        z += X[i * dim + d] * sw->clf_w[d];
                    }
                    float sig = 1.0f / (1.0f + expf(-z));
                    float err = (yy[i] ? w_pos : w_neg) * (sig - yy[i]) / n_clf;
                    for (int d = 0; d < dim; d++) {
                        grad[d] += err * X[i * dim + d];
                    }
                    grad_b += err;
                }
                for (int d = 0; d < dim; d++) {
                    sw->clf_w[d] -= lr * (grad[d] + sw->clf_w[d]);
                }
                sw->clf_bias -= lr * grad_b;
            }
            sw->clf_fitted = 1;
        }
    }

    // GP predict + score/argmax
    gp_predict(&sw->gp_score, sw->d_candidates, sw->d_pred_y, n_cands, sw->stream);
    gp_predict(&sw->gp_cost, sw->d_candidates, sw->d_pred_c, n_cands, sw->stream);
    cudaMemcpyAsync(sw->h_pred, sw->d_pred_y, n_cands * sizeof(float),
        cudaMemcpyDeviceToHost, sw->stream);
    cudaMemcpyAsync(sw->h_pred + n_cands, sw->d_pred_c, n_cands * sizeof(float),
        cudaMemcpyDeviceToHost, sw->stream);
    cudaStreamSynchronize(sw->stream);

    int best = 0;
    float best_s = -FLT_MAX;
    for (int i = 0; i < n_cands; i++) {
        float s = sw->h_pred[i];
        if (!is_fixed_cost) {
            float cn = sw->h_pred[n_cands + i];
            float c = expf(cn * (log_c_max - log_c_min) + log_c_min);
            s *= (c < sw->max_suggestion_cost ? 1.0f : 0.0f)
                * (1.0f - fabsf(target_cost - cn));
        }
        if (sw->clf_fitted) {
            float z = sw->clf_bias;
            for (int d = 0; d < dim; d++) {
                z += sw->h_cands[i * dim + d] * sw->clf_w[d];
            }
            s *= 1.0f / (1.0f + expf(-z));
        }
        if (s > best_s) {
            best_s = s;
            best = i;
        }
    }
    memcpy(out, &sw->h_cands[best * dim], dim * sizeof(float));
    float y_norm = sw->h_pred[best], c_norm = sw->h_pred[n_cands + best];

    info.predicted_score = sw->space->optimize_direction
        * (y_norm * (max_score - min_score) + min_score);
    info.predicted_cost = expf(c_norm * (log_c_max - log_c_min) + log_c_min);
    info.rating = best_s;
    info.n_pareto = n_use;
    info.n_gp_obs = n_gp;
    info.n_candidates = n_cands;
    return info;
}
