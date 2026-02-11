#include <metal_stdlib>
using namespace metal;

// ===================================================================
// SIMD-optimized Q8_0 matvec — 32 threads per row using simd_sum
// Each simdgroup cooperatively computes one row's dot product.
// Multiple simdgroups per threadgroup for better GPU occupancy.
// ===================================================================

kernel void q8_matvec(
    device const uchar *W [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y       [[buffer(2)]],
    constant uint &nb     [[buffer(3)]],
    constant uint &rows   [[buffer(4)]],
    uint row  [[threadgroup_position_in_grid]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    // Each simdgroup (32 lanes) computes one additional row
    uint actual_row = row * 8 + sg;  // 8 simdgroups per threadgroup
    if (actual_row >= rows) return;

    device const uchar *rp = W + actual_row * nb * 34u;
    float sum = 0.0f;

    for (uint b = 0; b < nb; b++) {
        device const uchar *blk = rp + b * 34u;
        ushort raw = ushort(uint(blk[0]) | (uint(blk[1]) << 8));
        float scale = float(as_type<half>(raw));

        // Each lane reads one of 32 int8 values
        float prod = float(((device const char *)(blk + 2))[lane]) * x[b * 32u + lane];

        // Hardware SIMD reduction across 32 lanes
        float block_sum = simd_sum(prod);

        if (lane == 0) sum += scale * block_sum;
    }

    if (lane == 0) y[actual_row] = sum;
}

// ===================================================================
// F16 batched matvec — Y[B, rows] = W[rows, cols] × X[B, cols]
// Reads W once for all B items. Key to throughput scaling.
// ===================================================================

kernel void f16_batch_matvec(
    device const half *W  [[buffer(0)]],   // [rows, cols]
    device const float *X [[buffer(1)]],   // [B, cols]
    device float *Y       [[buffer(2)]],   // [B, rows]
    constant uint &cols   [[buffer(3)]],
    constant uint &rows   [[buffer(4)]],
    constant uint &B      [[buffer(5)]],
    uint row_group [[threadgroup_position_in_grid]],
    uint sg        [[simdgroup_index_in_threadgroup]],
    uint lane      [[thread_index_in_simdgroup]])
{
    uint row = row_group * 8 + sg;
    if (row >= rows) return;

    device const half *w = W + uint64_t(row) * cols;

    // Per-batch accumulators (register-allocated)
    float sums[32];  // MAX_BATCH=32
    for (uint b = 0; b < B; b++) sums[b] = 0.0f;

    // Read W once, compute dot product for all B items
    uint cols_aligned = cols & ~127u;
    for (uint base = lane * 4; base < cols_aligned; base += 128) {
        half4 wv = *(device const half4*)(w + base);
        float4 wf = float4(wv);
        for (uint b = 0; b < B; b++) {
            float4 xv = *(device const float4*)(X + b * cols + base);
            sums[b] += dot(wf, xv);
        }
    }
    for (uint i = cols_aligned + lane; i < cols; i += 32) {
        float wf = float(w[i]);
        for (uint b = 0; b < B; b++)
            sums[b] += wf * X[b * cols + i];
    }

    // SIMD reduce each batch item
    for (uint b = 0; b < B; b++) {
        float s = simd_sum(sums[b]);
        if (lane == 0) Y[b * rows + row] = s;
    }
}

// ===================================================================
// F16 matvec — half-precision weights, float accumulation
// Bandwidth-optimized: coalesced half4 reads, SIMD reduction
// ===================================================================

kernel void f16_matvec(
    device const half *W [[buffer(0)]],   // [rows, cols] row-major
    device const float *x [[buffer(1)]],  // [cols]
    device float *y       [[buffer(2)]],  // [rows]
    constant uint &cols   [[buffer(3)]],
    constant uint &rows   [[buffer(4)]],
    uint row_group [[threadgroup_position_in_grid]],
    uint sg        [[simdgroup_index_in_threadgroup]],
    uint lane      [[thread_index_in_simdgroup]])
{
    uint row = row_group * 8 + sg;  // 8 simdgroups per threadgroup
    if (row >= rows) return;

    device const half *w = W + uint64_t(row) * cols;
    float sum = 0.0f;

    // 32 lanes × half4 = 128 elements per iteration, perfectly coalesced
    uint cols_aligned = cols & ~127u;
    for (uint base = lane * 4; base < cols_aligned; base += 128) {
        half4 wv = *(device const half4*)(w + base);
        float4 xv = *(device const float4*)(x + base);
        sum += dot(float4(wv), xv);
    }

    // Remainder (rarely needed — Qwen3 dims are all multiples of 128)
    for (uint i = cols_aligned + lane; i < cols; i += 32)
        sum += float(w[i]) * x[i];

    sum = simd_sum(sum);
    if (lane == 0) y[row] = sum;
}

// ===================================================================
// RMSNorm: out = gamma * x / sqrt(mean(x^2) + eps)
// ===================================================================

kernel void rms_norm(
    device const float *x     [[buffer(0)]],
    device const float *gamma [[buffer(1)]],
    device float *out         [[buffer(2)]],
    constant uint &dim        [[buffer(3)]],
    constant float &eps       [[buffer(4)]],
    uint bid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint off = bid * dim;
    threadgroup float shared[256];
    float partial = 0.0f;
    for (uint i = tid; i < dim; i += tg_size)
        partial += x[off + i] * x[off + i];
    shared[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(shared[0] / float(dim) + eps);
    for (uint i = tid; i < dim; i += tg_size)
        out[off + i] = x[off + i] * inv_rms * gamma[i];
}

// ===================================================================
// Per-head RMSNorm: apply RMSNorm to each head independently
// vec: [n_heads * hd], gamma: [hd] (shared across heads)
// One threadgroup per head.
// ===================================================================

kernel void per_head_rms_norm(
    device float *vec           [[buffer(0)]],
    device const float *gamma   [[buffer(1)]],
    constant uint &hd           [[buffer(2)]],
    constant float &eps         [[buffer(3)]],
    uint h   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    device float *head = vec + h * hd;
    threadgroup float shared[256];

    float partial = 0.0f;
    for (uint i = tid; i < hd; i += tg_size)
        partial += head[i] * head[i];
    shared[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(shared[0] / float(hd) + eps);
    for (uint i = tid; i < hd; i += tg_size)
        head[i] = head[i] * inv_rms * gamma[i];
}

// ===================================================================
// RoPE — one thread per (head, dim_pair) for Q and K
// ===================================================================

kernel void rope(
    device float *q         [[buffer(0)]],
    device float *k         [[buffer(1)]],
    constant uint &Hq       [[buffer(2)]],
    constant uint &Hkv      [[buffer(3)]],
    constant uint &hd       [[buffer(4)]],
    constant uint &pos      [[buffer(5)]],
    constant float &theta   [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    uint half_hd = hd / 2;
    uint q_pairs = Hq * half_hd;
    uint total = q_pairs + Hkv * half_hd;
    if (idx >= total) return;

    device float *vec;
    uint d;
    if (idx < q_pairs) {
        uint h = idx / half_hd;
        d = idx % half_hd;
        vec = q + h * hd;
    } else {
        uint ki = idx - q_pairs;
        uint h = ki / half_hd;
        d = ki % half_hd;
        vec = k + h * hd;
    }

    float freq = 1.0f / pow(theta, float(2 * d) / float(hd));
    float angle = float(pos) * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);
    float x0 = vec[d];
    float x1 = vec[d + half_hd];
    vec[d]           = x0 * cos_a - x1 * sin_a;
    vec[d + half_hd] = x0 * sin_a + x1 * cos_a;
}

// ===================================================================
// KV cache store — copy K/V vectors into cache at position pos
// ===================================================================

kernel void kv_cache_store(
    device const float *k    [[buffer(0)]],
    device const float *v    [[buffer(1)]],
    device float *k_cache    [[buffer(2)]],
    device float *v_cache    [[buffer(3)]],
    constant uint &Hkv       [[buffer(4)]],
    constant uint &max_seq   [[buffer(5)]],
    constant uint &hd        [[buffer(6)]],
    constant uint &pos       [[buffer(7)]],
    uint idx [[thread_position_in_grid]])
{
    uint total = Hkv * hd;
    if (idx >= total) return;
    uint h = idx / hd;
    uint d = idx % hd;
    uint ci = (h * max_seq + pos) * hd + d;
    k_cache[ci] = k[idx];
    v_cache[ci] = v[idx];
}

// ===================================================================
// Attention — fused Q@K^T, softmax, scores@V
// One threadgroup per query head (32 heads = 32 threadgroups)
// ===================================================================

kernel void attention(
    device const float *Q     [[buffer(0)]],   // [Hq * hd]
    device const float *Kc    [[buffer(1)]],   // [Hkv * max_seq * hd]
    device const float *Vc    [[buffer(2)]],   // [Hkv * max_seq * hd]
    device float *attn_out    [[buffer(3)]],   // [Hq * hd]
    constant uint &n_attend   [[buffer(4)]],
    constant uint &hd         [[buffer(5)]],
    constant uint &max_seq    [[buffer(6)]],
    constant uint &group_ratio [[buffer(7)]],
    constant float &inv_sqrt_hd [[buffer(8)]],
    uint hq  [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint hkv = hq / group_ratio;
    device const float *qp = Q + hq * hd;
    device const float *kp = Kc + hkv * max_seq * hd;
    device const float *vp = Vc + hkv * max_seq * hd;
    device float *out = attn_out + hq * hd;

    // Step 1: Compute attention scores Q @ K^T
    threadgroup float scores[4096]; // max_seq_len
    for (uint j = tid; j < n_attend; j += tg_size) {
        float dot = 0;
        for (uint d = 0; d < hd; d++)
            dot += qp[d] * kp[j * hd + d];
        scores[j] = dot * inv_sqrt_hd;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Find max for numerical stability
    threadgroup float sh[256];
    float local_max = -1e30f;
    for (uint j = tid; j < n_attend; j += tg_size)
        local_max = max(local_max, scores[j]);
    sh[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] = max(sh[tid], sh[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = sh[0];

    // Step 3: Exp + sum
    float local_sum = 0;
    for (uint j = tid; j < n_attend; j += tg_size) {
        scores[j] = exp(scores[j] - max_val);
        local_sum += scores[j];
    }
    sh[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / sh[0];

    // Step 4: Normalize scores
    for (uint j = tid; j < n_attend; j += tg_size)
        scores[j] *= inv_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 5: Weighted value sum: out[d] = sum_j scores[j] * V[j][d]
    for (uint d = tid; d < hd; d += tg_size) {
        float val = 0;
        for (uint j = 0; j < n_attend; j++)
            val += scores[j] * vp[j * hd + d];
        out[d] = val;
    }
}

// ===================================================================
// SiLU(gate) * up — fused element-wise
// ===================================================================

kernel void silu_mul(
    device float *gate        [[buffer(0)]],
    device const float *up    [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    float g = gate[i];
    gate[i] = (g / (1.0f + exp(-g))) * up[i];
}

// ===================================================================
// Residual add: x += y
// ===================================================================

kernel void residual_add(
    device float *x           [[buffer(0)]],
    device const float *y     [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    x[i] += y[i];
}

// ===================================================================
// Training kernels — F16 tiled matmul, transpose, batch RoPE, etc.
// ===================================================================

// F16 tiled matmul: C[M,N] = A[M,K] @ W[N,K]^T
// A: float, W: half, C: float. Tile size 32.
kernel void f16_matmul(
    device const float *A [[buffer(0)]],
    device const half  *W [[buffer(1)]],
    device float       *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &K [[buffer(4)]],
    constant uint &N [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]])
{
    const uint T = 32;
    threadgroup float tA[T][T+1];
    threadgroup float tW[T][T+1];
    uint row = gid.y * T + tid.y;
    uint col = gid.x * T + tid.x;
    float sum = 0;
    for (uint t = 0; t < K; t += T) {
        tA[tid.y][tid.x] = (row < M && t+tid.x < K) ? A[row * K + t + tid.x] : 0;
        tW[tid.y][tid.x] = (col < N && t+tid.y < K) ? float(W[col * K + t + tid.y]) : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < T; k++) sum += tA[tid.y][k] * tW[k][tid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

// F16 matmul no-transpose: C[M,K] = A[M,N] @ W[N,K]
// For backward: dX = dY @ W (weight not transposed)
kernel void f16_matmul_nt(
    device const float *A [[buffer(0)]],
    device const half  *W [[buffer(1)]],
    device float       *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]])
{
    const uint T = 32;
    threadgroup float tA[T][T+1];
    threadgroup float tW[T][T+1];
    uint row = gid.y * T + tid.y;
    uint col = gid.x * T + tid.x;
    float sum = 0;
    for (uint t = 0; t < N; t += T) {
        tA[tid.y][tid.x] = (row < M && t+tid.x < N) ? A[row * N + t + tid.x] : 0;
        tW[tid.y][tid.x] = (t+tid.y < N && col < K) ? float(W[(t+tid.y) * K + col]) : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < T; k++) sum += tA[tid.y][k] * tW[k][tid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < K) C[row * K + col] = sum;
}

// Transpose: [N, H*D] -> [H, N, D]
kernel void transpose_heads_fwd(
    device const float *src [[buffer(0)]],
    device float *dst       [[buffer(1)]],
    constant uint &seq_len  [[buffer(2)]],
    constant uint &n_heads  [[buffer(3)]],
    constant uint &head_dim [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    uint HD = n_heads * head_dim;
    uint total = seq_len * HD;
    if (idx >= total) return;
    uint n = idx / HD;
    uint rem = idx % HD;
    uint h = rem / head_dim;
    uint d = rem % head_dim;
    dst[h * seq_len * head_dim + n * head_dim + d] = src[idx];
}

// Transpose back: [H, N, D] -> [N, H*D]
kernel void transpose_heads_rev(
    device const float *src [[buffer(0)]],
    device float *dst       [[buffer(1)]],
    constant uint &seq_len  [[buffer(2)]],
    constant uint &n_heads  [[buffer(3)]],
    constant uint &head_dim [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    uint ND = seq_len * head_dim;
    uint total = n_heads * ND;
    if (idx >= total) return;
    uint h = idx / ND;
    uint rem = idx % ND;
    uint n = rem / head_dim;
    uint d = rem % head_dim;
    dst[n * n_heads * head_dim + h * head_dim + d] = src[idx];
}

// Repeat KV heads: [Hkv, N, D] -> [Hq, N, D]
kernel void repeat_kv(
    device const float *src [[buffer(0)]],
    device float *dst       [[buffer(1)]],
    constant uint &n_kv     [[buffer(2)]],
    constant uint &seq_len  [[buffer(3)]],
    constant uint &head_dim [[buffer(4)]],
    constant uint &group_ratio [[buffer(5)]],
    uint idx [[thread_position_in_grid]])
{
    uint ND = seq_len * head_dim;
    uint total = n_kv * group_ratio * ND;
    if (idx >= total) return;
    uint hq = idx / ND;
    uint rem = idx % ND;
    uint hkv = hq / group_ratio;
    dst[idx] = src[hkv * ND + rem];
}

// Batch RoPE for training: apply to [H, N, D] in-place
kernel void rope_train(
    device float *q         [[buffer(0)]],
    device float *k         [[buffer(1)]],
    constant uint &Hq       [[buffer(2)]],
    constant uint &Hkv      [[buffer(3)]],
    constant uint &hd       [[buffer(4)]],
    constant uint &seq_len  [[buffer(5)]],
    constant float &theta   [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    uint half_hd = hd / 2;
    uint q_total = Hq * seq_len * half_hd;
    uint total = q_total + Hkv * seq_len * half_hd;
    if (idx >= total) return;

    device float *vec;
    uint pos, d;
    if (idx < q_total) {
        uint hi = idx / (seq_len * half_hd);
        uint r = idx % (seq_len * half_hd);
        pos = r / half_hd;
        d = r % half_hd;
        vec = q + hi * seq_len * hd + pos * hd;
    } else {
        uint ki = idx - q_total;
        uint hi = ki / (seq_len * half_hd);
        uint r = ki % (seq_len * half_hd);
        pos = r / half_hd;
        d = r % half_hd;
        vec = k + hi * seq_len * hd + pos * hd;
    }

    float freq = 1.0f / pow(theta, float(2 * d) / float(hd));
    float angle = float(pos) * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);
    float x0 = vec[d];
    float x1 = vec[d + half_hd];
    vec[d]           = x0 * cos_a - x1 * sin_a;
    vec[d + half_hd] = x0 * sin_a + x1 * cos_a;
}

// SiLU*mul backward: given d_out, gate_pre_silu, up
// d_gate = d_out * up * silu'(gate)
// d_up   = d_out * silu(gate)
kernel void silu_mul_backward(
    device const float *dout  [[buffer(0)]],
    device const float *gate  [[buffer(1)]],
    device const float *up    [[buffer(2)]],
    device float *dgate       [[buffer(3)]],
    device float *dup         [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    float g = gate[i];
    float u = up[i];
    float dy = dout[i];
    float sig = 1.0f / (1.0f + exp(-g));
    float silu_g = g * sig;
    float silu_grad = sig * (1.0f + g * (1.0f - sig));
    dgate[i] = dy * u * silu_grad;
    dup[i] = dy * silu_g;
}

// RMSNorm backward for training (batched)
// dx_i = gamma_i * dy_i / rms - x_i / (dim * rms^3) * sum_j(dy_j * gamma_j * x_j)
kernel void rms_norm_train_backward(
    device const float *x     [[buffer(0)]],
    device const float *dy    [[buffer(1)]],
    device const float *gamma [[buffer(2)]],
    device float *dx          [[buffer(3)]],
    constant uint &dim        [[buffer(4)]],
    constant float &eps       [[buffer(5)]],
    uint bid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint off = bid * dim;
    threadgroup float shared[256];

    // sum(x^2) for RMS
    float partial = 0;
    for (uint i = tid; i < dim; i += tg_size)
        partial += x[off+i] * x[off+i];
    shared[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size/2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid+s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(shared[0] / float(dim) + eps);

    // sum(dy * gamma * x) for gradient correction
    partial = 0;
    for (uint i = tid; i < dim; i += tg_size)
        partial += dy[off+i] * gamma[i] * x[off+i];
    shared[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size/2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid+s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float coeff = shared[0] * inv_rms * inv_rms / float(dim);

    for (uint i = tid; i < dim; i += tg_size)
        dx[off+i] = (gamma[i] * dy[off+i] - coeff * x[off+i]) * inv_rms;
}

// Scale tensor: x *= alpha
kernel void scale_tensor(
    device float *x      [[buffer(0)]],
    constant float &alpha [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    x[i] *= alpha;
}

// Add scaled: y += alpha * x
kernel void add_scaled(
    device float *y       [[buffer(0)]],
    device const float *x [[buffer(1)]],
    constant float &alpha [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    y[i] += alpha * x[i];
}

// Softmax cross-entropy loss + gradient (for training)
kernel void softmax_ce_train(
    device const float *logits   [[buffer(0)]],
    device const uint  *targets  [[buffer(1)]],
    device float       *losses   [[buffer(2)]],
    device float       *dlogits  [[buffer(3)]],
    constant uint &vocab_size    [[buffer(4)]],
    uint bid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint off = bid * vocab_size;
    uint tgt = targets[bid];
    threadgroup float shared[256];

    // Find max
    float local_max = -1e30f;
    for (uint i = tid; i < vocab_size; i += tg_size)
        local_max = max(local_max, logits[off + i]);
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size/2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = max(shared[tid], shared[tid+s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared[0];

    // Sum exp
    float local_sum = 0;
    for (uint i = tid; i < vocab_size; i += tg_size)
        local_sum += exp(logits[off + i] - max_val);
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size/2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid+s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float log_sum = log(shared[0]) + max_val;

    // Loss and gradient
    if (tid == 0) losses[bid] = log_sum - logits[off + tgt];
    for (uint i = tid; i < vocab_size; i += tg_size) {
        float prob = exp(logits[off + i] - log_sum);
        dlogits[off + i] = prob - (i == tgt ? 1.0f : 0.0f);
    }
}

// ===================================================================
// Attention training forward: causal self-attention with probs saving
// Q, K, V: [H, N, D] (K,V already expanded via repeat_kv)
// out: [H, N, D], probs: [H, N, N]
// Dispatch: [H, N, 1] threadgroups, each handles one (head, query_pos)
// ===================================================================
kernel void attn_train_fwd(
    device const float *Q      [[buffer(0)]],
    device const float *K      [[buffer(1)]],
    device const float *V      [[buffer(2)]],
    device float       *out    [[buffer(3)]],
    device float       *probs  [[buffer(4)]],
    constant uint &N           [[buffer(5)]],
    constant uint &D           [[buffer(6)]],
    constant float &scale      [[buffer(7)]],
    uint2 bid2 [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 tg_size2 [[threads_per_threadgroup]])
{
    uint h = bid2.x;
    uint i = bid2.y;
    uint tid = tid2.x;
    uint tg_size = tg_size2.x;
    uint ND = N * D;
    uint NN = N * N;

    const device float *q_row = Q + h * ND + i * D;
    const device float *K_h = K + h * ND;
    const device float *V_h = V + h * ND;
    device float *out_row = out + h * ND + i * D;
    device float *prob_row = probs + h * NN + i * N;

    // Compute attention scores: score[j] = Q[h,i,:] . K[h,j,:] * scale
    threadgroup float scores[4096];
    for (uint j = tid; j <= i; j += tg_size) {
        float dot = 0;
        for (uint d = 0; d < D; d++)
            dot += q_row[d] * K_h[j * D + d];
        scores[j] = dot * scale;
    }
    for (uint j = i + 1 + tid; j < N; j += tg_size)
        scores[j] = -1e30f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Softmax: find max
    threadgroup float shared[256];
    float local_max = -1e30f;
    for (uint j = tid; j < N; j += tg_size)
        local_max = max(local_max, scores[j]);
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size/2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = max(shared[tid], shared[tid+s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared[0];

    // Exp and sum
    float local_sum = 0;
    for (uint j = tid; j <= i; j += tg_size) {
        scores[j] = exp(scores[j] - max_val);
        local_sum += scores[j];
    }
    for (uint j = i + 1 + tid; j < N; j += tg_size)
        scores[j] = 0;
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size/2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid+s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / shared[0];

    // Normalize and save probs
    for (uint j = tid; j < N; j += tg_size) {
        float p = scores[j] * inv_sum;
        scores[j] = p;
        prob_row[j] = p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Weighted sum: out[h,i,d] = sum_j probs[j] * V[h,j,d]
    for (uint d = tid; d < D; d += tg_size) {
        float val = 0;
        for (uint j = 0; j <= i; j++)
            val += scores[j] * V_h[j * D + d];
        out_row[d] = val;
    }
}

// ===================================================================
// Attention backward: compute d_score and dQ
// Dispatch: [H, N, 1]
// ===================================================================
kernel void attn_bwd_dq(
    device const float *d_out   [[buffer(0)]],
    device const float *probs   [[buffer(1)]],
    device const float *V       [[buffer(2)]],
    device const float *K       [[buffer(3)]],
    device float       *d_score [[buffer(4)]],
    device float       *dQ      [[buffer(5)]],
    constant uint &N            [[buffer(6)]],
    constant uint &D            [[buffer(7)]],
    constant float &scale       [[buffer(8)]],
    uint2 bid2 [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 tg_size2 [[threads_per_threadgroup]])
{
    uint h = bid2.x;
    uint i = bid2.y;
    uint tid = tid2.x;
    uint tg_size = tg_size2.x;
    uint ND = N * D;
    uint NN = N * N;

    const device float *dout_row = d_out + h * ND + i * D;
    const device float *prob_row = probs + h * NN + i * N;
    const device float *V_h = V + h * ND;
    const device float *K_h = K + h * ND;
    device float *ds_row = d_score + h * NN + i * N;
    device float *dQ_row = dQ + h * ND + i * D;

    // Compute d_raw[j] = dout[i] . V[j]
    threadgroup float d_raw[4096];
    for (uint j = tid; j <= i; j += tg_size) {
        float dot = 0;
        for (uint d = 0; d < D; d++)
            dot += dout_row[d] * V_h[j * D + d];
        d_raw[j] = dot;
    }
    for (uint j = i + 1 + tid; j < N; j += tg_size)
        d_raw[j] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // sum_k probs[k] * d_raw[k]
    threadgroup float shared[256];
    float local_sum = 0;
    for (uint j = tid; j <= i; j += tg_size)
        local_sum += prob_row[j] * d_raw[j];
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size/2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid+s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float dp_sum = shared[0];

    // d_score[j] = probs[j] * (d_raw[j] - dp_sum) * scale
    for (uint j = tid; j < N; j += tg_size) {
        float ds = (j <= i) ? prob_row[j] * (d_raw[j] - dp_sum) * scale : 0;
        d_raw[j] = ds;
        ds_row[j] = ds;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // dQ[h,i,d] = sum_j d_score[j] * K[h,j,d]
    for (uint d = tid; d < D; d += tg_size) {
        float val = 0;
        for (uint j = 0; j <= i; j++)
            val += d_raw[j] * K_h[j * D + d];
        dQ_row[d] = val;
    }
}

// ===================================================================
// Attention backward: compute dK and dV
// Dispatch: [H, N, 1] — one threadgroup per (head, key_position)
// ===================================================================
kernel void attn_bwd_dkv(
    device const float *d_score [[buffer(0)]],
    device const float *Q       [[buffer(1)]],
    device const float *probs   [[buffer(2)]],
    device const float *d_out   [[buffer(3)]],
    device float       *dK      [[buffer(4)]],
    device float       *dV      [[buffer(5)]],
    constant uint &N            [[buffer(6)]],
    constant uint &D            [[buffer(7)]],
    uint2 bid2 [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 tg_size2 [[threads_per_threadgroup]])
{
    uint h = bid2.x;
    uint j = bid2.y;
    uint tid = tid2.x;
    uint tg_size = tg_size2.x;
    uint ND = N * D;
    uint NN = N * N;

    const device float *Q_h = Q + h * ND;
    const device float *dout_h = d_out + h * ND;
    device float *dK_row = dK + h * ND + j * D;
    device float *dV_row = dV + h * ND + j * D;

    for (uint d = tid; d < D; d += tg_size) {
        float dk_val = 0;
        float dv_val = 0;
        for (uint i = j; i < N; i++) {
            float ds = d_score[h * NN + i * N + j];
            float pr = probs[h * NN + i * N + j];
            dk_val += ds * Q_h[i * D + d];
            dv_val += pr * dout_h[i * D + d];
        }
        dK_row[d] = dk_val;
        dV_row[d] = dv_val;
    }
}

// ===================================================================
// Repeat KV backward: reduce dK_expanded[Hq, N, D] -> dK[Hkv, N, D]
// ===================================================================
kernel void repeat_kv_bwd(
    device const float *d_expanded [[buffer(0)]],
    device float       *d_kv       [[buffer(1)]],
    constant uint &n_kv            [[buffer(2)]],
    constant uint &seq_len         [[buffer(3)]],
    constant uint &head_dim        [[buffer(4)]],
    constant uint &group_ratio     [[buffer(5)]],
    uint idx [[thread_position_in_grid]])
{
    uint ND = seq_len * head_dim;
    uint total = n_kv * ND;
    if (idx >= total) return;

    uint hkv = idx / ND;
    uint rem = idx % ND;
    float sum = 0;
    for (uint g = 0; g < group_ratio; g++)
        sum += d_expanded[(hkv * group_ratio + g) * ND + rem];
    d_kv[idx] = sum;
}

// ===================================================================
// RoPE backward: inverse rotation (transpose of rotation matrix)
// ===================================================================
kernel void rope_train_bwd(
    device float *q         [[buffer(0)]],
    device float *k         [[buffer(1)]],
    constant uint &Hq       [[buffer(2)]],
    constant uint &Hkv      [[buffer(3)]],
    constant uint &hd       [[buffer(4)]],
    constant uint &seq_len  [[buffer(5)]],
    constant float &theta   [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    uint half_hd = hd / 2;
    uint q_total = Hq * seq_len * half_hd;
    uint total = q_total + Hkv * seq_len * half_hd;
    if (idx >= total) return;

    device float *vec;
    uint pos, d;
    if (idx < q_total) {
        uint hi = idx / (seq_len * half_hd);
        uint r = idx % (seq_len * half_hd);
        pos = r / half_hd;
        d = r % half_hd;
        vec = q + hi * seq_len * hd + pos * hd;
    } else {
        uint ki = idx - q_total;
        uint hi = ki / (seq_len * half_hd);
        uint r = ki % (seq_len * half_hd);
        pos = r / half_hd;
        d = r % half_hd;
        vec = k + hi * seq_len * hd + pos * hd;
    }

    float freq = 1.0f / pow(theta, float(2 * d) / float(hd));
    float angle = float(pos) * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);
    float x0 = vec[d];
    float x1 = vec[d + half_hd];
    // Inverse rotation: [cos, sin; -sin, cos]
    vec[d]           = x0 * cos_a + x1 * sin_a;
    vec[d + half_hd] = -x0 * sin_a + x1 * cos_a;
}

// ===================================================================
// Float-float matmul: C[M,N] = A[M,K] @ W[N,K]^T (both float, for LoRA)
// ===================================================================
kernel void float_matmul(
    device const float *A [[buffer(0)]],
    device const float *W [[buffer(1)]],
    device float       *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &K [[buffer(4)]],
    constant uint &N [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]])
{
    const uint T = 16;
    threadgroup float tA[T][T+1];
    threadgroup float tW[T][T+1];
    uint row = gid.y * T + tid.y;
    uint col = gid.x * T + tid.x;
    float sum = 0;
    for (uint t = 0; t < K; t += T) {
        tA[tid.y][tid.x] = (row < M && t+tid.x < K) ? A[row * K + t + tid.x] : 0;
        tW[tid.y][tid.x] = (col < N && t+tid.y < K) ? W[col * K + t + tid.y] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < T; k++) sum += tA[tid.y][k] * tW[k][tid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

// Float matmul transposed-A: C[M,N] = A[K,M]^T @ B[K,N]
kernel void float_matmul_tn(
    device const float *A [[buffer(0)]],
    device const float *B [[buffer(1)]],
    device float       *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &K [[buffer(4)]],
    constant uint &N [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]])
{
    const uint T = 16;
    threadgroup float tA[T][T+1];
    threadgroup float tB[T][T+1];
    uint row = gid.y * T + tid.y;
    uint col = gid.x * T + tid.x;
    float sum = 0;
    for (uint t = 0; t < K; t += T) {
        tA[tid.y][tid.x] = (row < M && t+tid.x < K) ? A[(t+tid.x) * M + row] : 0;
        tB[tid.y][tid.x] = (col < N && t+tid.y < K) ? B[(t+tid.y) * N + col] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < T; k++) sum += tA[tid.y][k] * tB[k][tid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

// Float matmul no-transpose: C[M,K] = A[M,N] @ W[N,K]
kernel void float_matmul_nt(
    device const float *A [[buffer(0)]],
    device const float *W [[buffer(1)]],
    device float       *C [[buffer(2)]],
    constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]])
{
    const uint T = 16;
    threadgroup float tA[T][T+1];
    threadgroup float tW[T][T+1];
    uint row = gid.y * T + tid.y;
    uint col = gid.x * T + tid.x;
    float sum = 0;
    for (uint t = 0; t < N; t += T) {
        tA[tid.y][tid.x] = (row < M && t+tid.x < N) ? A[row * N + t + tid.x] : 0;
        tW[tid.y][tid.x] = (t+tid.y < N && col < K) ? W[(t+tid.y) * K + col] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < T; k++) sum += tA[tid.y][k] * tW[k][tid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < K) C[row * K + col] = sum;
}

// Element-wise clamp: x = clamp(x, -max_val, max_val)
kernel void clamp_tensor(
    device float *x        [[buffer(0)]],
    constant float &max_val [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    x[i] = clamp(x[i], -max_val, max_val);
}

// Buffer copy: dst = src
kernel void buf_copy(
    device const float *src [[buffer(0)]],
    device float       *dst [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    dst[i] = src[i];
}
