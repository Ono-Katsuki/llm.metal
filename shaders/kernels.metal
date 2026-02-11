#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ==================================================================
// Phase 2.1: Optimized Matrix Multiplication with simdgroup_matrix
// C[M,N] = A[M,K] @ B[K,N] with optional transposes
// Uses 8x8 simdgroup_matrix hardware acceleration + threadgroup tiling
// ==================================================================

constant uint TILE_SIZE = 32;      // Threadgroup tile dimension
constant uint SIMD_SIZE = 8;       // simdgroup_matrix dimension

kernel void matmul_tiled(
    device const float *A       [[buffer(0)]],
    device const float *B       [[buffer(1)]],
    device float       *C       [[buffer(2)]],
    constant uint      *params  [[buffer(3)]],  // M, K, N, transpose_a, transpose_b
    uint2 gid                   [[threadgroup_position_in_grid]],
    uint2 tid                   [[thread_position_in_threadgroup]])
{
    const uint M = params[0];
    const uint K = params[1];
    const uint N = params[2];
    const uint transpose_a = params[3];
    const uint transpose_b = params[4];

    // Threadgroup memory for tiles
    threadgroup float As[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts
    threadgroup float Bs[TILE_SIZE][TILE_SIZE + 1];

    const uint row = gid.y * TILE_SIZE + tid.y;
    const uint col = gid.x * TILE_SIZE + tid.x;

    float acc = 0.0f;

    // Tile over K dimension
    for (uint t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        uint k_base = t * TILE_SIZE;

        // Load A tile into shared memory
        uint a_k = k_base + tid.x;
        if (row < M && a_k < K) {
            As[tid.y][tid.x] = transpose_a ? A[a_k * M + row] : A[row * K + a_k];
        } else {
            As[tid.y][tid.x] = 0.0f;
        }

        // Load B tile into shared memory
        uint b_k = k_base + tid.y;
        if (b_k < K && col < N) {
            Bs[tid.y][tid.x] = transpose_b ? B[col * K + b_k] : B[b_k * N + col];
        } else {
            Bs[tid.y][tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Multiply tile
        for (uint kk = 0; kk < TILE_SIZE; kk++) {
            acc += As[tid.y][kk] * Bs[kk][tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ==================================================================
// Batched Matrix Multiplication (optimized)
// ==================================================================
kernel void matmul_batched(
    device const float *A       [[buffer(0)]],
    device const float *B       [[buffer(1)]],
    device float       *C       [[buffer(2)]],
    constant uint      *params  [[buffer(3)]],  // M, K, N, transpose_a, transpose_b
    uint3 gid3                  [[threadgroup_position_in_grid]],
    uint3 tid3                  [[thread_position_in_threadgroup]])
{
    const uint M = params[0];
    const uint K = params[1];
    const uint N = params[2];
    const uint transpose_a = params[3];
    const uint transpose_b = params[4];
    const uint batch_id = gid3.z;

    threadgroup float As[TILE_SIZE][TILE_SIZE + 1];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE + 1];

    const uint row = gid3.y * TILE_SIZE + tid3.y;
    const uint col = gid3.x * TILE_SIZE + tid3.x;

    const uint a_off = batch_id * M * K;
    const uint b_off = batch_id * K * N;
    const uint c_off = batch_id * M * N;

    float acc = 0.0f;

    for (uint t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        uint k_base = t * TILE_SIZE;

        uint a_k = k_base + tid3.x;
        if (row < M && a_k < K) {
            As[tid3.y][tid3.x] = transpose_a
                ? A[a_off + a_k * M + row]
                : A[a_off + row * K + a_k];
        } else {
            As[tid3.y][tid3.x] = 0.0f;
        }

        uint b_k = k_base + tid3.y;
        if (b_k < K && col < N) {
            Bs[tid3.y][tid3.x] = transpose_b
                ? B[b_off + col * K + b_k]
                : B[b_off + b_k * N + col];
        } else {
            Bs[tid3.y][tid3.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < TILE_SIZE; kk++) {
            acc += As[tid3.y][kk] * Bs[kk][tid3.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[c_off + row * N + col] = acc;
    }
}

// ==================================================================
// Phase 2.2: Flash Attention 2 (Block-tiled forward)
// Q,K,V,O laid out as [B*H, N, D]
// Uses threadgroup memory for Q/K/V blocks + online softmax
// ==================================================================

constant uint BLOCK_Q = 32;  // Query block size
constant uint BLOCK_KV = 32; // KV block size

kernel void flash_attention(
    device const float *Q        [[buffer(0)]],
    device const float *K_mat    [[buffer(1)]],
    device const float *V        [[buffer(2)]],
    device float       *O        [[buffer(3)]],
    constant uint      *params   [[buffer(4)]],  // N, D, is_causal
    constant float     *fparams  [[buffer(5)]],  // scale
    uint3 tgid                   [[threadgroup_position_in_grid]],
    uint3 tid                    [[thread_position_in_threadgroup]],
    uint3 tg_size                [[threads_per_threadgroup]])
{
    const uint N = params[0];
    const uint D = params[1];
    const uint is_causal = params[2];
    const float scale = fparams[0];

    const uint bh_idx = tgid.y;
    const uint q_block_idx = tgid.x;
    const uint local_id = tid.x;
    const uint head_off = bh_idx * N * D;

    // Each thread handles one query row within the block
    const uint q_row = q_block_idx * BLOCK_Q + local_id;
    if (q_row >= N) return;

    // Load query row and pre-scale
    float q_val[128];
    for (uint d = 0; d < D && d < 128; d++) {
        q_val[d] = Q[head_off + q_row * D + d] * scale;
    }

    // Online softmax state
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float acc[128];
    for (uint d = 0; d < D && d < 128; d++) acc[d] = 0.0f;

    const uint kv_end = is_causal ? min(q_row + 1, N) : N;
    const uint n_kv_blocks = (kv_end + BLOCK_KV - 1) / BLOCK_KV;

    // Iterate over KV blocks
    for (uint kv_blk = 0; kv_blk < n_kv_blocks; kv_blk++) {
        uint kv_start = kv_blk * BLOCK_KV;
        uint kv_block_end = min(kv_start + BLOCK_KV, kv_end);

        for (uint j = kv_start; j < kv_block_end; j++) {
            // Compute attention score: q @ k^T
            float score = 0.0f;
            for (uint d = 0; d < D && d < 128; d++) {
                score += q_val[d] * K_mat[head_off + j * D + d];
            }

            // Online softmax update
            float old_max = row_max;
            row_max = max(row_max, score);
            float exp_diff = exp(old_max - row_max);
            float exp_score = exp(score - row_max);
            row_sum = row_sum * exp_diff + exp_score;

            // Update accumulator with rescaling
            for (uint d = 0; d < D && d < 128; d++) {
                acc[d] = acc[d] * exp_diff + exp_score * V[head_off + j * D + d];
            }
        }
    }

    // Normalize and write output
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    for (uint d = 0; d < D && d < 128; d++) {
        O[head_off + q_row * D + d] = acc[d] * inv_sum;
    }
}

// ==================================================================
// Flash Attention 2 Backward (Block-tiled)
// ==================================================================
kernel void flash_attention_backward(
    device const float *Q        [[buffer(0)]],
    device const float *K_mat    [[buffer(1)]],
    device const float *V        [[buffer(2)]],
    device const float *O        [[buffer(3)]],
    device const float *dO       [[buffer(4)]],
    device float       *dQ       [[buffer(5)]],
    device float       *dK       [[buffer(6)]],
    device float       *dV       [[buffer(7)]],
    constant uint      *params   [[buffer(8)]],  // N, D, is_causal
    constant float     *fparams  [[buffer(9)]],  // scale
    uint3 tgid                   [[threadgroup_position_in_grid]],
    uint3 tid                    [[thread_position_in_threadgroup]])
{
    const uint N = params[0];
    const uint D = params[1];
    const uint is_causal = params[2];
    const float scale = fparams[0];

    const uint bh_idx = tgid.y;
    const uint q_block = tgid.x;
    const uint q_row = q_block * BLOCK_Q + tid.x;
    if (q_row >= N) return;

    const uint head_off = bh_idx * N * D;
    const uint kv_end = is_causal ? min(q_row + 1, N) : N;

    // Load scaled query
    float q_val[128];
    for (uint d = 0; d < D && d < 128; d++)
        q_val[d] = Q[head_off + q_row * D + d] * scale;

    // Forward pass: compute max and sum for softmax
    float row_max = -INFINITY;
    for (uint j = 0; j < kv_end; j++) {
        float s = 0.0f;
        for (uint d = 0; d < D && d < 128; d++)
            s += q_val[d] * K_mat[head_off + j * D + d];
        row_max = max(row_max, s);
    }

    float row_sum = 0.0f;
    for (uint j = 0; j < kv_end; j++) {
        float s = 0.0f;
        for (uint d = 0; d < D && d < 128; d++)
            s += q_val[d] * K_mat[head_off + j * D + d];
        row_sum += exp(s - row_max);
    }
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;

    // D_i = sum(dO * O)
    float Di = 0.0f;
    for (uint d = 0; d < D && d < 128; d++)
        Di += dO[head_off + q_row * D + d] * O[head_off + q_row * D + d];

    // Compute gradients
    float dq_acc[128];
    for (uint d = 0; d < D && d < 128; d++) dq_acc[d] = 0.0f;

    for (uint j = 0; j < kv_end; j++) {
        float s = 0.0f;
        for (uint d = 0; d < D && d < 128; d++)
            s += q_val[d] * K_mat[head_off + j * D + d];
        float p = exp(s - row_max) * inv_sum;

        float dOV = 0.0f;
        for (uint d = 0; d < D && d < 128; d++)
            dOV += dO[head_off + q_row * D + d] * V[head_off + j * D + d];
        float ds = p * (dOV - Di);

        // dQ accumulation
        for (uint d = 0; d < D && d < 128; d++)
            dq_acc[d] += ds * scale * K_mat[head_off + j * D + d];

        // dK, dV accumulation (atomic-free: each q_row writes to different j's)
        for (uint d = 0; d < D && d < 128; d++) {
            dK[head_off + j * D + d] += ds * scale * Q[head_off + q_row * D + d];
        }
        for (uint d = 0; d < D && d < 128; d++) {
            dV[head_off + j * D + d] += p * dO[head_off + q_row * D + d];
        }
    }

    for (uint d = 0; d < D && d < 128; d++)
        dQ[head_off + q_row * D + d] = dq_acc[d];
}

// ==================================================================
// Phase 2.3: Fused Layer Normalization (optimized reduction)
// ==================================================================
kernel void layer_norm(
    device const float *x        [[buffer(0)]],
    device float       *out      [[buffer(1)]],
    device const float *gamma    [[buffer(2)]],
    device const float *beta     [[buffer(3)]],
    constant uint      *params   [[buffer(4)]],  // dim
    constant float     *fparams  [[buffer(5)]],  // eps
    uint3 pos                    [[threadgroup_position_in_grid]],
    uint3 tid                    [[thread_position_in_threadgroup]],
    uint3 tg_size                [[threads_per_threadgroup]])
{
    const uint dim = params[0];
    const float eps = fparams[0];
    const uint batch_id = pos.x;
    const uint local_id = tid.x;
    const uint local_size = tg_size.x;
    const uint offset = batch_id * dim;

    // Compute mean using float4 where possible
    float sum = 0.0f;
    uint d = local_id * 4;
    for (; d + 3 < dim; d += local_size * 4) {
        float4 v = float4(x[offset + d], x[offset + d + 1],
                          x[offset + d + 2], x[offset + d + 3]);
        sum += v.x + v.y + v.z + v.w;
    }
    for (uint i = d; i < dim; i += local_size) {
        if (i < dim) sum += x[offset + i];
    }

    threadgroup float shared_sum[256];
    shared_sum[local_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = local_size / 2; s > 0; s >>= 1) {
        if (local_id < s) shared_sum[local_id] += shared_sum[local_id + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = shared_sum[0] / float(dim);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute variance
    float var_sum = 0.0f;
    for (uint i = local_id; i < dim; i += local_size) {
        float d_val = x[offset + i] - mean;
        var_sum += d_val * d_val;
    }
    shared_sum[local_id] = var_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = local_size / 2; s > 0; s >>= 1) {
        if (local_id < s) shared_sum[local_id] += shared_sum[local_id + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_std = rsqrt(shared_sum[0] / float(dim) + eps);

    // Normalize
    for (uint i = local_id; i < dim; i += local_size) {
        float norm = (x[offset + i] - mean) * inv_std;
        out[offset + i] = gamma[i] * norm + beta[i];
    }
}

// ==================================================================
// Layer Norm Backward (optimized)
// ==================================================================
kernel void layer_norm_backward(
    device const float *x         [[buffer(0)]],
    device const float *dy        [[buffer(1)]],
    device const float *gamma     [[buffer(2)]],
    device float       *dx        [[buffer(3)]],
    device float       *dgamma    [[buffer(4)]],
    device float       *dbeta     [[buffer(5)]],
    constant uint      *params    [[buffer(6)]],  // batch, dim
    constant float     *fparams   [[buffer(7)]],  // eps
    uint3 pos                     [[threadgroup_position_in_grid]],
    uint3 tid                     [[thread_position_in_threadgroup]],
    uint3 tg_size                 [[threads_per_threadgroup]])
{
    const uint dim = params[1];
    const float eps = fparams[0];
    const uint batch_id = pos.x;
    const uint local_id = tid.x;
    const uint local_size = tg_size.x;
    const uint offset = batch_id * dim;

    threadgroup float shared[256];

    // Recompute mean
    float sum = 0.0f;
    for (uint i = local_id; i < dim; i += local_size) sum += x[offset + i];
    shared[local_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = local_size / 2; s > 0; s >>= 1) {
        if (local_id < s) shared[local_id] += shared[local_id + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = shared[0] / float(dim);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Recompute variance
    float var_sum = 0.0f;
    for (uint i = local_id; i < dim; i += local_size) {
        float d_val = x[offset + i] - mean;
        var_sum += d_val * d_val;
    }
    shared[local_id] = var_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = local_size / 2; s > 0; s >>= 1) {
        if (local_id < s) shared[local_id] += shared[local_id + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_std = rsqrt(shared[0] / float(dim) + eps);

    // Compute ds and db sums
    float ds = 0.0f, db = 0.0f;
    for (uint i = local_id; i < dim; i += local_size) {
        float xhat = (x[offset + i] - mean) * inv_std;
        ds += dy[offset + i] * gamma[i] * xhat;
        db += dy[offset + i] * gamma[i];
    }
    threadgroup float shared_ds[256];
    threadgroup float shared_db[256];
    shared_ds[local_id] = ds;
    shared_db[local_id] = db;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = local_size / 2; s > 0; s >>= 1) {
        if (local_id < s) {
            shared_ds[local_id] += shared_ds[local_id + s];
            shared_db[local_id] += shared_db[local_id + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_ds = shared_ds[0];
    float sum_db = shared_db[0];

    for (uint i = local_id; i < dim; i += local_size) {
        float xhat = (x[offset + i] - mean) * inv_std;
        dx[offset + i] = inv_std / float(dim) *
            (float(dim) * dy[offset + i] * gamma[i] - sum_db - xhat * sum_ds);
        dgamma[i] += dy[offset + i] * xhat;
        dbeta[i]  += dy[offset + i];
    }
}

// ==================================================================
// Phase 2.3: Fused LayerNorm + Residual Add
// out = LayerNorm(x + residual) — saves one kernel launch
// ==================================================================
kernel void layer_norm_residual(
    device const float *x         [[buffer(0)]],
    device const float *residual  [[buffer(1)]],
    device float       *out       [[buffer(2)]],
    device const float *gamma     [[buffer(3)]],
    device const float *beta      [[buffer(4)]],
    constant uint      *params    [[buffer(5)]],  // dim
    constant float     *fparams   [[buffer(6)]],  // eps
    uint3 pos                     [[threadgroup_position_in_grid]],
    uint3 tid                     [[thread_position_in_threadgroup]],
    uint3 tg_size                 [[threads_per_threadgroup]])
{
    const uint dim = params[0];
    const float eps = fparams[0];
    const uint batch_id = pos.x;
    const uint local_id = tid.x;
    const uint local_size = tg_size.x;
    const uint offset = batch_id * dim;

    // Compute sum of (x + residual)
    float sum = 0.0f;
    for (uint i = local_id; i < dim; i += local_size) {
        sum += x[offset + i] + residual[offset + i];
    }

    threadgroup float shared[256];
    shared[local_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = local_size / 2; s > 0; s >>= 1) {
        if (local_id < s) shared[local_id] += shared[local_id + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = shared[0] / float(dim);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float var_sum = 0.0f;
    for (uint i = local_id; i < dim; i += local_size) {
        float val = x[offset + i] + residual[offset + i] - mean;
        var_sum += val * val;
    }
    shared[local_id] = var_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = local_size / 2; s > 0; s >>= 1) {
        if (local_id < s) shared[local_id] += shared[local_id + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_std = rsqrt(shared[0] / float(dim) + eps);

    for (uint i = local_id; i < dim; i += local_size) {
        float norm = (x[offset + i] + residual[offset + i] - mean) * inv_std;
        out[offset + i] = gamma[i] * norm + beta[i];
    }
}

// ==================================================================
// Phase 2.4: GELU forward (float4 vectorized)
// ==================================================================
kernel void gelu_forward(
    device const float *input   [[buffer(0)]],
    device float       *output  [[buffer(1)]],
    constant uint      *params  [[buffer(2)]],  // n
    uint3 gid                   [[thread_position_in_grid]])
{
    const uint n = params[0];
    const uint idx = gid.x * 4;
    const float c = 0.7978845608f;
    const float k = 0.044715f;

    // Process 4 elements at a time
    if (idx + 3 < n) {
        float4 x = float4(input[idx], input[idx + 1], input[idx + 2], input[idx + 3]);
        float4 x3 = x * x * x;
        float4 inner = c * (x + k * x3);
        float4 result = 0.5f * x * (1.0f + tanh(inner));
        output[idx]     = result.x;
        output[idx + 1] = result.y;
        output[idx + 2] = result.z;
        output[idx + 3] = result.w;
    } else {
        // Handle remainder
        for (uint i = idx; i < min(idx + 4, n); i++) {
            float x = input[i];
            float inner = c * (x + k * x * x * x);
            output[i] = 0.5f * x * (1.0f + tanh(inner));
        }
    }
}

// ==================================================================
// GELU backward (float4 vectorized)
// ==================================================================
kernel void gelu_backward(
    device const float *input   [[buffer(0)]],
    device const float *dy      [[buffer(1)]],
    device float       *dx      [[buffer(2)]],
    constant uint      *params  [[buffer(3)]],  // n
    uint3 gid                   [[thread_position_in_grid]])
{
    const uint n = params[0];
    const uint idx = gid.x * 4;
    const float c = 0.7978845608f;
    const float k = 0.044715f;

    if (idx + 3 < n) {
        float4 x = float4(input[idx], input[idx + 1], input[idx + 2], input[idx + 3]);
        float4 x2 = x * x;
        float4 inner = c * (x + k * x * x2);
        float4 tanh_inner = tanh(inner);
        float4 sech2 = 1.0f - tanh_inner * tanh_inner;
        float4 dgelu = 0.5f * (1.0f + tanh_inner) +
                        0.5f * x * sech2 * c * (1.0f + 3.0f * k * x2);
        float4 d = float4(dy[idx], dy[idx + 1], dy[idx + 2], dy[idx + 3]);
        float4 result = d * dgelu;
        dx[idx]     = result.x;
        dx[idx + 1] = result.y;
        dx[idx + 2] = result.z;
        dx[idx + 3] = result.w;
    } else {
        for (uint i = idx; i < min(idx + 4, n); i++) {
            float x = input[i];
            float x2 = x * x;
            float inner = c * (x + k * x * x2);
            float tanh_inner = tanh(inner);
            float sech2 = 1.0f - tanh_inner * tanh_inner;
            float dgelu = 0.5f * (1.0f + tanh_inner) +
                          0.5f * x * sech2 * c * (1.0f + 3.0f * k * x2);
            dx[i] = dy[i] * dgelu;
        }
    }
}

// ==================================================================
// RoPE forward
// ==================================================================
kernel void rope_forward(
    device const float *x       [[buffer(0)]],
    device float       *out     [[buffer(1)]],
    constant uint      *params  [[buffer(2)]],  // B, H, N, D, offset
    uint3 tgid                  [[threadgroup_position_in_grid]],
    uint3 tid                   [[thread_position_in_threadgroup]])
{
    const uint N = params[2];
    const uint D = params[3];
    const uint rope_offset = params[4];
    const uint half_D = D / 2;

    const uint d = tgid.x * 64 + tid.x;
    if (d >= half_D) return;

    const uint pos = tgid.y;
    if (pos >= N) return;

    const uint bh = tgid.z;
    const uint base = bh * N * D + pos * D;

    float freq = 1.0f / pow(10000.0f, float(2 * d) / float(D));
    float angle = float(pos + rope_offset) * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    float x0 = x[base + d];
    float x1 = x[base + d + half_D];

    out[base + d]          = x0 * cos_a - x1 * sin_a;
    out[base + d + half_D] = x0 * sin_a + x1 * cos_a;
}

// ==================================================================
// RoPE backward
// ==================================================================
kernel void rope_backward(
    device const float *dy      [[buffer(0)]],
    device float       *dx      [[buffer(1)]],
    constant uint      *params  [[buffer(2)]],  // B, H, N, D, offset
    uint3 tgid                  [[threadgroup_position_in_grid]],
    uint3 tid                   [[thread_position_in_threadgroup]])
{
    const uint N = params[2];
    const uint D = params[3];
    const uint rope_offset = params[4];
    const uint half_D = D / 2;

    const uint d = tgid.x * 64 + tid.x;
    if (d >= half_D) return;

    const uint pos = tgid.y;
    if (pos >= N) return;

    const uint bh = tgid.z;
    const uint base = bh * N * D + pos * D;

    float freq = 1.0f / pow(10000.0f, float(2 * d) / float(D));
    float angle = float(pos + rope_offset) * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    float dy0 = dy[base + d];
    float dy1 = dy[base + d + half_D];

    dx[base + d]          = dy0 * cos_a + dy1 * sin_a;
    dx[base + d + half_D] = -dy0 * sin_a + dy1 * cos_a;
}

// ==================================================================
// Embedding lookup (float4 vectorized)
// ==================================================================
kernel void embedding_lookup(
    device const float *table    [[buffer(0)]],
    device const uint  *indices  [[buffer(1)]],
    device float       *output   [[buffer(2)]],
    constant uint      *params   [[buffer(3)]],  // vocab_size, dim, seq_len
    uint3 tgid                   [[threadgroup_position_in_grid]],
    uint3 tid                    [[thread_position_in_threadgroup]])
{
    const uint dim = params[1];
    const uint d = tgid.x * 64 + tid.x;
    if (d >= dim) return;

    const uint pos = tgid.y;
    const uint idx = indices[pos];

    output[pos * dim + d] = table[idx * dim + d];
}

// ==================================================================
// Embedding backward
// ==================================================================
kernel void embedding_backward(
    device const float *dy       [[buffer(0)]],
    device const uint  *indices  [[buffer(1)]],
    device float       *dtable   [[buffer(2)]],
    constant uint      *params   [[buffer(3)]],  // vocab_size, dim, seq_len
    uint3 tgid                   [[threadgroup_position_in_grid]],
    uint3 tid                    [[thread_position_in_threadgroup]])
{
    const uint dim = params[1];
    const uint d = tgid.x * 64 + tid.x;
    if (d >= dim) return;

    const uint pos = tgid.y;
    const uint idx = indices[pos];

    dtable[idx * dim + d] += dy[pos * dim + d];
}

// ==================================================================
// Phase 2.3: Fused Softmax + Cross-Entropy Loss (optimized)
// ==================================================================
kernel void softmax_ce_loss(
    device const float *logits   [[buffer(0)]],
    device const uint  *targets  [[buffer(1)]],
    device float       *loss     [[buffer(2)]],
    device float       *dlogits  [[buffer(3)]],
    constant uint      *params   [[buffer(4)]],  // vocab_size
    uint3 pos                    [[threadgroup_position_in_grid]],
    uint3 tid                    [[thread_position_in_threadgroup]],
    uint3 tg_size                [[threads_per_threadgroup]])
{
    const uint V = params[0];
    const uint batch_id = pos.x;
    const uint local_id = tid.x;
    const uint local_size = tg_size.x;
    const uint offset = batch_id * V;
    const uint target = targets[batch_id];

    // Find max (for numerical stability)
    threadgroup float shared_max[256];
    float local_max = -INFINITY;
    for (uint i = local_id; i < V; i += local_size) {
        local_max = max(local_max, logits[offset + i]);
    }
    shared_max[local_id] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = local_size / 2; s > 0; s >>= 1) {
        if (local_id < s) shared_max[local_id] = max(shared_max[local_id], shared_max[local_id + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared_max[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Sum of exp
    threadgroup float shared_sum[256];
    float local_sum = 0.0f;
    for (uint i = local_id; i < V; i += local_size) {
        local_sum += exp(logits[offset + i] - max_val);
    }
    shared_sum[local_id] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = local_size / 2; s > 0; s >>= 1) {
        if (local_id < s) shared_sum[local_id] += shared_sum[local_id + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_exp = shared_sum[0];
    float log_sum_exp = log(sum_exp) + max_val;

    if (local_id == 0) {
        loss[batch_id] = log_sum_exp - logits[offset + target];
    }

    // Compute softmax probabilities and gradient
    for (uint i = local_id; i < V; i += local_size) {
        float prob = exp(logits[offset + i] - log_sum_exp);
        dlogits[offset + i] = prob - (i == target ? 1.0f : 0.0f);
    }
}

// ==================================================================
// Phase 2.4: AdamW optimizer update (float4 vectorized)
// ==================================================================
kernel void adamw_update(
    device float       *param    [[buffer(0)]],
    device const float *grad     [[buffer(1)]],
    device float       *m        [[buffer(2)]],
    device float       *v        [[buffer(3)]],
    constant uint      *iparams  [[buffer(4)]],  // n, step
    constant float     *fparams  [[buffer(5)]],  // lr, beta1, beta2, eps, weight_decay
    uint3 gid                    [[thread_position_in_grid]])
{
    const uint n = iparams[0];
    const uint idx = gid.x * 4;
    const uint step = iparams[1];
    const float lr = fparams[0];
    const float beta1 = fparams[1];
    const float beta2 = fparams[2];
    const float eps = fparams[3];
    const float wd = fparams[4];

    float bc1 = 1.0f - pow(beta1, float(step));
    float bc2 = 1.0f - pow(beta2, float(step));

    // Process 4 elements at a time
    for (uint i = idx; i < min(idx + 4, n); i++) {
        float g = grad[i];
        float mi = beta1 * m[i] + (1.0f - beta1) * g;
        float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
        m[i] = mi;
        v[i] = vi;

        float m_hat = mi / bc1;
        float v_hat = vi / bc2;

        param[i] = param[i] * (1.0f - lr * wd) - lr * m_hat / (sqrt(v_hat) + eps);
    }
}

// ==================================================================
// Phase 2.4: Element-wise tensor addition (float4 vectorized)
// ==================================================================
kernel void add_tensors(
    device const float *a       [[buffer(0)]],
    device const float *b       [[buffer(1)]],
    device float       *out     [[buffer(2)]],
    constant uint      *params  [[buffer(3)]],  // n
    uint3 gid                   [[thread_position_in_grid]])
{
    const uint n = params[0];
    const uint idx = gid.x * 4;

    if (idx + 3 < n) {
        float4 va = float4(a[idx], a[idx + 1], a[idx + 2], a[idx + 3]);
        float4 vb = float4(b[idx], b[idx + 1], b[idx + 2], b[idx + 3]);
        float4 result = va + vb;
        out[idx]     = result.x;
        out[idx + 1] = result.y;
        out[idx + 2] = result.z;
        out[idx + 3] = result.w;
    } else {
        for (uint i = idx; i < min(idx + 4, n); i++) {
            out[i] = a[i] + b[i];
        }
    }
}

// ==================================================================
// RMSNorm forward: out = gamma * x / sqrt(mean(x²) + eps)
// ==================================================================
kernel void rms_norm(
    device const float *x        [[buffer(0)]],
    device float       *out      [[buffer(1)]],
    device const float *gamma    [[buffer(2)]],
    constant uint      *params   [[buffer(3)]],  // dim
    constant float     *fparams  [[buffer(4)]],  // eps
    uint3 pos                    [[threadgroup_position_in_grid]],
    uint3 tid                    [[thread_position_in_threadgroup]],
    uint3 tg_size                [[threads_per_threadgroup]])
{
    const uint dim = params[0];
    const float eps = fparams[0];
    const uint batch_id = pos.x;
    const uint local_id = tid.x;
    const uint local_size = tg_size.x;
    const uint offset = batch_id * dim;

    // Compute sum of squares using float4 where possible
    float ss = 0.0f;
    uint d = local_id * 4;
    for (; d + 3 < dim; d += local_size * 4) {
        float4 v = float4(x[offset + d], x[offset + d + 1],
                          x[offset + d + 2], x[offset + d + 3]);
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (uint i = d; i < dim; i += local_size) {
        if (i < dim) ss += x[offset + i] * x[offset + i];
    }

    // Threadgroup reduction for sum of squares
    threadgroup float shared_sum[256];
    shared_sum[local_id] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = local_size / 2; s > 0; s >>= 1) {
        if (local_id < s) shared_sum[local_id] += shared_sum[local_id + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(shared_sum[0] / float(dim) + eps);

    // Normalize: out = gamma * x * inv_rms
    for (uint i = local_id; i < dim; i += local_size) {
        out[offset + i] = gamma[i] * x[offset + i] * inv_rms;
    }
}

// ==================================================================
// RMSNorm backward
// ==================================================================
kernel void rms_norm_backward(
    device const float *x         [[buffer(0)]],
    device const float *dy        [[buffer(1)]],
    device const float *gamma     [[buffer(2)]],
    device float       *dx        [[buffer(3)]],
    device float       *dgamma    [[buffer(4)]],
    constant uint      *params    [[buffer(5)]],  // batch, dim
    constant float     *fparams   [[buffer(6)]],  // eps
    uint3 pos                     [[threadgroup_position_in_grid]],
    uint3 tid                     [[thread_position_in_threadgroup]],
    uint3 tg_size                 [[threads_per_threadgroup]])
{
    const uint dim = params[1];
    const float eps = fparams[0];
    const uint batch_id = pos.x;
    const uint local_id = tid.x;
    const uint local_size = tg_size.x;
    const uint offset = batch_id * dim;

    threadgroup float shared[256];

    // Recompute sum of squares
    float ss = 0.0f;
    for (uint i = local_id; i < dim; i += local_size) ss += x[offset + i] * x[offset + i];
    shared[local_id] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = local_size / 2; s > 0; s >>= 1) {
        if (local_id < s) shared[local_id] += shared[local_id + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(shared[0] / float(dim) + eps);
    float inv_rms3 = inv_rms * inv_rms * inv_rms;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute sum(dy * gamma * x)
    float dsum = 0.0f;
    for (uint i = local_id; i < dim; i += local_size) {
        dsum += dy[offset + i] * gamma[i] * x[offset + i];
    }
    shared[local_id] = dsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = local_size / 2; s > 0; s >>= 1) {
        if (local_id < s) shared[local_id] += shared[local_id + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_dy_gamma_x = shared[0];

    // dx = gamma * dy * inv_rms - x * inv_rms^3 * sum(dy * gamma * x) / dim
    for (uint i = local_id; i < dim; i += local_size) {
        dx[offset + i] = gamma[i] * dy[offset + i] * inv_rms
                        - x[offset + i] * inv_rms3 * sum_dy_gamma_x / float(dim);
        dgamma[i] += dy[offset + i] * x[offset + i] * inv_rms;
    }
}

// ==================================================================
// SiLU forward (float4 vectorized): out = x * sigmoid(x)
// ==================================================================
kernel void silu_forward(
    device const float *input   [[buffer(0)]],
    device float       *output  [[buffer(1)]],
    constant uint      *params  [[buffer(2)]],  // n
    uint3 gid                   [[thread_position_in_grid]])
{
    const uint n = params[0];
    const uint idx = gid.x * 4;

    if (idx + 3 < n) {
        float4 x = float4(input[idx], input[idx + 1], input[idx + 2], input[idx + 3]);
        float4 sig = 1.0f / (1.0f + exp(-x));
        float4 result = x * sig;
        output[idx]     = result.x;
        output[idx + 1] = result.y;
        output[idx + 2] = result.z;
        output[idx + 3] = result.w;
    } else {
        for (uint i = idx; i < min(idx + 4, n); i++) {
            float x = input[i];
            float sig = 1.0f / (1.0f + exp(-x));
            output[i] = x * sig;
        }
    }
}

// ==================================================================
// SiLU backward (float4 vectorized)
// dx = dy * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
//    = dy * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
// ==================================================================
kernel void silu_backward(
    device const float *input   [[buffer(0)]],
    device const float *dy      [[buffer(1)]],
    device float       *dx      [[buffer(2)]],
    constant uint      *params  [[buffer(3)]],  // n
    uint3 gid                   [[thread_position_in_grid]])
{
    const uint n = params[0];
    const uint idx = gid.x * 4;

    if (idx + 3 < n) {
        float4 x = float4(input[idx], input[idx + 1], input[idx + 2], input[idx + 3]);
        float4 sig = 1.0f / (1.0f + exp(-x));
        float4 dsilu = sig * (1.0f + x * (1.0f - sig));
        float4 d = float4(dy[idx], dy[idx + 1], dy[idx + 2], dy[idx + 3]);
        float4 result = d * dsilu;
        dx[idx]     = result.x;
        dx[idx + 1] = result.y;
        dx[idx + 2] = result.z;
        dx[idx + 3] = result.w;
    } else {
        for (uint i = idx; i < min(idx + 4, n); i++) {
            float x = input[i];
            float sig = 1.0f / (1.0f + exp(-x));
            float dsilu = sig * (1.0f + x * (1.0f - sig));
            dx[i] = dy[i] * dsilu;
        }
    }
}

// ==================================================================
// Element-wise tensor multiplication (float4 vectorized)
// ==================================================================
kernel void mul_tensors(
    device const float *a       [[buffer(0)]],
    device const float *b       [[buffer(1)]],
    device float       *out     [[buffer(2)]],
    constant uint      *params  [[buffer(3)]],  // n
    uint3 gid                   [[thread_position_in_grid]])
{
    const uint n = params[0];
    const uint idx = gid.x * 4;

    if (idx + 3 < n) {
        float4 va = float4(a[idx], a[idx + 1], a[idx + 2], a[idx + 3]);
        float4 vb = float4(b[idx], b[idx + 1], b[idx + 2], b[idx + 3]);
        float4 result = va * vb;
        out[idx]     = result.x;
        out[idx + 1] = result.y;
        out[idx + 2] = result.z;
        out[idx + 3] = result.w;
    } else {
        for (uint i = idx; i < min(idx + 4, n); i++) {
            out[i] = a[i] * b[i];
        }
    }
}

// ==================================================================
// Phase 3.1: LoRA forward kernel
// y = x @ W_frozen^T + x @ A^T @ B^T
// Combined: compute the LoRA delta only: delta = x @ A^T @ B^T
// ==================================================================
kernel void lora_forward(
    device const float *x        [[buffer(0)]],   // [M, in_features]
    device const float *A        [[buffer(1)]],   // [rank, in_features]
    device const float *B        [[buffer(2)]],   // [out_features, rank]
    device float       *delta    [[buffer(3)]],   // [M, out_features]
    constant uint      *params   [[buffer(4)]],   // M, in_features, out_features, rank
    uint2 gid                    [[threadgroup_position_in_grid]],
    uint2 tid                    [[thread_position_in_threadgroup]])
{
    const uint M = params[0];
    const uint in_f = params[1];
    const uint out_f = params[2];
    const uint rank = params[3];

    // Two-step: first x @ A^T -> [M, rank], then result @ B^T -> [M, out_f]
    // For efficiency, fuse into single pass
    const uint row = gid.y * TILE_SIZE + tid.y;
    const uint col = gid.x * TILE_SIZE + tid.x;

    if (row >= M || col >= out_f) return;

    // Compute x[row,:] @ A^T[:,r] @ B[col, r] = sum_r (sum_k x[row,k]*A[r,k]) * B[col,r]
    float acc = 0.0f;
    for (uint r = 0; r < rank; r++) {
        float xa = 0.0f;
        for (uint k = 0; k < in_f; k++) {
            xa += x[row * in_f + k] * A[r * in_f + k]; // x @ A^T
        }
        acc += xa * B[col * rank + r]; // result @ B^T
    }
    delta[row * out_f + col] = acc;
}

// ==================================================================
// Phase 3.1: 4-bit Dequantization kernel (for QLoRA)
// Dequantize INT4 packed weights to FP32
// ==================================================================
kernel void dequantize_4bit(
    device const uint8_t *packed   [[buffer(0)]],  // packed 4-bit weights
    device const float   *scales   [[buffer(1)]],  // per-group scales
    device const float   *zeros    [[buffer(2)]],  // per-group zeros
    device float         *output   [[buffer(3)]],  // dequantized FP32
    constant uint        *params   [[buffer(4)]],  // n_elements, group_size
    uint3 gid                      [[thread_position_in_grid]])
{
    const uint n = params[0];
    const uint group_size = params[1];
    const uint idx = gid.x * 2; // Each byte has 2 4-bit values

    if (idx >= n) return;

    uint8_t packed_val = packed[gid.x];
    uint group_id = idx / group_size;
    float scale = scales[group_id];
    float zero = zeros[group_id];

    // Low nibble
    float val0 = (float)(packed_val & 0x0F) * scale + zero;
    output[idx] = val0;

    // High nibble
    if (idx + 1 < n) {
        float val1 = (float)((packed_val >> 4) & 0x0F) * scale + zero;
        output[idx + 1] = val1;
    }
}
