#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {


template <typename StateT>
__device__ __forceinline__ void cp_async_16b(StateT* smem_ptr,
                                             const StateT* gmem_ptr) {
  const uint32_t smem_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :
               : "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;\n" ::: "memory");
}

template <typename StateT, int ChunkV, int DimK, int Stages>
__device__ __forceinline__ void copy_state_chunk(StateT* shared_state,
                                                 const StateT* state, int chunk,
                                                 int thread, int threads) {
  constexpr int kElementsPerCopy = 16 / sizeof(StateT);
  constexpr int kCopiesPerChunk = ChunkV * DimK / kElementsPerCopy;
  const int stage = chunk % Stages;
  for (int copy = thread; copy < kCopiesPerChunk; copy += threads) {
    const int element = copy * kElementsPerCopy;
    cp_async_16b(shared_state + stage * ChunkV * DimK + element,
                 state + chunk * ChunkV * DimK + element);
  }
  cp_async_commit();
}

template <typename StateT>
__device__ __forceinline__ float4 load_state4(const StateT* state);

template <>
__device__ __forceinline__ float4 load_state4<float>(const float* state) {
  return *reinterpret_cast<const float4*>(state);
}

template <>
__device__ __forceinline__ float4
load_state4<__nv_bfloat16>(const __nv_bfloat16* state) {
  const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(state);
  const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(state + 2);
  return make_float4(__bfloat162float(lo.x), __bfloat162float(lo.y),
                     __bfloat162float(hi.x), __bfloat162float(hi.y));
}

template <typename StateT>
__device__ __forceinline__ void store_state4(StateT* state, float4 value);

template <>
__device__ __forceinline__ void store_state4<float>(float* state,
                                                    float4 value) {
  *reinterpret_cast<float4*>(state) = value;
}

template <>
__device__ __forceinline__ void store_state4<__nv_bfloat16>(
    __nv_bfloat16* state, float4 value) {
  *reinterpret_cast<__nv_bfloat162*>(state) =
      __floats2bfloat162_rn(value.x, value.y);
  *reinterpret_cast<__nv_bfloat162*>(state + 2) =
      __floats2bfloat162_rn(value.z, value.w);
}

constexpr int kDimK = 128;
constexpr int kDimV = 128;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kChunkV = 32;
constexpr int kNumChunks = kDimV / kChunkV;
constexpr int kRowsPerWarp = kChunkV / kWarps;
constexpr int kMaxMtpTokens = 8;
constexpr int kDtBiasFloat32 = 0;
constexpr int kDtBiasBFloat16 = 1;
constexpr int kDtBiasFloat16 = 2;

struct GdnDecodeStrides {
  int64_t mixed_row;
  int64_t a_row;
  int64_t b_row;
  int64_t gate_row;
  int64_t state_slot;
};

__device__ __forceinline__ float sigmoid_fast(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float silu_fast(float x) {
  return x * sigmoid_fast(x);
}

__device__ __forceinline__ float softplus_fast(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}

__device__ __forceinline__ float load_dt_bias(const void* dt_bias, int head,
                                              int dt_bias_type) {
  if (dt_bias_type == kDtBiasBFloat16) {
    return __bfloat162float(static_cast<const __nv_bfloat16*>(dt_bias)[head]);
  }
  if (dt_bias_type == kDtBiasFloat16) {
    return __half2float(static_cast<const __half*>(dt_bias)[head]);
  }
  return static_cast<const float*>(dt_bias)[head];
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

struct Sum2 {
  float x;
  float y;
};

__device__ __forceinline__ Sum2 warp_reduce_sum_pair(float x, float y) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffffu, x, offset);
    y += __shfl_xor_sync(0xffffffffu, y, offset);
  }
  return {x, y};
}

template <typename StateT, int ValueHeadsPerKeyHead, bool SigmoidGate>
__global__ __launch_bounds__(kThreads, 2) void gdn_decode_post_conv_mtp_kernel(
    const __nv_bfloat16* __restrict__ mixed_qkv,
    const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
    const float* __restrict__ a_log, const void* __restrict__ dt_bias,
    const int* __restrict__ state_indices, const int* __restrict__ cu_seqlens,
    const int* __restrict__ num_accepted_tokens, StateT* __restrict__ state,
    const __nv_bfloat16* __restrict__ output_gate,
    const void* __restrict__ norm_weight, __nv_bfloat16* __restrict__ out,
    int H, int HV, int state_indices_width, int dt_bias_type,
    bool norm_weight_is_bf16, float scale, float norm_eps,
    GdnDecodeStrides strides) {
  const int request = blockIdx.x;
  const int value_head = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int bos = cu_seqlens[request];
  const int eos = cu_seqlens[request + 1];
  const int num_tokens = eos - bos;
  if (num_tokens <= 0) {
    return;
  }

  const int accepted = num_accepted_tokens[request];
  const int source_slot =
      accepted > 0 && accepted <= state_indices_width
          ? state_indices[request * state_indices_width + accepted - 1]
          : 0;
  if (source_slot <= 0 || num_tokens > kMaxMtpTokens) {
    for (int linear = tid; linear < num_tokens * kDimV; linear += kThreads) {
      const int token = bos + linear / kDimV;
      const int value = linear % kDimV;
      const int64_t out_offset =
          (static_cast<int64_t>(token) * HV + value_head) * kDimV + value;
      out[out_offset] = __float2bfloat16(0.0f);
    }
    return;
  }

  const int key_head = value_head / ValueHeadsPerKeyHead;
  __shared__ StateT shared_state[2][kChunkV][kDimK];
  __shared__ float shared_q[kMaxMtpTokens][kDimK];
  __shared__ float shared_k[kMaxMtpTokens][kDimK];
  __shared__ __nv_bfloat16 shared_v[kMaxMtpTokens][kDimV];
  __shared__ __nv_bfloat16 shared_out[kMaxMtpTokens][kDimV];
  __shared__ float shared_decay[kMaxMtpTokens];
  __shared__ float shared_beta[kMaxMtpTokens];

  StateT* source_state =
      state + static_cast<int64_t>(source_slot) * strides.state_slot +
      value_head * kDimV * kDimK;
  copy_state_chunk<StateT, kChunkV, kDimK, 2>(&shared_state[0][0][0],
                                              source_state, 0, tid, kThreads);

  if (warp < num_tokens) {
    const int t = warp;
    const int token = bos + t;
    const int64_t mixed_base = static_cast<int64_t>(token) * strides.mixed_row;
    float q_values[4];
    float k_values[4];
    float q_square = 0.0f;
    float k_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      q_values[i] =
          __bfloat162float(mixed_qkv[mixed_base + key_head * kDimK + dim]);
      k_values[i] = __bfloat162float(
          mixed_qkv[mixed_base + H * kDimK + key_head * kDimK + dim]);
      shared_v[t][dim] =
          mixed_qkv[mixed_base + 2 * H * kDimK + value_head * kDimV + dim];
      q_square += q_values[i] * q_values[i];
      k_square += k_values[i] * k_values[i];
    }
    const Sum2 qk_sums = warp_reduce_sum_pair(q_square, k_square);
    const float q_scale = __shfl_sync(
        0xffffffffu, lane == 0 ? rsqrtf(qk_sums.x + 1.0e-6f) * scale : 0.0f, 0);
    const float k_scale = __shfl_sync(
        0xffffffffu, lane == 0 ? rsqrtf(qk_sums.y + 1.0e-6f) : 0.0f, 0);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      shared_q[t][dim] = q_values[i] * q_scale;
      shared_k[t][dim] = k_values[i] * k_scale;
    }
    if (lane == 0) {
      const float a_value = __bfloat162float(
          a[static_cast<int64_t>(token) * strides.a_row + value_head]);
      const float b_value = __bfloat162float(
          b[static_cast<int64_t>(token) * strides.b_row + value_head]);
      const float g = -__expf(a_log[value_head]) *
                      softplus_fast(a_value + load_dt_bias(dt_bias, value_head,
                                                           dt_bias_type));
      shared_decay[t] = __expf(g);
      shared_beta[t] = sigmoid_fast(b_value);
    }
  }
  __syncthreads();

  const int k_base = lane * 4;
  int rows[kRowsPerWarp];
#pragma unroll
  for (int row = 0; row < kRowsPerWarp; ++row) {
    rows[row] = warp + row * kWarps;
  }

#pragma unroll
  for (int chunk = 0; chunk < kNumChunks; ++chunk) {
    cp_async_wait_all();
    __syncthreads();
    if (chunk + 1 < kNumChunks) {
      copy_state_chunk<StateT, kChunkV, kDimK, 2>(
          &shared_state[0][0][0], source_state, chunk + 1, tid, kThreads);
    }

    float h[kRowsPerWarp][4];
#pragma unroll
    for (int row = 0; row < kRowsPerWarp; ++row) {
      const float4 state_value =
          load_state4(&shared_state[chunk & 1][rows[row]][k_base]);
      h[row][0] = state_value.x;
      h[row][1] = state_value.y;
      h[row][2] = state_value.z;
      h[row][3] = state_value.w;
    }

    for (int t = 0; t < num_tokens; ++t) {
      const float4 q4 = *reinterpret_cast<const float4*>(&shared_q[t][k_base]);
      const float4 k4 = *reinterpret_cast<const float4*>(&shared_k[t][k_base]);
      const float q_values[4] = {q4.x, q4.y, q4.z, q4.w};
      const float k_values[4] = {k4.x, k4.y, k4.z, k4.w};

      float dot_hk[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row][i] *= shared_decay[t];
          dot_hk[row] += h[row][i] * k_values[i];
        }
      }
      const Sum2 dot_hk_01 = warp_reduce_sum_pair(dot_hk[0], dot_hk[1]);
      const Sum2 dot_hk_23 = warp_reduce_sum_pair(dot_hk[2], dot_hk[3]);
      const float reduced_hk[kRowsPerWarp] = {dot_hk_01.x, dot_hk_01.y,
                                              dot_hk_23.x, dot_hk_23.y};

      float dot_hq[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
        const int value = chunk * kChunkV + rows[row];
        const float delta =
            (__bfloat162float(shared_v[t][value]) - reduced_hk[row]) *
            shared_beta[t];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row][i] += k_values[i] * delta;
          dot_hq[row] += h[row][i] * q_values[i];
        }
      }
      const Sum2 dot_hq_01 = warp_reduce_sum_pair(dot_hq[0], dot_hq[1]);
      const Sum2 dot_hq_23 = warp_reduce_sum_pair(dot_hq[2], dot_hq[3]);
      if (lane == 0) {
        shared_out[t][chunk * kChunkV + rows[0]] =
            __float2bfloat16(dot_hq_01.x);
        shared_out[t][chunk * kChunkV + rows[1]] =
            __float2bfloat16(dot_hq_01.y);
        shared_out[t][chunk * kChunkV + rows[2]] =
            __float2bfloat16(dot_hq_23.x);
        shared_out[t][chunk * kChunkV + rows[3]] =
            __float2bfloat16(dot_hq_23.y);
      }

      const int destination_slot =
          state_indices[request * state_indices_width + t];
      if (destination_slot > 0) {
        StateT* destination_state =
            state +
            static_cast<int64_t>(destination_slot) * strides.state_slot +
            value_head * kDimV * kDimK;
#pragma unroll
        for (int row = 0; row < kRowsPerWarp; ++row) {
          const int value = chunk * kChunkV + rows[row];
          store_state4(destination_state + value * kDimK + k_base,
                       make_float4(h[row][0], h[row][1], h[row][2], h[row][3]));
        }
      }
    }
  }
  __syncthreads();

  if (warp < num_tokens) {
    const int t = warp;
    float output_values[4];
    float sum_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int value = lane + i * 32;
      output_values[i] = __bfloat162float(shared_out[t][value]);
      sum_square += output_values[i] * output_values[i];
    }
    sum_square = warp_reduce_sum(sum_square);
    const float rstd =
        rsqrtf(sum_square / static_cast<float>(kDimV) + norm_eps);
    const int token = bos + t;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int value = lane + i * 32;
      const float gate_input = __bfloat162float(
          output_gate[static_cast<int64_t>(token) * strides.gate_row +
                      value_head * kDimV + value]);
      const float gate =
          SigmoidGate ? sigmoid_fast(gate_input) : silu_fast(gate_input);
      const float weight =
          norm_weight_is_bf16
              ? __bfloat162float(
                    static_cast<const __nv_bfloat16*>(norm_weight)[value])
              : static_cast<const float*>(norm_weight)[value];
      const int64_t out_offset =
          (static_cast<int64_t>(token) * HV + value_head) * kDimV + value;
      out[out_offset] =
          __float2bfloat16(output_values[i] * rstd * weight * gate);
    }
  }
}

// Deferred commit, served form (throughput arm, 19 Oct; M2a).
//
// Per request row r (a spec decode), with acc = num_accepted[r] (previous step), T = this step's tokens,
// pos0 = seq_lens[r] - T, prev_pos0 = pos0 - acc:
//   valid  = flag[flag_index[r]] == (ident[r], prev_pos0 + 1)   (the previous step ran deferred for this request)
//   source = valid ? commit[r] (+ replay of the previous step's acc buffered tokens) : read_state[slots[r, acc-1]]
//   near   = a multiple of block_size lies in [pos0 - 2W - 2, pos0 + 2W + 2]  (W = slots width)
// MODE_STEP:  near -> per-token states written to tok[r, t] for t < T (the pinned kernel's layout, so the block-
//             crossing copies and prefix-cache snapshots read what they always read); flag untouched.
//             else -> commit[r] = source (bf16, skipped when it is already there), no per-token stores, this step's
//             normalized k / v / decay / beta stored in buf[r] (this value head's 32 KB row block of the spec slot 1
//             page, written after every state read), flag = (ident, pos0 + 1).
// MODE_FLUSH: valid -> tok[r, acc - 1] = source (restores the pinned layout before a non-deferred path); no outputs.
// Pointers are base + index * stride, so the same kernel serves the real pool and a row-indexed shadow (self-check).
constexpr int kModeStep = 0;
constexpr int kModeFlush = 1;

struct DeferredPtrs {
  __nv_bfloat16* commit; const int* commit_index; int64_t commit_stride; int commit_index_stride;
  __nv_bfloat16* buf; const int* buf_index; int64_t buf_stride; int buf_index_stride;
  __nv_bfloat16* tok; const int* tok_index; int64_t tok_stride; int tok_width;
  const __nv_bfloat16* read_state; int64_t read_stride;
  int* flags; const int* flag_index; int flag_index_stride; const int* ident; int ident_stride;
  const int* seq_lens; int block_size;
};

template <int ValueHeadsPerKeyHead, bool SigmoidGate, int Mode>
__global__ __launch_bounds__(kThreads, 2) void gdn_deferred2_kernel(
    const __nv_bfloat16* __restrict__ mixed_qkv,
    const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
    const float* __restrict__ a_log, const void* __restrict__ dt_bias,
    const int* __restrict__ state_indices, const int* __restrict__ cu_seqlens,
    const int* __restrict__ num_accepted_tokens,
    const __nv_bfloat16* __restrict__ output_gate,
    const void* __restrict__ norm_weight, __nv_bfloat16* __restrict__ out,
    int H, int HV, int state_indices_width, int dt_bias_type,
    bool norm_weight_is_bf16, float scale, float norm_eps,
    GdnDecodeStrides strides, DeferredPtrs p) {
  using StateT = __nv_bfloat16;
  const int request = blockIdx.x;
  const int value_head = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int bos = cu_seqlens[request];
  const int eos = cu_seqlens[request + 1];
  const int num_tokens = eos - bos;
  if (num_tokens <= 0) {
    return;
  }
  const int W = state_indices_width;
  const int accepted = num_accepted_tokens[request];
  const int pos0 = p.seq_lens[request] - num_tokens;
  const int prev_pos0 = pos0 - accepted;
  const int fidx = p.flag_index[request * p.flag_index_stride];
  const int ident = p.ident[request * p.ident_stride];
  // Flags are per (block, value head): each CTA reads and writes only its own pair, so no CTA of this launch can see
  // another's update.
  int* flag = p.flags + 2 * (static_cast<int64_t>(fidx) * HV + value_head);
  const bool valid = W >= 2 && accepted >= 1 && accepted <= kMaxMtpTokens && fidx > 0 &&
                     flag[0] == ident && flag[1] == prev_pos0 + 1;
  const int lo = max(pos0 - 2 * W - 2, 0);
  const int hi = pos0 + 2 * W + 2;
  const bool near = (hi / p.block_size) != (lo / p.block_size) || W < 2;
  const int source_slot =
      accepted > 0 && accepted <= W ? state_indices[request * W + accepted - 1] : 0;
  if (Mode == kModeFlush && !valid) {
    return;
  }
  if ((!valid && source_slot <= 0) || num_tokens > kMaxMtpTokens) {
    if (Mode == kModeStep) {
      for (int linear = tid; linear < num_tokens * kDimV; linear += kThreads) {
        const int token = bos + linear / kDimV;
        const int value = linear % kDimV;
        const int64_t out_offset =
            (static_cast<int64_t>(token) * HV + value_head) * kDimV + value;
        out[out_offset] = __float2bfloat16(0.0f);
      }
    }
    return;
  }
  const int replay = valid ? accepted : 0;
  const int64_t head_off = static_cast<int64_t>(value_head) * kDimV * kDimK;
  StateT* commit_state =
      p.commit + static_cast<int64_t>(p.commit_index[request * p.commit_index_stride]) * p.commit_stride + head_off;
  const StateT* source_state =
      valid ? commit_state : p.read_state + static_cast<int64_t>(source_slot) * p.read_stride + head_off;
  // Buffer for this (request, value head): k fp32[8][128] | v bf16[8][128] | decay, beta fp32[8][2].
  StateT* buf_head =
      p.buf + static_cast<int64_t>(p.buf_index[request * p.buf_index_stride]) * p.buf_stride + head_off;
  float* buf_k = reinterpret_cast<float*>(buf_head);
  __nv_bfloat16* buf_v = reinterpret_cast<__nv_bfloat16*>(buf_k + kMaxMtpTokens * kDimK);
  float* buf_db = reinterpret_cast<float*>(buf_v + kMaxMtpTokens * kDimV);

  const int key_head = value_head / ValueHeadsPerKeyHead;
  __shared__ StateT shared_state[2][kChunkV][kDimK];
  __shared__ float shared_q[kMaxMtpTokens][kDimK];
  __shared__ float shared_k[kMaxMtpTokens][kDimK];
  __shared__ __nv_bfloat16 shared_v[kMaxMtpTokens][kDimV];
  __shared__ __nv_bfloat16 shared_out[kMaxMtpTokens][kDimV];
  __shared__ float shared_decay[kMaxMtpTokens];
  __shared__ float shared_beta[kMaxMtpTokens];
  __shared__ float shared_pk[kMaxMtpTokens][kDimK];
  __shared__ __nv_bfloat16 shared_pv[kMaxMtpTokens][kDimV];
  __shared__ float shared_pdecay[kMaxMtpTokens];
  __shared__ float shared_pbeta[kMaxMtpTokens];

  copy_state_chunk<StateT, kChunkV, kDimK, 2>(&shared_state[0][0][0], source_state, 0, tid, kThreads);
  for (int linear = tid; linear < replay * kDimK; linear += kThreads) {
    const int t = linear / kDimK, d = linear % kDimK;
    shared_pk[t][d] = buf_k[t * kDimK + d];
    shared_pv[t][d] = buf_v[t * kDimV + d];
  }
  if (tid < replay) {
    shared_pdecay[tid] = buf_db[tid * 2];
    shared_pbeta[tid] = buf_db[tid * 2 + 1];
  }

  if (Mode == kModeStep && warp < num_tokens) {
    const int t = warp;
    const int token = bos + t;
    const int64_t mixed_base = static_cast<int64_t>(token) * strides.mixed_row;
    float q_values[4];
    float k_values[4];
    float q_square = 0.0f;
    float k_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      q_values[i] =
          __bfloat162float(mixed_qkv[mixed_base + key_head * kDimK + dim]);
      k_values[i] = __bfloat162float(
          mixed_qkv[mixed_base + H * kDimK + key_head * kDimK + dim]);
      shared_v[t][dim] =
          mixed_qkv[mixed_base + 2 * H * kDimK + value_head * kDimV + dim];
      q_square += q_values[i] * q_values[i];
      k_square += k_values[i] * k_values[i];
    }
    const Sum2 qk_sums = warp_reduce_sum_pair(q_square, k_square);
    const float q_scale = __shfl_sync(
        0xffffffffu, lane == 0 ? rsqrtf(qk_sums.x + 1.0e-6f) * scale : 0.0f, 0);
    const float k_scale = __shfl_sync(
        0xffffffffu, lane == 0 ? rsqrtf(qk_sums.y + 1.0e-6f) : 0.0f, 0);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      shared_q[t][dim] = q_values[i] * q_scale;
      shared_k[t][dim] = k_values[i] * k_scale;
    }
    if (lane == 0) {
      const float a_value = __bfloat162float(
          a[static_cast<int64_t>(token) * strides.a_row + value_head]);
      const float b_value = __bfloat162float(
          b[static_cast<int64_t>(token) * strides.b_row + value_head]);
      const float g = -__expf(a_log[value_head]) *
                      softplus_fast(a_value + load_dt_bias(dt_bias, value_head,
                                                           dt_bias_type));
      shared_decay[t] = __expf(g);
      shared_beta[t] = sigmoid_fast(b_value);
    }
  }
  __syncthreads();

  const bool write_commit = Mode == kModeStep && !near && (replay > 0 || source_state != commit_state);
  const int k_base = lane * 4;
  int rows[kRowsPerWarp];
#pragma unroll
  for (int row = 0; row < kRowsPerWarp; ++row) {
    rows[row] = warp + row * kWarps;
  }

#pragma unroll
  for (int chunk = 0; chunk < kNumChunks; ++chunk) {
    cp_async_wait_all();
    __syncthreads();
    if (chunk + 1 < kNumChunks) {
      copy_state_chunk<StateT, kChunkV, kDimK, 2>(
          &shared_state[0][0][0], source_state, chunk + 1, tid, kThreads);
    }

    float h[kRowsPerWarp][4];
#pragma unroll
    for (int row = 0; row < kRowsPerWarp; ++row) {
      const float4 state_value =
          load_state4(&shared_state[chunk & 1][rows[row]][k_base]);
      h[row][0] = state_value.x;
      h[row][1] = state_value.y;
      h[row][2] = state_value.z;
      h[row][3] = state_value.w;
    }

    for (int t = 0; t < replay; ++t) {
      const float4 k4 = *reinterpret_cast<const float4*>(&shared_pk[t][k_base]);
      const float k_values[4] = {k4.x, k4.y, k4.z, k4.w};
      float dot_hk[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row][i] *= shared_pdecay[t];
          dot_hk[row] += h[row][i] * k_values[i];
        }
      }
      const Sum2 dot_hk_01 = warp_reduce_sum_pair(dot_hk[0], dot_hk[1]);
      const Sum2 dot_hk_23 = warp_reduce_sum_pair(dot_hk[2], dot_hk[3]);
      const float reduced_hk[kRowsPerWarp] = {dot_hk_01.x, dot_hk_01.y,
                                              dot_hk_23.x, dot_hk_23.y};
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
        const int value = chunk * kChunkV + rows[row];
        const float delta =
            (__bfloat162float(shared_pv[t][value]) - reduced_hk[row]) *
            shared_pbeta[t];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row][i] += k_values[i] * delta;
        }
      }
    }
    if (replay > 0) {
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
        const __nv_bfloat162 lo2 = __floats2bfloat162_rn(h[row][0], h[row][1]);
        const __nv_bfloat162 hi2 = __floats2bfloat162_rn(h[row][2], h[row][3]);
        h[row][0] = __bfloat162float(lo2.x);
        h[row][1] = __bfloat162float(lo2.y);
        h[row][2] = __bfloat162float(hi2.x);
        h[row][3] = __bfloat162float(hi2.y);
      }
    }
    if (Mode == kModeFlush) {
      StateT* dest =
          p.tok + static_cast<int64_t>(p.tok_index[request * p.tok_width + accepted - 1]) * p.tok_stride + head_off;
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
        const int value = chunk * kChunkV + rows[row];
        store_state4(dest + value * kDimK + k_base,
                     make_float4(h[row][0], h[row][1], h[row][2], h[row][3]));
      }
      continue;
    }
    if (write_commit) {
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
        const int value = chunk * kChunkV + rows[row];
        store_state4(commit_state + value * kDimK + k_base,
                     make_float4(h[row][0], h[row][1], h[row][2], h[row][3]));
      }
    }

    for (int t = 0; t < num_tokens; ++t) {
      const float4 q4 = *reinterpret_cast<const float4*>(&shared_q[t][k_base]);
      const float4 k4 = *reinterpret_cast<const float4*>(&shared_k[t][k_base]);
      const float q_values[4] = {q4.x, q4.y, q4.z, q4.w};
      const float k_values[4] = {k4.x, k4.y, k4.z, k4.w};

      float dot_hk[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row][i] *= shared_decay[t];
          dot_hk[row] += h[row][i] * k_values[i];
        }
      }
      const Sum2 dot_hk_01 = warp_reduce_sum_pair(dot_hk[0], dot_hk[1]);
      const Sum2 dot_hk_23 = warp_reduce_sum_pair(dot_hk[2], dot_hk[3]);
      const float reduced_hk[kRowsPerWarp] = {dot_hk_01.x, dot_hk_01.y,
                                              dot_hk_23.x, dot_hk_23.y};

      float dot_hq[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
        const int value = chunk * kChunkV + rows[row];
        const float delta =
            (__bfloat162float(shared_v[t][value]) - reduced_hk[row]) *
            shared_beta[t];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row][i] += k_values[i] * delta;
          dot_hq[row] += h[row][i] * q_values[i];
        }
      }
      const Sum2 dot_hq_01 = warp_reduce_sum_pair(dot_hq[0], dot_hq[1]);
      const Sum2 dot_hq_23 = warp_reduce_sum_pair(dot_hq[2], dot_hq[3]);
      if (lane == 0) {
        shared_out[t][chunk * kChunkV + rows[0]] =
            __float2bfloat16(dot_hq_01.x);
        shared_out[t][chunk * kChunkV + rows[1]] =
            __float2bfloat16(dot_hq_01.y);
        shared_out[t][chunk * kChunkV + rows[2]] =
            __float2bfloat16(dot_hq_23.x);
        shared_out[t][chunk * kChunkV + rows[3]] =
            __float2bfloat16(dot_hq_23.y);
      }
      if (near) {
        const int destination_slot = p.tok_index[request * p.tok_width + t];
        if (destination_slot > 0) {
          StateT* destination_state = p.tok + static_cast<int64_t>(destination_slot) * p.tok_stride + head_off;
#pragma unroll
          for (int row = 0; row < kRowsPerWarp; ++row) {
            const int value = chunk * kChunkV + rows[row];
            store_state4(destination_state + value * kDimK + k_base,
                         make_float4(h[row][0], h[row][1], h[row][2], h[row][3]));
          }
        }
      }
    }
  }
  if (Mode == kModeFlush) {
    return;
  }
  __syncthreads();

  if (!near) {
    // Every read of the source state is done (the last chunk's wait + barrier above), so the buffer may overwrite
    // this head's rows of the spec slot 1 page even when the source was that page.
    for (int linear = tid; linear < num_tokens * kDimK; linear += kThreads) {
      const int t = linear / kDimK, d = linear % kDimK;
      buf_k[t * kDimK + d] = shared_k[t][d];
      buf_v[t * kDimV + d] = shared_v[t][d];
    }
    if (tid < num_tokens) {
      buf_db[tid * 2] = shared_decay[tid];
      buf_db[tid * 2 + 1] = shared_beta[tid];
    }
    if (tid == 0 && fidx > 0) {
      flag[0] = ident;
      flag[1] = pos0 + 1;
    }
  } else if (tid == 0 && fidx > 0) {
    flag[0] = 0;  // a pinned-layout step: no buffer for this block and head
    flag[1] = 0;
  }

  if (warp < num_tokens) {
    const int t = warp;
    float output_values[4];
    float sum_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int value = lane + i * 32;
      output_values[i] = __bfloat162float(shared_out[t][value]);
      sum_square += output_values[i] * output_values[i];
    }
    sum_square = warp_reduce_sum(sum_square);
    const float rstd =
        rsqrtf(sum_square / static_cast<float>(kDimV) + norm_eps);
    const int token = bos + t;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int value = lane + i * 32;
      const float gate_input = __bfloat162float(
          output_gate[static_cast<int64_t>(token) * strides.gate_row +
                      value_head * kDimV + value]);
      const float gate =
          SigmoidGate ? sigmoid_fast(gate_input) : silu_fast(gate_input);
      const float weight =
          norm_weight_is_bf16
              ? __bfloat162float(
                    static_cast<const __nv_bfloat16*>(norm_weight)[value])
              : static_cast<const float*>(norm_weight)[value];
      const int64_t out_offset =
          (static_cast<int64_t>(token) * HV + value_head) * kDimV + value;
      out[out_offset] =
          __float2bfloat16(output_values[i] * rstd * weight * gate);
    }
  }
}

}  // namespace

// ctypes entry (bf16 state, HV/H = 3). mode 0 = deferred step, 1 = flush, 2 = upstream per-token kernel (rebuilt).
extern "C" int gdn2_launch(int mode, int sigmoid, const void* mixed_qkv, const void* a, const void* b,
                           const void* a_log, const void* dt_bias, int dt_bias_type, const void* state_indices,
                           int width, const void* cu_seqlens, const void* num_accepted, void* state,
                           const void* output_gate, const void* norm_weight, int norm_bf16, void* out, int N, int H,
                           int HV, float scale, float norm_eps, long long mixed_row, long long a_row, long long b_row,
                           long long gate_row, long long state_slot, const void* ptrs, void* stream_ptr) {
  if (HV != 3 * H) return -1;
  const GdnDecodeStrides strides{mixed_row, a_row, b_row, gate_row, state_slot};
  const dim3 grid(N, HV);
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
  using S = __nv_bfloat16;
  const S* mq = static_cast<const S*>(mixed_qkv);
  const S* av = static_cast<const S*>(a);
  const S* bv = static_cast<const S*>(b);
  const float* al = static_cast<const float*>(a_log);
  const int* si = static_cast<const int*>(state_indices);
  const int* cu = static_cast<const int*>(cu_seqlens);
  const int* na = static_cast<const int*>(num_accepted);
  const S* og = static_cast<const S*>(output_gate);
  S* o = static_cast<S*>(out);
  if (mode == 2) {
    if (sigmoid)
      gdn_decode_post_conv_mtp_kernel<S, 3, true><<<grid, kThreads, 0, stream>>>(
          mq, av, bv, al, dt_bias, si, cu, na, static_cast<S*>(state), og, norm_weight, o, H, HV, width,
          dt_bias_type, norm_bf16 != 0, scale, norm_eps, strides);
    else
      gdn_decode_post_conv_mtp_kernel<S, 3, false><<<grid, kThreads, 0, stream>>>(
          mq, av, bv, al, dt_bias, si, cu, na, static_cast<S*>(state), og, norm_weight, o, H, HV, width,
          dt_bias_type, norm_bf16 != 0, scale, norm_eps, strides);
    return static_cast<int>(cudaGetLastError());
  }
  const DeferredPtrs p = *static_cast<const DeferredPtrs*>(ptrs);  // void*: an anonymous-namespace type would make this symbol internal
#define D2(M, G)                                                                                              \
  gdn_deferred2_kernel<3, G, M><<<grid, kThreads, 0, stream>>>(mq, av, bv, al, dt_bias, si, cu, na, og,        \
                                                               norm_weight, o, H, HV, width, dt_bias_type,     \
                                                               norm_bf16 != 0, scale, norm_eps, strides, p)
  if (mode == 0) {
    if (sigmoid) D2(kModeStep, true); else D2(kModeStep, false);
  } else {
    if (sigmoid) D2(kModeFlush, true); else D2(kModeFlush, false);
  }
  return static_cast<int>(cudaGetLastError());
}

extern "C" int gdn2_ptrs_size() { return static_cast<int>(sizeof(DeferredPtrs)); }
