import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

from pycuda.compiler import SourceModule

_BLOCK_SIZE = 256

_KERNEL_CODE = r"""
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)

__global__ __launch_bounds__(256, 4) void LayerNormCUDAKernel(
    const float4* __restrict__ input,
    const float4* __restrict__ gamma,
    const float4* __restrict__ beta,
    float4* __restrict__ output,
    int row_size_f4,
    int row_size,
    float eps)
{
    const int row = blockIdx.x;
    const float4* row_in = input + (size_t)row * row_size_f4;
    float4* row_out = output + (size_t)row * row_size_f4;
    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp_id = tid >> 5;

    __shared__ double s_sum[WARPS_PER_BLOCK];
    __shared__ double s_sumsq[WARPS_PER_BLOCK];

    double sum = 0.0, sumsq = 0.0;
    for (int j = tid; j < row_size_f4; j += BLOCK_SIZE) {
        float4 v = row_in[j];
        sum   += (double)v.x + v.y + v.z + v.w;
        sumsq += (double)v.x * v.x + (double)v.y * v.y
               + (double)v.z * v.z + (double)v.w * v.w;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum   += __shfl_down_sync(0xffffffff, sum,   offset);
        sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);
    }

    if (lane == 0) {
        s_sum[warp_id] = sum;
        s_sumsq[warp_id] = sumsq;
    }
    __syncthreads();

    __shared__ float mu, inv_sigma;
    if (warp_id == 0) {
        double s  = (lane < WARPS_PER_BLOCK) ? s_sum[lane]   : 0.0;
        double sq = (lane < WARPS_PER_BLOCK) ? s_sumsq[lane] : 0.0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            s  += __shfl_down_sync(0xffffffff, s,  offset);
            sq += __shfl_down_sync(0xffffffff, sq, offset);
        }
        if (lane == 0) {
            double mean = s / row_size;
            double var = sq / row_size - mean * mean;
            mu = (float)mean;
            inv_sigma = 1.0f / sqrtf((float)var + eps);
        }
    }
    __syncthreads();

    const float mu_r = mu;
    const float inv_sigma_r = inv_sigma;
    const float neg_mu_inv_sigma = -mu_r * inv_sigma_r;

    for (int j = tid; j < row_size_f4; j += BLOCK_SIZE) {
        float4 x = row_in[j];
        float4 g = gamma[j];
        float4 b = beta[j];
        float4 out;
        out.x = fmaf(inv_sigma_r, x.x * g.x, fmaf(neg_mu_inv_sigma, g.x, b.x));
        out.y = fmaf(inv_sigma_r, x.y * g.y, fmaf(neg_mu_inv_sigma, g.y, b.y));
        out.z = fmaf(inv_sigma_r, x.z * g.z, fmaf(neg_mu_inv_sigma, g.z, b.z));
        out.w = fmaf(inv_sigma_r, x.w * g.w, fmaf(neg_mu_inv_sigma, g.w, b.w));
        row_out[j] = out;
    }
}
"""

class _LayerNormState:
    def __init__(self):
        self.d_in = None
        self.d_out = None
        self.d_g = None
        self.d_b = None
        self.total_bytes = 0
        self.row_bytes = 0
        self.out = None

    def ensure(self, total_floats, row_size, input):
        total_b = total_floats * 4
        row_b = row_size * 4
        if total_b > self.total_bytes:
            if self.d_in is not None:
                self.d_in.free()
                self.d_out.free()
            self.d_in = cuda.mem_alloc(total_b)
            self.d_out = cuda.mem_alloc(total_b)
            self.out = np.empty_like(input)
            self.total_bytes = total_b
        if row_b > self.row_bytes:
            if self.d_g is not None:
                self.d_g.free()
                self.d_b.free()
            self.d_g = cuda.mem_alloc(row_b)
            self.d_b = cuda.mem_alloc(row_b)
            self.row_bytes = row_b


_state = _LayerNormState()
_mod = SourceModule(_KERNEL_CODE, options=["-O3", "-use_fast_math"])
_kernel = _mod.get_function("LayerNormCUDAKernel")

def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """
    Apply Layer Normalization to each row of the input matrix.

    Parameters
    ----------
    input : list or numpy.ndarray of float
        Flattened matrix in row-major order. Its length must be divisible by row_size.
    gamma : list or numpy.ndarray of float
        Scale parameter, length = row_size.
    beta : list or numpy.ndarray of float
        Shift parameter, length = row_size.
    row_size : int
        Number of features per row (i.e., number of columns).
    eps : float, optional
        Small constant for numerical stability.

    Returns
    -------
    numpy.ndarray
        Flattened matrix of the same shape as input, containing the row-wise
        normalized results.
    """
    a = np.asarray(input, dtype=np.float32).ravel()
    g = np.asarray(gamma, dtype=np.float32).ravel()
    b = np.asarray(beta, dtype=np.float32).ravel()

    _state.ensure(a.size, row_size, a)

    cuda.memcpy_htod(_state.d_in, a)
    cuda.memcpy_htod(_state.d_g, g)
    cuda.memcpy_htod(_state.d_b, b)

    _kernel(
        _state.d_in, _state.d_g, _state.d_b, _state.d_out,
        np.int32(row_size // 4), np.int32(row_size), np.float32(eps),
        block=(_BLOCK_SIZE, 1, 1),
        grid=(a.size // row_size, 1),
    )

    cuda.memcpy_dtoh(_state.out, _state.d_out)
    return _state.out
