import triton
import triton.language as tl
import torch

@triton.jit
def test_dot_kernel(q_ptr, k_ptr, out_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    q = tl.load(q_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :])
    k = tl.load(k_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :])

    # This line is the test: will it raise error on trans_b?
    dot_result = tl.dot(q, k, trans_b=True)

    tl.store(out_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :], dot_result)

# Allocate input/output
BLOCK_M = 16
BLOCK_N = 16
q = torch.randn((BLOCK_M, BLOCK_N), device='cuda', dtype=torch.float32)
k = torch.randn((BLOCK_N, BLOCK_M), device='cuda', dtype=torch.float32)
out = torch.empty((BLOCK_M, BLOCK_N), device='cuda', dtype=torch.float32)

# Launch kernel
test_dot_kernel[(1,)](
    q_ptr=q,
    k_ptr=k,
    out_ptr=out,
    BLOCK_M=BLOCK_M,
    BLOCK_N=BLOCK_N,
)

print("✅ Triton `tl.dot(..., trans_b=True)` works!")

