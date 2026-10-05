import torch

import triton
import triton.language as tl
import torch.nn.functional as F

def naive_softmax(x: torch.tensor) -> torch.tensor:
    x_max = x.max(dim=1)[0]
    safe_x = x - x_max[:, None]
    numerator = torch.exp(safe_x)
    denominator = numerator.sum(dim=1)
    sm_out = numerator/denominator[:, None]
    return sm_out

@triton.jit
def _softmax_kernel(
    output_ptr,
    output_stride,
    input_ptr,
    input_stride,
    num_cols,
    block_size: tl.constexpr,
):
    row_index = tl.program_id(0)
    row_start = input_ptr + (row_index * input_stride)
    col_offsets = tl.arange(0, block_size)
    mask = col_offsets < num_cols
    input_pointers = row_start + col_offsets

    row = tl.load(input_pointers, mask=mask, other=float("-inf"))

    safe_row = row - tl.max(row, axis=0)
    num = tl.exp(safe_row)
    denm = tl.sum(num, axis=0)
    sm_out = num/denm

    output_row_ptr = output_ptr + (row_index * output_stride)
    output_pointers = output_row_ptr + col_offsets
    tl.store(output_pointers, sm_out, mask=mask)


def softmax_triton(x: torch.tensor) -> torch.tensor:
    rows, cols = x.shape
    assert x.dim() == 2, "Only accepts 2D dimensions"
    block_size = triton.next_power_of_2(cols)
    num_warps = 4
    if block_size > 2047:
        num_warps = 8
    if block_size > 4095:
        num_warps = 16

    grid = (rows,)
    sm_out = torch.zeros_like(x)
    _softmax_kernel[grid](
        sm_out,
        sm_out.stride(0),
        x,
        x.stride(0),
        cols,
        block_size=block_size,
        num_warps = num_warps
    )
    return sm_out

sample = torch.tensor([[1,2,3,4,5],[5,4,3,2,1]], dtype=torch.float32, device="cuda")

verify = F.softmax(sample, dim=1)

print(verify)

softmax = softmax_triton(sample)
print(softmax)

print(verify == softmax)