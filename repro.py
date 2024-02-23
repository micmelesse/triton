
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_root/3h/c3hndqx4v4wu62wc5blcsg6vukffsqgjnq2fega2zkpkjzjchuff.py
# Source Nodes: [abs_1, add, truediv], Original ATen: [aten.abs, aten.add, aten.div]
# abs_1 => abs_1
# add => add
# truediv => div
print("async_compile.triton")
triton_poi_fused_abs_add_div_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(
    size_hints=[32], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_div_0', 'mutated_arg_names': [], 'no_x_dim': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.abs(tmp0)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 / tmp3
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


async_compile.wait(globals())
del async_compile

def call(args):
    print("call")
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (2, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty((2, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [abs_1, add, truediv], Original ATen: [aten.abs, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        print("triton_poi_fused_abs_add_div_0:", triton_poi_fused_abs_add_div_0)
        print("arg0_1:", arg0_1)
        print("buf0:", buf0)
        print("stream0:", stream0)
        triton_poi_fused_abs_add_div_0.run(arg0_1, buf0, 2, grid=grid(2), stream=stream0)
        run_intermediate_hooks('div', buf0)
        del arg0_1
        return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    print("benchmark_compiled_module")
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    print("__main__")
    # from torch._inductor.wrapper_benchmark import compiled_module_main
    # compiled_module_main('None', benchmark_compiled_module)

    # just run once
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    call([arg0_1])
