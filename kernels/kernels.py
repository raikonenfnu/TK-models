import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl
import shark_turbine.kernel.wave as tkw
import torch
from numpy.testing import assert_allclose, assert_equal
import pytest
import os
import sympy

# def test_copy():
#     M = tkl.sym.M
#     N = tkl.sym.N
#     BLOCK_M = tkl.sym.BLOCK_M
#     BLOCK_N = tkl.sym.BLOCK_N
#     ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD
#     ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

#     constraints: list[tkw.Constraint] = [
#         tkw.HardwareConstraint(
#             threads_per_wave=64,
#             waves_per_block=(1, 1, 1),
#             vector_shapes={M: 1, N: BLOCK_N},
#         )
#     ]
#     # 1. Kernel (global memory)
#     # 1.b Grids (a group of blocks)
#     # 2. Blocks (a group of warps, multiple kernels)
#     # 3. Warp (minimum number of threads that GPU launches, MI300 -> 64, RDNA3 -> 32)
#     # 4. Thread

#     # (4, 2 ,1) -> (4 * 64, 2 , 1) -> number of threads in a block.
#     # hiplaunchkernel(gridX, gridY, gridZ, blockX, blockY, blockZ, kernel)
#     # (1,1,1) (2,2,1) -> thread_id.x
#     # tensor<2x2> tensor<2x2>, add them
#     #  (1,1,1) (2,2,1) -> thread_id.x
#     # output[thread_id.x, thread_id.y]] = lhs[thread_id.x, thread_id.y] + rhs[thread_id.x, thread_id.y]
#     # Total number of threads launched (gridX * blockX * numWarpsInX, -||-, )
#     # tensor<256x128>
#     # BLOCK_M = 1, BLOCK_N =128
#     # each block will have tensor<1x128>
#     # each warp will have tensor<1x128>
#     # 1 block = (1 * 64, 1, 1)
#     # 1 thread will take vector<2xf32>
#     # 1 thread will take vector<2xf32>
#     # tid.x vector<2xf32>
    

#     # ELEMS_PER_THREAD: 2
#     # num_of_waveX = BLOCK_M / WAVE_M
#     # (num_of_waveX * warpSize, num_of_WaveY, num_of_WaveZ)
#     constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
#     constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
#     constraints += [tkw.WaveConstraint(M, BLOCK_M)]
#     constraints += [tkw.WaveConstraint(N, BLOCK_N)]

#     @tkw.wave(constraints)
#     def test(
#         a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
#         b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
#     ):
#         res = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
#         tkw.write(res, b, elements_per_thread=ELEMS_PER_THREAD)

#     config = {"backend": "rocm", "device": "hip", "target": "gfx942"}

#     shape = (256, 128)
#     a = torch.randn(shape, dtype=torch.float16)
#     b = torch.zeros(shape, dtype=torch.float16)
#     with tk.gen.TestLaunchContext(
#         {
#             M: shape[0],
#             N: shape[1],
#             BLOCK_M: 1,
#             BLOCK_N: 128,
#             ELEMS_PER_THREAD: 2,
#             ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
#         },
#         canonicalize=True,
#         # run=True,
#         # run_config=config,
#     ):
#         import pdb; pdb.set_trace()
#         test(a, b)
#         assert_allclose(a, b)


# def test_add():
#     M = tkl.sym.M
#     N = tkl.sym.N
#     BLOCK_M = tkl.sym.BLOCK_M
#     BLOCK_N = tkl.sym.BLOCK_N
#     ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD
#     ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

#     constraints: list[tkw.Constraint] = [
#         tkw.HardwareConstraint(
#             threads_per_wave=64,
#             waves_per_block=(1, 1, 1),
#             vector_shapes={M: 1, N: BLOCK_N},
#         )
#     ]
#     constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
#     constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
#     constraints += [tkw.WaveConstraint(M, BLOCK_M)]
#     constraints += [tkw.WaveConstraint(N, BLOCK_N)]

#     @tkw.wave(constraints)
#     def test(
#         a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
#         b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
#         c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
#     ):
#         lhs = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
#         rhs = tkw.read(b, elements_per_thread=ELEMS_PER_THREAD)
#         res = lhs + rhs
#         res2 = tkw.exp2(res)
#         tkw.write(res2, c, elements_per_thread=ELEMS_PER_THREAD)

#     config = {"backend": "rocm", "device": "hip", "target": "gfx942"}

#     shape = (256, 128)
#     a = torch.randn(shape, dtype=torch.float16)
#     b = torch.randn(shape, dtype=torch.float16)
#     c = torch.zeros(shape, dtype=torch.float16)
#     ref = torch.exp2(a + b)
#     # ref = a + b
#     with tk.gen.TestLaunchContext(
#         {
#             M: shape[0],
#             N: shape[1],
#             BLOCK_M: 1,
#             BLOCK_N: 128,
#             ELEMS_PER_THREAD: 2,
#             ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
#         },
#         canonicalize=True,
#         run=True,
#         run_config=config,
#     ):
#         test(a, b, c)
#         assert_allclose(ref, c, atol=0.07)

# def test_reduce():
#     M = tkl.sym.M
#     N = tkl.sym.N
#     BLOCK_M = tkl.sym.BLOCK_M
#     BLOCK_N = tkl.sym.BLOCK_N
#     ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD
#     ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

#     constraints: list[tkw.Constraint] = [
#         tkw.HardwareConstraint(
#             threads_per_wave=64,
#             waves_per_block=(1, 1, 1),
#             vector_shapes={M: 1, N: BLOCK_N},
#         )
#     ]
#     constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
#     constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
#     constraints += [tkw.WaveConstraint(M, BLOCK_M)]
#     constraints += [tkw.WaveConstraint(N, BLOCK_N)]

#     @tkw.wave(constraints)
#     def test(
#         a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
#         b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
#         c: tkl.Memory[M, ADDRESS_SPACE, tkl.f16],
#     ):
#         lhs = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
#         rhs = tkw.read(b, elements_per_thread=ELEMS_PER_THREAD)
#         res = lhs * rhs
#         res = tkw.max(res, dim=N)
#         tkw.write(res, c, elements_per_thread=1)

#     config = {"backend": "rocm", "device": "hip", "target": "gfx942"}

#     shape = (256, 128)
#     a = torch.randn(shape, dtype=torch.float16)
#     b = torch.randn(shape, dtype=torch.float16)
#     c = torch.zeros((shape[0],), dtype=torch.float16)
#     ref = torch.max((a * b),dim=-1)
#     with tk.gen.TestLaunchContext(
#         {
#             M: shape[0],
#             N: shape[1],
#             BLOCK_M: 1,
#             BLOCK_N: 128,
#             ELEMS_PER_THREAD: 2,
#             ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
#         },
#        canonicalize=True,
#        run=True,
#        run_config=config,
#    ):
#        import pdb;pdb.set_trace()
#        test(a, b, c)
#        assert_allclose(ref.values, c, atol=0.07)

def test_partial_reduce():
    M = tkl.sym.M
    N = tkl.sym.N
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 1, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, N, 0)]
    constraints += [tkw.TilingConstraint(N, BLOCK_N)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        c: tkl.Memory[M, ADDRESS_SPACE, tkl.f32],
    ):
        init_max = tkl.Register[M, tkl.f32](0)
        @tkw.reduction(N, init_args=[init_max])
        def repeat(
            partial_max: tkl.Register[M, tkl.f32],
        ) -> tkl.Register[M, tkl.f32]:
            lhs = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
            rhs = tkw.read(b, elements_per_thread=ELEMS_PER_THREAD)
            res = lhs * rhs
            partial_max = tkw.sum(res, partial_max, dim=N)
            return partial_max
        tkw.write(repeat, c, elements_per_thread=1)

    config = {"backend": "rocm", "device": "hip", "target": "gfx942"}

    shape = (256, 1024)
    a = torch.randn(shape, dtype=torch.float32)
    b = torch.randn(shape, dtype=torch.float32)
    c = torch.zeros((shape[0],), dtype=torch.float32)
    ref = torch.sum((a * b),dim=-1)
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            BLOCK_M: 2,
            BLOCK_N: 128,
            ELEMS_PER_THREAD: 2,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run=False,
        run_config=config,
    ):
        with open("partial_reduce_bm_2.mlir", "w") as f:
            f.write(str(test(a, b, c).module_op))
        # test(a, b, c)
        # assert_allclose(ref, c, atol=0.07)

def test_toy_online_softmax():
    M = tkl.sym.M
    N = tkl.sym.N
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 1, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, N, 0)]
    constraints += [tkw.TilingConstraint(N, BLOCK_N)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        c: tkl.Memory[M, ADDRESS_SPACE, tkl.f32],
    ):
        init_sum = tkl.Register[M, tkl.f32](0)
        init_max = tkl.Register[M, tkl.f32](-1e6)
        @tkw.reduction(N, init_args=[init_max, init_sum])
        def repeat(
            partial_max: tkl.Register[M, tkl.f32],
            partial_sum: tkl.Register[M, tkl.f32],
        ) -> tkl.Register[M, tkl.f32]:
            lhs = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
            rhs = tkw.read(b, elements_per_thread=ELEMS_PER_THREAD)
            res = lhs * rhs
            partial_max = tkw.max(res, partial_max, dim=N)
            partial_sum = tkw.sum(res, partial_sum, dim=N)
            return partial_max, partial_sum
        final_max, final_sum = repeat
        output = final_max / final_sum
        tkw.write(output, c, elements_per_thread=1)
        # tkw.write(final_sum, d, elements_per_thread=1)

    config = {"backend": "rocm", "device": "hip", "target": "gfx942"}

    shape = (256, 1024)
    a = torch.randn(shape, dtype=torch.float32)
    b = torch.randn(shape, dtype=torch.float32)
    c = torch.zeros((shape[0],), dtype=torch.float32)
    d = torch.zeros((shape[0],), dtype=torch.float32)
    c_sum = torch.sum((a * b),dim=-1)
    c_max = torch.max((a * b),dim=-1).values
    ref = c_max / c_sum
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            BLOCK_M: 2,
            BLOCK_N: 128,
            ELEMS_PER_THREAD: 2,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run=True,
        run_config=config,
    ):
        with open("softmax_bm_2.mlir", "w") as f:
            f.write(str(test(a, b, c).module_op))
        # test(a, b, c)
        # assert_allclose(ref, c, atol=0.07)

def test_online_softmax():
    M = tkl.sym.M
    N = tkl.sym.N
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 1, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, N, 0)]
    constraints += [tkw.TilingConstraint(N, BLOCK_N)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        c: tkl.Memory[M, ADDRESS_SPACE, tkl.f32],
    ):
        init_sum = tkl.Register[M, tkl.f32](0)
        init_max = tkl.Register[M, tkl.f32](-1e6)
        @tkw.reduction(N, init_args=[init_max, init_sum])
        def repeat(
            partial_max: tkl.Register[M, tkl.f32],
            partial_d: tkl.Register[M, tkl.f32],
        ) -> tkl.Register[M, tkl.f32]:
            src = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
            m_j = tkw.max(src, partial_max, dim=N)
            e_m = tkw.exp2(partial_max - m_j)
            e_x = tkw.exp2(src - m_j)
            e_pd = partial_d * e_m
            d_j =  tkw.sum(e_x, e_pd, dim=N)
            return m_j, d_j
        final_max, final_sum = repeat
        output = final_max / final_sum
        tkw.write(output, c, elements_per_thread=1)
        # tkw.write(final_sum, d, elements_per_thread=1)

    config = {"backend": "rocm", "device": "hip", "target": "gfx942"}

    shape = (256, 1024)
    a = torch.randn(shape, dtype=torch.float32)
    # b = torch.randn(shape, dtype=torch.float32)
    c = torch.zeros((shape[0],), dtype=torch.float32)
    # d = torch.zeros((shape[0],), dtype=torch.float32)
    # c_sum = torch.sum((a * b),dim=-1)
    # c_max = torch.max((a * b),dim=-1).values
    # ref = c_max / c_sum
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            BLOCK_M: 2,
            BLOCK_N: 128,
            ELEMS_PER_THREAD: 2,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run=True,
        run_config=config,
    ):
        # with open("softmax_bm_1.mlir", "w") as f:
        #     f.write(str(test(a, b, c).module_op))
        test(a, c)
        import pdb; pdb.set_trace()
        assert_allclose(ref, c, atol=0.07)

def test_symbolic_range_reduce_max():
    M = tkl.sym.M
    N = tkl.sym.N
    wave_size = 64
    BLOCK_M = 1
    BLOCK_N = tkl.sym.BLOCK_N
    ELEMS_PER_THREAD = BLOCK_N / wave_size
    # BLOCK_N = sympy.ceiling(tkl.sym.N / wave_size) * wave_size
    # ELEMS_PER_THREAD = BLOCK_N // wave_size
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 1, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, N, 0)]
    constraints += [tkw.TilingConstraint(N, BLOCK_N)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, ADDRESS_SPACE, tkl.f16],
    ):
        init_max = tkl.Register[M, tkl.f16](-1e6)

        @tkw.reduction(N, init_args=[init_max])
        def repeat(
            partial_max: tkl.Register[M, tkl.f16],
        ) -> tkl.Register[M, tkl.f16]:
            lhs = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
            rhs = tkw.read(b, elements_per_thread=ELEMS_PER_THREAD)
            res = lhs * rhs
            partial_max = tkw.max(res, partial_max, dim=N)
            return partial_max

        tkw.write(repeat, c, elements_per_thread=1)

    config = {"backend": "rocm", "device": "hip", "target": "gfx942"}

    shape = (256, 256)
    a = torch.randn(shape, dtype=torch.float16)
    b = torch.randn(shape, dtype=torch.float16)
    c = torch.zeros((shape[0],), dtype=torch.float16)
    ref = torch.max((a * b), dim=-1)
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            BLOCK_N: min(128, shape[1]),
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run=False,
        run_config=config,
    ):
        with open("partial_reduce_nos.mlir", "w") as f:
            f.write(str(test(a, b, c).module_op))
        # Assert equal does cast to boolean on torch.Tensor
        # which causes issues, hence we cast to numpy before
        # checking.
        # test(a, b, c)
        # assert_equal(c, ref.values.numpy())

if __name__ == "__main__":
    test_toy_online_softmax()
    # test_partial_reduce()
    # test_tiled_reduce_max()