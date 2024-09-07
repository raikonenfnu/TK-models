#translation = #iree_codegen.translation_info<None workgroup_size = [64, 1, 1] subgroup_size = 64>
module attributes {transform.with_named_sequence} {
  stream.executable private @test {
    stream.executable.export public @test workgroups() -> (index, index, index) {
      %c1 = arith.constant 1 : index
      %c128 = arith.constant 128 : index
      stream.return %c1, %c128, %c1 : index, index, index
    }
    builtin.module {
      func.func @test(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) attributes {translation_info = #translation} {
        %c32_i32 = arith.constant 32 : i32
        %c16_i32 = arith.constant 16 : i32
        %c8_i32 = arith.constant 8 : i32
        %c4_i32 = arith.constant 4 : i32
        %c2_i32 = arith.constant 2 : i32
        %c64_i32 = arith.constant 64 : i32
        %c1_i32 = arith.constant 1 : i32
        %c1024 = arith.constant 1024 : index
        %c64 = arith.constant 64 : index
        %c128 = arith.constant 128 : index
        %c3 = arith.constant 3 : index
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        %c8 = arith.constant 8 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
        %workgroup_id_0 = stream.dispatch.workgroup.id[0] : index
        %workgroup_id_1 = stream.dispatch.workgroup.id[1] : index
        %thread_id_x = gpu.thread_id  x
        %thread_id_y = gpu.thread_id  y
        %0:2 = scf.for %arg3 = %c0 to %c8 step %c1 iter_args(%arg4 = %cst, %arg5 = %cst) -> (vector<1xf32>, vector<1xf32>) {
          %6 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<256x1024xf32, strided<[1024, 1], offset: ?>>
          %7 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<256x1024xf32, strided<[1024, 1], offset: ?>>
          %8 = arith.muli %workgroup_id_1, %c2 : index
          %9 = arith.muli %thread_id_y, %c3 : index
          %10 = arith.addi %9, %8 : index
          %11 = arith.muli %thread_id_x, %c2 : index
          %12 = arith.muli %arg3, %c128 : index
          %13 = arith.divsi %thread_id_x, %c64 : index
          %14 = arith.muli %13, %c128 : index
          %15 = arith.muli %workgroup_id_0, %c1024 : index
          %16 = arith.addi %15, %14 : index
          %17 = arith.addi %16, %12 : index
          %18 = arith.addi %17, %11 : index
          %19 = vector.load %6[%10, %18] : memref<256x1024xf32, strided<[1024, 1], offset: ?>>, vector<2xf32>
          %20 = arith.addi %10, %c1 : index
          %21 = vector.load %6[%20, %18] : memref<256x1024xf32, strided<[1024, 1], offset: ?>>, vector<2xf32>
          %22 = vector.load %7[%10, %18] : memref<256x1024xf32, strided<[1024, 1], offset: ?>>, vector<2xf32>
          %23 = vector.load %7[%20, %18] : memref<256x1024xf32, strided<[1024, 1], offset: ?>>, vector<2xf32>
          %24 = arith.mulf %19, %22 : vector<2xf32>
          %25 = arith.mulf %21, %23 : vector<2xf32>
          %26 = vector.extract_strided_slice %24 {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
          %27 = vector.extract_strided_slice %24 {offsets = [1], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
          %28 = arith.addf %26, %27 : vector<1xf32>
          %29 = vector.extract %28[0] : f32 from vector<1xf32>
          %shuffleResult, %valid = gpu.shuffle  xor %29, %c1_i32, %c64_i32 : f32
          %30 = vector.broadcast %shuffleResult : f32 to vector<1xf32>
          %31 = arith.addf %28, %30 : vector<1xf32>
          %32 = vector.extract %31[0] : f32 from vector<1xf32>
          %shuffleResult_0, %valid_1 = gpu.shuffle  xor %32, %c2_i32, %c64_i32 : f32
          %33 = vector.broadcast %shuffleResult_0 : f32 to vector<1xf32>
          %34 = arith.addf %31, %33 : vector<1xf32>
          %35 = vector.extract %34[0] : f32 from vector<1xf32>
          %shuffleResult_2, %valid_3 = gpu.shuffle  xor %35, %c4_i32, %c64_i32 : f32
          %36 = vector.broadcast %shuffleResult_2 : f32 to vector<1xf32>
          %37 = arith.addf %34, %36 : vector<1xf32>
          %38 = vector.extract %37[0] : f32 from vector<1xf32>
          %shuffleResult_4, %valid_5 = gpu.shuffle  xor %38, %c8_i32, %c64_i32 : f32
          %39 = vector.broadcast %shuffleResult_4 : f32 to vector<1xf32>
          %40 = arith.addf %37, %39 : vector<1xf32>
          %41 = vector.extract %40[0] : f32 from vector<1xf32>
          %shuffleResult_6, %valid_7 = gpu.shuffle  xor %41, %c16_i32, %c64_i32 : f32
          %42 = vector.broadcast %shuffleResult_6 : f32 to vector<1xf32>
          %43 = arith.addf %40, %42 : vector<1xf32>
          %44 = vector.extract %43[0] : f32 from vector<1xf32>
          %shuffleResult_8, %valid_9 = gpu.shuffle  xor %44, %c32_i32, %c64_i32 : f32
          %45 = vector.broadcast %shuffleResult_8 : f32 to vector<1xf32>
          %46 = arith.addf %43, %45 : vector<1xf32>
          %47 = arith.addf %arg4, %46 : vector<1xf32>
          %48 = vector.extract_strided_slice %25 {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
          %49 = vector.extract_strided_slice %25 {offsets = [1], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
          %50 = arith.addf %48, %49 : vector<1xf32>
          %51 = vector.extract %50[0] : f32 from vector<1xf32>
          %shuffleResult_10, %valid_11 = gpu.shuffle  xor %51, %c1_i32, %c64_i32 : f32
          %52 = vector.broadcast %shuffleResult_10 : f32 to vector<1xf32>
          %53 = arith.addf %50, %52 : vector<1xf32>
          %54 = vector.extract %53[0] : f32 from vector<1xf32>
          %shuffleResult_12, %valid_13 = gpu.shuffle  xor %54, %c2_i32, %c64_i32 : f32
          %55 = vector.broadcast %shuffleResult_12 : f32 to vector<1xf32>
          %56 = arith.addf %53, %55 : vector<1xf32>
          %57 = vector.extract %56[0] : f32 from vector<1xf32>
          %shuffleResult_14, %valid_15 = gpu.shuffle  xor %57, %c4_i32, %c64_i32 : f32
          %58 = vector.broadcast %shuffleResult_14 : f32 to vector<1xf32>
          %59 = arith.addf %56, %58 : vector<1xf32>
          %60 = vector.extract %59[0] : f32 from vector<1xf32>
          %shuffleResult_16, %valid_17 = gpu.shuffle  xor %60, %c8_i32, %c64_i32 : f32
          %61 = vector.broadcast %shuffleResult_16 : f32 to vector<1xf32>
          %62 = arith.addf %59, %61 : vector<1xf32>
          %63 = vector.extract %62[0] : f32 from vector<1xf32>
          %shuffleResult_18, %valid_19 = gpu.shuffle  xor %63, %c16_i32, %c64_i32 : f32
          %64 = vector.broadcast %shuffleResult_18 : f32 to vector<1xf32>
          %65 = arith.addf %62, %64 : vector<1xf32>
          %66 = vector.extract %65[0] : f32 from vector<1xf32>
          %shuffleResult_20, %valid_21 = gpu.shuffle  xor %66, %c32_i32, %c64_i32 : f32
          %67 = vector.broadcast %shuffleResult_20 : f32 to vector<1xf32>
          %68 = arith.addf %65, %67 : vector<1xf32>
          %69 = arith.addf %arg5, %68 : vector<1xf32>
          scf.yield %47, %69 : vector<1xf32>, vector<1xf32>
        }
        %1 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<256xf32, strided<[1], offset: ?>>
        %2 = arith.muli %workgroup_id_1, %c2 : index
        %3 = arith.muli %thread_id_y, %c3 : index
        %4 = arith.addi %3, %2 : index
        vector.store %0#0, %1[%4] : memref<256xf32, strided<[1], offset: ?>>, vector<1xf32>
        %5 = arith.addi %4, %c1 : index
        vector.store %0#1, %1[%5] : memref<256xf32, strided<[1], offset: ?>>, vector<1xf32>
        return
      }
    }
  }
}
