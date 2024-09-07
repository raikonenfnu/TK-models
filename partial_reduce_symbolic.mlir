#translation = #iree_codegen.translation_info<None workgroup_size = [64, 1, 1] subgroup_size = 64>
module attributes {transform.with_named_sequence} {
  stream.executable private @test {
    stream.executable.export public @test workgroups() -> (index, index, index) {
      %c1 = arith.constant 1 : index
      %c256 = arith.constant 256 : index
      stream.return %c1, %c256, %c1 : index, index, index
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
        %c256 = arith.constant 256 : index
        %c64 = arith.constant 64 : index
        %c128 = arith.constant 128 : index
        %c4 = arith.constant 4 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant dense<0xFC00> : vector<1xf16>
        %workgroup_id_0 = stream.dispatch.workgroup.id[0] : index
        %workgroup_id_1 = stream.dispatch.workgroup.id[1] : index
        %thread_id_x = gpu.thread_id  x
        %thread_id_y = gpu.thread_id  y
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<256x256xf16, strided<[256, 1], offset: ?>>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<256x256xf16, strided<[256, 1], offset: ?>>
        %2 = arith.muli %thread_id_y, %c2 : index
        %3 = arith.addi %2, %workgroup_id_1 : index
        %4 = arith.muli %thread_id_x, %c4 : index
        %5 = arith.divsi %thread_id_x, %c64 : index
        %6 = arith.muli %5, %c128 : index
        %7 = arith.muli %workgroup_id_0, %c256 : index
        %8 = arith.addi %7, %6 : index
        %9 = scf.for %arg3 = %c0 to %c2 step %c1 iter_args(%arg4 = %cst) -> (vector<1xf16>) {
          %13 = arith.muli %arg3, %c128 : index
          %14 = arith.addi %8, %13 : index
          %15 = arith.addi %14, %4 : index
          %16 = vector.load %0[%3, %15] : memref<256x256xf16, strided<[256, 1], offset: ?>>, vector<4xf16>
          %17 = arith.addi %15, %c128 : index
          %18 = vector.load %0[%3, %17] : memref<256x256xf16, strided<[256, 1], offset: ?>>, vector<4xf16>
          %19 = vector.load %1[%3, %15] : memref<256x256xf16, strided<[256, 1], offset: ?>>, vector<4xf16>
          %20 = vector.load %1[%3, %17] : memref<256x256xf16, strided<[256, 1], offset: ?>>, vector<4xf16>
          %21 = arith.mulf %16, %19 : vector<4xf16>
          %22 = arith.mulf %18, %20 : vector<4xf16>
          %23 = vector.extract_strided_slice %21 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %24 = vector.extract_strided_slice %21 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %25 = arith.maximumf %23, %24 : vector<1xf16>
          %26 = vector.extract %25[0] : f16 from vector<1xf16>
          %27 = arith.extf %26 : f16 to f32
          %shuffleResult, %valid = gpu.shuffle  xor %27, %c1_i32, %c64_i32 : f32
          %28 = arith.truncf %shuffleResult : f32 to f16
          %29 = vector.broadcast %28 : f16 to vector<1xf16>
          %30 = arith.maximumf %25, %29 : vector<1xf16>
          %31 = vector.extract %30[0] : f16 from vector<1xf16>
          %32 = arith.extf %31 : f16 to f32
          %shuffleResult_0, %valid_1 = gpu.shuffle  xor %32, %c2_i32, %c64_i32 : f32
          %33 = arith.truncf %shuffleResult_0 : f32 to f16
          %34 = vector.broadcast %33 : f16 to vector<1xf16>
          %35 = arith.maximumf %30, %34 : vector<1xf16>
          %36 = vector.extract %35[0] : f16 from vector<1xf16>
          %37 = arith.extf %36 : f16 to f32
          %shuffleResult_2, %valid_3 = gpu.shuffle  xor %37, %c4_i32, %c64_i32 : f32
          %38 = arith.truncf %shuffleResult_2 : f32 to f16
          %39 = vector.broadcast %38 : f16 to vector<1xf16>
          %40 = arith.maximumf %35, %39 : vector<1xf16>
          %41 = vector.extract %40[0] : f16 from vector<1xf16>
          %42 = arith.extf %41 : f16 to f32
          %shuffleResult_4, %valid_5 = gpu.shuffle  xor %42, %c8_i32, %c64_i32 : f32
          %43 = arith.truncf %shuffleResult_4 : f32 to f16
          %44 = vector.broadcast %43 : f16 to vector<1xf16>
          %45 = arith.maximumf %40, %44 : vector<1xf16>
          %46 = vector.extract %45[0] : f16 from vector<1xf16>
          %47 = arith.extf %46 : f16 to f32
          %shuffleResult_6, %valid_7 = gpu.shuffle  xor %47, %c16_i32, %c64_i32 : f32
          %48 = arith.truncf %shuffleResult_6 : f32 to f16
          %49 = vector.broadcast %48 : f16 to vector<1xf16>
          %50 = arith.maximumf %45, %49 : vector<1xf16>
          %51 = vector.extract %50[0] : f16 from vector<1xf16>
          %52 = arith.extf %51 : f16 to f32
          %shuffleResult_8, %valid_9 = gpu.shuffle  xor %52, %c32_i32, %c64_i32 : f32
          %53 = arith.truncf %shuffleResult_8 : f32 to f16
          %54 = vector.broadcast %53 : f16 to vector<1xf16>
          %55 = arith.maximumf %50, %54 : vector<1xf16>
          %56 = arith.maximumf %arg4, %55 : vector<1xf16>
          %57 = vector.extract_strided_slice %22 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %58 = vector.extract_strided_slice %22 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf16> to vector<1xf16>
          %59 = arith.maximumf %57, %58 : vector<1xf16>
          %60 = vector.extract %59[0] : f16 from vector<1xf16>
          %61 = arith.extf %60 : f16 to f32
          %shuffleResult_10, %valid_11 = gpu.shuffle  xor %61, %c1_i32, %c64_i32 : f32
          %62 = arith.truncf %shuffleResult_10 : f32 to f16
          %63 = vector.broadcast %62 : f16 to vector<1xf16>
          %64 = arith.maximumf %59, %63 : vector<1xf16>
          %65 = vector.extract %64[0] : f16 from vector<1xf16>
          %66 = arith.extf %65 : f16 to f32
          %shuffleResult_12, %valid_13 = gpu.shuffle  xor %66, %c2_i32, %c64_i32 : f32
          %67 = arith.truncf %shuffleResult_12 : f32 to f16
          %68 = vector.broadcast %67 : f16 to vector<1xf16>
          %69 = arith.maximumf %64, %68 : vector<1xf16>
          %70 = vector.extract %69[0] : f16 from vector<1xf16>
          %71 = arith.extf %70 : f16 to f32
          %shuffleResult_14, %valid_15 = gpu.shuffle  xor %71, %c4_i32, %c64_i32 : f32
          %72 = arith.truncf %shuffleResult_14 : f32 to f16
          %73 = vector.broadcast %72 : f16 to vector<1xf16>
          %74 = arith.maximumf %69, %73 : vector<1xf16>
          %75 = vector.extract %74[0] : f16 from vector<1xf16>
          %76 = arith.extf %75 : f16 to f32
          %shuffleResult_16, %valid_17 = gpu.shuffle  xor %76, %c8_i32, %c64_i32 : f32
          %77 = arith.truncf %shuffleResult_16 : f32 to f16
          %78 = vector.broadcast %77 : f16 to vector<1xf16>
          %79 = arith.maximumf %74, %78 : vector<1xf16>
          %80 = vector.extract %79[0] : f16 from vector<1xf16>
          %81 = arith.extf %80 : f16 to f32
          %shuffleResult_18, %valid_19 = gpu.shuffle  xor %81, %c16_i32, %c64_i32 : f32
          %82 = arith.truncf %shuffleResult_18 : f32 to f16
          %83 = vector.broadcast %82 : f16 to vector<1xf16>
          %84 = arith.maximumf %79, %83 : vector<1xf16>
          %85 = vector.extract %84[0] : f16 from vector<1xf16>
          %86 = arith.extf %85 : f16 to f32
          %shuffleResult_20, %valid_21 = gpu.shuffle  xor %86, %c32_i32, %c64_i32 : f32
          %87 = arith.truncf %shuffleResult_20 : f32 to f16
          %88 = vector.broadcast %87 : f16 to vector<1xf16>
          %89 = arith.maximumf %84, %88 : vector<1xf16>
          %90 = arith.maximumf %56, %89 : vector<1xf16>
          scf.yield %90 : vector<1xf16>
        }
        %10 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<256xf16, strided<[1], offset: ?>>
        %11 = arith.muli %thread_id_y, %c2 : index
        %12 = arith.addi %11, %workgroup_id_1 : index
        vector.store %9, %10[%12] : memref<256xf16, strided<[1], offset: ?>>, vector<1xf16>
        return
      }
    }
  }
}
