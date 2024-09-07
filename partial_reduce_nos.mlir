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
        %4 = arith.muli %thread_id_x, %c2 : index
        %5 = arith.divsi %thread_id_x, %c64 : index
        %6 = arith.muli %5, %c128 : index
        %7 = arith.muli %workgroup_id_0, %c256 : index
        %8 = arith.addi %7, %6 : index
        %9 = scf.for %arg3 = %c0 to %c2 step %c1 iter_args(%arg4 = %cst) -> (vector<1xf16>) {
          %13 = arith.muli %arg3, %c128 : index
          %14 = arith.addi %8, %13 : index
          %15 = arith.addi %14, %4 : index
          %16 = vector.load %0[%3, %15] : memref<256x256xf16, strided<[256, 1], offset: ?>>, vector<2xf16>
          %17 = vector.load %1[%3, %15] : memref<256x256xf16, strided<[256, 1], offset: ?>>, vector<2xf16>
          %18 = arith.mulf %16, %17 : vector<2xf16>
          %19 = vector.extract_strided_slice %18 {offsets = [0], sizes = [1], strides = [1]} : vector<2xf16> to vector<1xf16>
          %20 = vector.extract_strided_slice %18 {offsets = [1], sizes = [1], strides = [1]} : vector<2xf16> to vector<1xf16>
          %21 = arith.maximumf %19, %20 : vector<1xf16>
          %22 = vector.extract %21[0] : f16 from vector<1xf16>
          %23 = arith.extf %22 : f16 to f32
          %shuffleResult, %valid = gpu.shuffle  xor %23, %c1_i32, %c64_i32 : f32
          %24 = arith.truncf %shuffleResult : f32 to f16
          %25 = vector.broadcast %24 : f16 to vector<1xf16>
          %26 = arith.maximumf %21, %25 : vector<1xf16>
          %27 = vector.extract %26[0] : f16 from vector<1xf16>
          %28 = arith.extf %27 : f16 to f32
          %shuffleResult_0, %valid_1 = gpu.shuffle  xor %28, %c2_i32, %c64_i32 : f32
          %29 = arith.truncf %shuffleResult_0 : f32 to f16
          %30 = vector.broadcast %29 : f16 to vector<1xf16>
          %31 = arith.maximumf %26, %30 : vector<1xf16>
          %32 = vector.extract %31[0] : f16 from vector<1xf16>
          %33 = arith.extf %32 : f16 to f32
          %shuffleResult_2, %valid_3 = gpu.shuffle  xor %33, %c4_i32, %c64_i32 : f32
          %34 = arith.truncf %shuffleResult_2 : f32 to f16
          %35 = vector.broadcast %34 : f16 to vector<1xf16>
          %36 = arith.maximumf %31, %35 : vector<1xf16>
          %37 = vector.extract %36[0] : f16 from vector<1xf16>
          %38 = arith.extf %37 : f16 to f32
          %shuffleResult_4, %valid_5 = gpu.shuffle  xor %38, %c8_i32, %c64_i32 : f32
          %39 = arith.truncf %shuffleResult_4 : f32 to f16
          %40 = vector.broadcast %39 : f16 to vector<1xf16>
          %41 = arith.maximumf %36, %40 : vector<1xf16>
          %42 = vector.extract %41[0] : f16 from vector<1xf16>
          %43 = arith.extf %42 : f16 to f32
          %shuffleResult_6, %valid_7 = gpu.shuffle  xor %43, %c16_i32, %c64_i32 : f32
          %44 = arith.truncf %shuffleResult_6 : f32 to f16
          %45 = vector.broadcast %44 : f16 to vector<1xf16>
          %46 = arith.maximumf %41, %45 : vector<1xf16>
          %47 = vector.extract %46[0] : f16 from vector<1xf16>
          %48 = arith.extf %47 : f16 to f32
          %shuffleResult_8, %valid_9 = gpu.shuffle  xor %48, %c32_i32, %c64_i32 : f32
          %49 = arith.truncf %shuffleResult_8 : f32 to f16
          %50 = vector.broadcast %49 : f16 to vector<1xf16>
          %51 = arith.maximumf %46, %50 : vector<1xf16>
          %52 = arith.maximumf %arg4, %51 : vector<1xf16>
          scf.yield %52 : vector<1xf16>
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
