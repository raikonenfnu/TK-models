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
        %cst = arith.constant dense<-1.000000e+06> : vector<1xf32>
        %cst_0 = arith.constant dense<0.000000e+00> : vector<1xf32>
        %workgroup_id_0 = stream.dispatch.workgroup.id[0] : index
        %workgroup_id_1 = stream.dispatch.workgroup.id[1] : index
        %thread_id_x = gpu.thread_id  x
        %thread_id_y = gpu.thread_id  y
        %0:4 = scf.for %arg3 = %c0 to %c8 step %c1 iter_args(%arg4 = %cst, %arg5 = %cst_0, %arg6 = %cst, %arg7 = %cst_0) -> (vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>) {
          %8 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<256x1024xf32, strided<[1024, 1], offset: ?>>
          %9 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<256x1024xf32, strided<[1024, 1], offset: ?>>
          %10 = arith.muli %workgroup_id_1, %c2 : index
          %11 = arith.muli %thread_id_y, %c3 : index
          %12 = arith.addi %11, %10 : index
          %13 = arith.muli %thread_id_x, %c2 : index
          %14 = arith.muli %arg3, %c128 : index
          %15 = arith.divsi %thread_id_x, %c64 : index
          %16 = arith.muli %15, %c128 : index
          %17 = arith.muli %workgroup_id_0, %c1024 : index
          %18 = arith.addi %17, %16 : index
          %19 = arith.addi %18, %14 : index
          %20 = arith.addi %19, %13 : index
          %21 = vector.load %8[%12, %20] : memref<256x1024xf32, strided<[1024, 1], offset: ?>>, vector<2xf32>
          %22 = arith.addi %12, %c1 : index
          %23 = vector.load %8[%22, %20] : memref<256x1024xf32, strided<[1024, 1], offset: ?>>, vector<2xf32>
          %24 = vector.load %9[%12, %20] : memref<256x1024xf32, strided<[1024, 1], offset: ?>>, vector<2xf32>
          %25 = vector.load %9[%22, %20] : memref<256x1024xf32, strided<[1024, 1], offset: ?>>, vector<2xf32>
          %26 = arith.mulf %21, %24 : vector<2xf32>
          %27 = arith.mulf %23, %25 : vector<2xf32>
          %28 = vector.extract_strided_slice %26 {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
          %29 = vector.extract_strided_slice %26 {offsets = [1], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
          %30 = arith.maximumf %28, %29 : vector<1xf32>
          %31 = vector.extract %30[0] : f32 from vector<1xf32>
          %shuffleResult, %valid = gpu.shuffle  xor %31, %c1_i32, %c64_i32 : f32
          %32 = vector.broadcast %shuffleResult : f32 to vector<1xf32>
          %33 = arith.maximumf %30, %32 : vector<1xf32>
          %34 = vector.extract %33[0] : f32 from vector<1xf32>
          %shuffleResult_1, %valid_2 = gpu.shuffle  xor %34, %c2_i32, %c64_i32 : f32
          %35 = vector.broadcast %shuffleResult_1 : f32 to vector<1xf32>
          %36 = arith.maximumf %33, %35 : vector<1xf32>
          %37 = vector.extract %36[0] : f32 from vector<1xf32>
          %shuffleResult_3, %valid_4 = gpu.shuffle  xor %37, %c4_i32, %c64_i32 : f32
          %38 = vector.broadcast %shuffleResult_3 : f32 to vector<1xf32>
          %39 = arith.maximumf %36, %38 : vector<1xf32>
          %40 = vector.extract %39[0] : f32 from vector<1xf32>
          %shuffleResult_5, %valid_6 = gpu.shuffle  xor %40, %c8_i32, %c64_i32 : f32
          %41 = vector.broadcast %shuffleResult_5 : f32 to vector<1xf32>
          %42 = arith.maximumf %39, %41 : vector<1xf32>
          %43 = vector.extract %42[0] : f32 from vector<1xf32>
          %shuffleResult_7, %valid_8 = gpu.shuffle  xor %43, %c16_i32, %c64_i32 : f32
          %44 = vector.broadcast %shuffleResult_7 : f32 to vector<1xf32>
          %45 = arith.maximumf %42, %44 : vector<1xf32>
          %46 = vector.extract %45[0] : f32 from vector<1xf32>
          %shuffleResult_9, %valid_10 = gpu.shuffle  xor %46, %c32_i32, %c64_i32 : f32
          %47 = vector.broadcast %shuffleResult_9 : f32 to vector<1xf32>
          %48 = arith.maximumf %45, %47 : vector<1xf32>
          %49 = arith.maximumf %arg4, %48 : vector<1xf32>
          %50 = vector.extract_strided_slice %27 {offsets = [0], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
          %51 = vector.extract_strided_slice %27 {offsets = [1], sizes = [1], strides = [1]} : vector<2xf32> to vector<1xf32>
          %52 = arith.maximumf %50, %51 : vector<1xf32>
          %53 = vector.extract %52[0] : f32 from vector<1xf32>
          %shuffleResult_11, %valid_12 = gpu.shuffle  xor %53, %c1_i32, %c64_i32 : f32
          %54 = vector.broadcast %shuffleResult_11 : f32 to vector<1xf32>
          %55 = arith.maximumf %52, %54 : vector<1xf32>
          %56 = vector.extract %55[0] : f32 from vector<1xf32>
          %shuffleResult_13, %valid_14 = gpu.shuffle  xor %56, %c2_i32, %c64_i32 : f32
          %57 = vector.broadcast %shuffleResult_13 : f32 to vector<1xf32>
          %58 = arith.maximumf %55, %57 : vector<1xf32>
          %59 = vector.extract %58[0] : f32 from vector<1xf32>
          %shuffleResult_15, %valid_16 = gpu.shuffle  xor %59, %c4_i32, %c64_i32 : f32
          %60 = vector.broadcast %shuffleResult_15 : f32 to vector<1xf32>
          %61 = arith.maximumf %58, %60 : vector<1xf32>
          %62 = vector.extract %61[0] : f32 from vector<1xf32>
          %shuffleResult_17, %valid_18 = gpu.shuffle  xor %62, %c8_i32, %c64_i32 : f32
          %63 = vector.broadcast %shuffleResult_17 : f32 to vector<1xf32>
          %64 = arith.maximumf %61, %63 : vector<1xf32>
          %65 = vector.extract %64[0] : f32 from vector<1xf32>
          %shuffleResult_19, %valid_20 = gpu.shuffle  xor %65, %c16_i32, %c64_i32 : f32
          %66 = vector.broadcast %shuffleResult_19 : f32 to vector<1xf32>
          %67 = arith.maximumf %64, %66 : vector<1xf32>
          %68 = vector.extract %67[0] : f32 from vector<1xf32>
          %shuffleResult_21, %valid_22 = gpu.shuffle  xor %68, %c32_i32, %c64_i32 : f32
          %69 = vector.broadcast %shuffleResult_21 : f32 to vector<1xf32>
          %70 = arith.maximumf %67, %69 : vector<1xf32>
          %71 = arith.maximumf %arg6, %70 : vector<1xf32>
          %72 = arith.addf %28, %29 : vector<1xf32>
          %73 = vector.extract %72[0] : f32 from vector<1xf32>
          %shuffleResult_23, %valid_24 = gpu.shuffle  xor %73, %c1_i32, %c64_i32 : f32
          %74 = vector.broadcast %shuffleResult_23 : f32 to vector<1xf32>
          %75 = arith.addf %72, %74 : vector<1xf32>
          %76 = vector.extract %75[0] : f32 from vector<1xf32>
          %shuffleResult_25, %valid_26 = gpu.shuffle  xor %76, %c2_i32, %c64_i32 : f32
          %77 = vector.broadcast %shuffleResult_25 : f32 to vector<1xf32>
          %78 = arith.addf %75, %77 : vector<1xf32>
          %79 = vector.extract %78[0] : f32 from vector<1xf32>
          %shuffleResult_27, %valid_28 = gpu.shuffle  xor %79, %c4_i32, %c64_i32 : f32
          %80 = vector.broadcast %shuffleResult_27 : f32 to vector<1xf32>
          %81 = arith.addf %78, %80 : vector<1xf32>
          %82 = vector.extract %81[0] : f32 from vector<1xf32>
          %shuffleResult_29, %valid_30 = gpu.shuffle  xor %82, %c8_i32, %c64_i32 : f32
          %83 = vector.broadcast %shuffleResult_29 : f32 to vector<1xf32>
          %84 = arith.addf %81, %83 : vector<1xf32>
          %85 = vector.extract %84[0] : f32 from vector<1xf32>
          %shuffleResult_31, %valid_32 = gpu.shuffle  xor %85, %c16_i32, %c64_i32 : f32
          %86 = vector.broadcast %shuffleResult_31 : f32 to vector<1xf32>
          %87 = arith.addf %84, %86 : vector<1xf32>
          %88 = vector.extract %87[0] : f32 from vector<1xf32>
          %shuffleResult_33, %valid_34 = gpu.shuffle  xor %88, %c32_i32, %c64_i32 : f32
          %89 = vector.broadcast %shuffleResult_33 : f32 to vector<1xf32>
          %90 = arith.addf %87, %89 : vector<1xf32>
          %91 = arith.addf %arg5, %90 : vector<1xf32>
          %92 = arith.addf %50, %51 : vector<1xf32>
          %93 = vector.extract %92[0] : f32 from vector<1xf32>
          %shuffleResult_35, %valid_36 = gpu.shuffle  xor %93, %c1_i32, %c64_i32 : f32
          %94 = vector.broadcast %shuffleResult_35 : f32 to vector<1xf32>
          %95 = arith.addf %92, %94 : vector<1xf32>
          %96 = vector.extract %95[0] : f32 from vector<1xf32>
          %shuffleResult_37, %valid_38 = gpu.shuffle  xor %96, %c2_i32, %c64_i32 : f32
          %97 = vector.broadcast %shuffleResult_37 : f32 to vector<1xf32>
          %98 = arith.addf %95, %97 : vector<1xf32>
          %99 = vector.extract %98[0] : f32 from vector<1xf32>
          %shuffleResult_39, %valid_40 = gpu.shuffle  xor %99, %c4_i32, %c64_i32 : f32
          %100 = vector.broadcast %shuffleResult_39 : f32 to vector<1xf32>
          %101 = arith.addf %98, %100 : vector<1xf32>
          %102 = vector.extract %101[0] : f32 from vector<1xf32>
          %shuffleResult_41, %valid_42 = gpu.shuffle  xor %102, %c8_i32, %c64_i32 : f32
          %103 = vector.broadcast %shuffleResult_41 : f32 to vector<1xf32>
          %104 = arith.addf %101, %103 : vector<1xf32>
          %105 = vector.extract %104[0] : f32 from vector<1xf32>
          %shuffleResult_43, %valid_44 = gpu.shuffle  xor %105, %c16_i32, %c64_i32 : f32
          %106 = vector.broadcast %shuffleResult_43 : f32 to vector<1xf32>
          %107 = arith.addf %104, %106 : vector<1xf32>
          %108 = vector.extract %107[0] : f32 from vector<1xf32>
          %shuffleResult_45, %valid_46 = gpu.shuffle  xor %108, %c32_i32, %c64_i32 : f32
          %109 = vector.broadcast %shuffleResult_45 : f32 to vector<1xf32>
          %110 = arith.addf %107, %109 : vector<1xf32>
          %111 = arith.addf %arg7, %110 : vector<1xf32>
          scf.yield %49, %91, %71, %111 : vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>
        }
        %1 = arith.divf %0#0, %0#1 : vector<1xf32>
        %2 = arith.divf %0#2, %0#3 : vector<1xf32>
        %3 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<256xf32, strided<[1], offset: ?>>
        %4 = arith.muli %workgroup_id_1, %c2 : index
        %5 = arith.muli %thread_id_y, %c3 : index
        %6 = arith.addi %5, %4 : index
        vector.store %1, %3[%6] : memref<256xf32, strided<[1], offset: ?>>, vector<1xf32>
        %7 = arith.addi %6, %c1 : index
        vector.store %2, %3[%7] : memref<256xf32, strided<[1], offset: ?>>, vector<1xf32>
        return
      }
    }
  }
}
