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
        %c128 = arith.constant 128 : index
        %c2 = arith.constant 2 : index
        %c0 = arith.constant 0 : index
        %workgroup_id_0 = stream.dispatch.workgroup.id[0] : index
        %workgroup_id_1 = stream.dispatch.workgroup.id[1] : index
        %thread_id_x = gpu.thread_id  x
        %thread_id_y = gpu.thread_id  y
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<256x128xf16, strided<[128, 1], offset: ?>>
        %1 = arith.addi %workgroup_id_1, %thread_id_y : index
        %2 = arith.muli %thread_id_x, %c2 : index
        %3 = arith.muli %workgroup_id_0, %c128 : index
        %4 = arith.addi %3, %2 : index
        %5 = vector.load %0[%1, %4] : memref<256x128xf16, strided<[128, 1], offset: ?>>, vector<2xf16>
        %6 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<256x128xf16, strided<[128, 1], offset: ?>>
        %7 = vector.load %6[%1, %4] : memref<256x128xf16, strided<[128, 1], offset: ?>>, vector<2xf16>
        %8 = arith.addf %5, %7 : vector<2xf16>
        %9 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<256x128xf16, strided<[128, 1], offset: ?>>
        vector.store %8, %9[%1, %4] : memref<256x128xf16, strided<[128, 1], offset: ?>>, vector<2xf16>
        return
      }
    }
  }
}
