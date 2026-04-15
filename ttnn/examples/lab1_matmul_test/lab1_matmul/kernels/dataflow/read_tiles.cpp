// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args
    int arg_idx = 0;
    uint32_t in0_base_addr = get_arg_val<uint32_t>(arg_idx++);  // A base address
    uint32_t in1_base_addr = get_arg_val<uint32_t>(arg_idx++);  // B base address
    uint32_t Mt = get_arg_val<uint32_t>(arg_idx++);             // tile rows in A and C
    uint32_t Kt = get_arg_val<uint32_t>(arg_idx++);             // tile cols in A, tile rows in B
    uint32_t Nt = get_arg_val<uint32_t>(arg_idx++);             // tile cols in B and C

    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;  // A tiles
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;  // B tiles

    constexpr uint32_t tile_size_bytes = get_tile_size(cb_in0);

    constexpr auto in0_layout_args = TensorAccessorArgs<0>();
    const auto in0_addr_gen = TensorAccessor(in0_layout_args, in0_base_addr, tile_size_bytes);

    constexpr auto in1_layout_args = TensorAccessorArgs<in0_layout_args.next_compile_time_args_offset()>();
    const auto in1_addr_gen = TensorAccessor(in1_layout_args, in1_base_addr, tile_size_bytes);

    // We need to feed the compute kernel one A tile and one B tile at a time,
    // in the correct order for tiled matrix multiplication.
    //
    // Recall from the tiled matmul algorithm:
    //   Each output tile C[mt, nt] = sum over kt of: A[mt, kt] * B[kt, nt]
    //
    // The compute kernel processes output tiles in row-major order (mt outer, nt inner),
    // and for each output tile accumulates over all kt steps.
    // So this reader must supply tiles in the same order:
    //   for each mt (output tile row)
    //     for each nt (output tile col)
    //       for each kt (inner dimension step)
    //         send A tile at row mt, col kt
    //         send B tile at row kt, col nt
    //
    // Tiles in A are stored in row-major tiled order, so the flat tile index for A[mt, kt] is:
    //   a_tile_idx = mt * Kt + kt
    //
    // Tiles in B are stored in row-major tiled order, so the flat tile index for B[kt, nt] is:
    //   b_tile_idx = kt * Nt + nt

    for (uint32_t mt = 0; mt < Mt; mt++) {
        for (uint32_t nt = 0; nt < Nt; nt++) {
            for (uint32_t kt = 0; kt < Kt; kt++) {
                cb_reserve_back(cb_in0, 1);
                cb_reserve_back(cb_in1, 1);

                uint32_t cb_in0_addr = get_write_ptr(cb_in0);
                uint32_t cb_in1_addr = get_write_ptr(cb_in1);

                // TODO: Compute the flat tile index for A[mt, kt].
                // Hint flat index is fundtion of variables: mt, Kt, kt
                uint32_t a_tile_idx = 0;  // <-- replace with correct expression

                // TODO: Compute the flat tile index for B[kt, nt].
                // Hint flat index is function of variables: kt, Nt, nt
                uint32_t b_tile_idx = 0;  // <-- replace with correct expression

                noc_async_read_tile(a_tile_idx, in0_addr_gen, cb_in0_addr);
                noc_async_read_tile(b_tile_idx, in1_addr_gen, cb_in1_addr);

                noc_async_read_barrier();

                cb_push_back(cb_in0, 1);
                cb_push_back(cb_in1, 1);
            }
        }
    }
}
