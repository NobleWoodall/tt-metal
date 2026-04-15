// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    // Compile-time tile dimension counts (set by host, enables compiler optimizations)
    constexpr uint32_t Mt = get_compile_time_arg_val(0);  // tile rows in A and C
    constexpr uint32_t Kt = get_compile_time_arg_val(1);  // inner tile dimension
    constexpr uint32_t Nt = get_compile_time_arg_val(2);  // tile cols in B and C

    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;    // A tiles from reader
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;    // B tiles from reader
    constexpr tt::CBIndex cb_out0 = tt::CBIndex::c_16;  // C output tiles to writer

    constexpr uint32_t dst_reg_idx = 0;

    // Initialize the Tensix matrix engine for matrix multiplication.
    // Note: do NOT use binary_op_init_common here - that is only for elementwise ops.
    mm_init(cb_in0, cb_in1, cb_out0);

    // Process output tiles in row-major order, matching the reader's ordering.
    // For each output tile C[mt, nt], accumulate A[mt, kt] * B[kt, nt] over all kt.
    for (uint32_t mt = 0; mt < Mt; mt++) {
        for (uint32_t nt = 0; nt < Nt; nt++) {
            // TODO: Acquire the destination register.
            // This must be called once per OUTPUT tile (here, before the kt loop),
            // not once per kt step. Acquiring zeroes out the register, so the
            // accumulation across all kt steps starts from zero.
            // Call: tile_regs_acquire();

            for (uint32_t kt = 0; kt < Kt; kt++) {
                // Wait for one A tile and one B tile to arrive from the reader.
                cb_wait_front(cb_in0, 1);
                cb_wait_front(cb_in1, 1);

                // Multiply A tile (index 0 in cb_in0) by B tile (index 0 in cb_in1)
                // and accumulate the result into dst_reg_idx.
                // matmul_tiles ADDS to whatever is already in the destination register,
                // which is why tile_regs_acquire() must be called before this loop.
                matmul_tiles(cb_in0, cb_in1, 0, 0, dst_reg_idx);

                // Free the consumed input tiles.
                cb_pop_front(cb_in0, 1);
                cb_pop_front(cb_in1, 1);
            }

            // TODO: Commit the destination register to signal the compute processor
            // is done writing. Call: tile_regs_commit();

            // Packer processor: wait for destination register to be ready,
            // then write the result tile to the output circular buffer.
            tile_regs_wait();
            cb_reserve_back(cb_out0, 1);
            pack_tile(dst_reg_idx, cb_out0);
            cb_push_back(cb_out0, 1);

            // TODO: Release the destination register so it can be reused for
            // the next output tile. Call: tile_regs_release();
        }
    }
}
