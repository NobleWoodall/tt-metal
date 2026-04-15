// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t out0_base_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles = get_arg_val<uint32_t>(1);

    constexpr tt::CBIndex cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t tile_size_bytes = get_tile_size(cb_out0);

    constexpr auto out0_layout_args = TensorAccessorArgs<0>();
    const auto out0_addr_gen = TensorAccessor(out0_layout_args, out0_base_addr, tile_size_bytes);

    // Write C tiles in row-major order (indices 0..n_tiles_C-1).
    // This matches the reader's output tile ordering (mt outer, nt inner).
    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_wait_front(cb_out0, 1);
        uint32_t cb_out0_addr = get_read_ptr(cb_out0);
        noc_async_write_tile(i, out0_addr_gen, cb_out0_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_out0, 1);
    }
}
