// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt-metalium/core_coord.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;
using namespace ttnn;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

// Reference matrix multiplication on host CPU for verification.
// Accumulates into float32 to minimize precision loss, then casts to bfloat16.
void reference_matmul(
    const std::vector<bfloat16>& A,
    const std::vector<bfloat16>& B,
    std::vector<bfloat16>& C,
    const uint32_t M,
    const uint32_t K,
    const uint32_t N) {
    TT_FATAL(A.size() == M * K, "A must have size M * K");
    TT_FATAL(B.size() == K * N, "B must have size K * N");
    TT_FATAL(C.size() == M * N, "C must have size M * N");

    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; k++) {
                acc += static_cast<float>(A[i * K + k]) * static_cast<float>(B[k * N + j]);
            }
            C[i * N + j] = static_cast<bfloat16>(acc);
        }
    }
}

struct ProgramState {
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device;
    Program program;
    tt::tt_metal::CoreCoord core;
    tt::tt_metal::distributed::MeshWorkload workload;
    tt::tt_metal::distributed::MeshCoordinateRange device_range;
    tt::tt_metal::distributed::MeshCommandQueue& cq;

    ProgramState(
        std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device,
        Program program,
        tt::tt_metal::CoreCoord core,
        tt::tt_metal::distributed::MeshWorkload workload,
        tt::tt_metal::distributed::MeshCoordinateRange device_range,
        tt::tt_metal::distributed::MeshCommandQueue& cq) :
        mesh_device(std::move(mesh_device)),
        program(std::move(program)),
        core(core),
        workload(std::move(workload)),
        device_range(std::move(device_range)),
        cq(cq) {}
};

ProgramState init_program() {
    constexpr int device_id = 0;
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device =
        tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
    tt::tt_metal::distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    tt::tt_metal::distributed::MeshWorkload workload;
    tt::tt_metal::distributed::MeshCoordinateRange device_range =
        tt::tt_metal::distributed::MeshCoordinateRange(mesh_device->shape());
    tt::tt_metal::CoreCoord core({0, 0});
    Program program = CreateProgram();

    return ProgramState(
        std::move(mesh_device), std::move(program), core, std::move(workload), std::move(device_range), cq);
}

void create_cb(Program& program, const tt::tt_metal::CoreCoord& core, uint32_t num_tiles, tt::CBIndex cb_index) {
    constexpr uint32_t single_tile_bytes = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
    constexpr tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;

    CircularBufferConfig cb_config = CircularBufferConfig(num_tiles * single_tile_bytes, {{cb_index, cb_data_format}})
                                         .set_page_size(cb_index, single_tile_bytes);
    tt_metal::CreateCircularBuffer(program, core, cb_config);
}

void matmul_tensix(
    const std::vector<bfloat16>& a,
    const std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    const uint32_t M,
    const uint32_t K,
    const uint32_t N,
    ProgramState& prog_state) {
    // Validate tile alignment
    TT_FATAL(TILE_HEIGHT == TILE_WIDTH, "Tile must be square");
    TT_FATAL(M % TILE_HEIGHT == 0, "M must be divisible by TILE_HEIGHT");
    TT_FATAL(K % TILE_HEIGHT == 0, "K must be divisible by TILE_HEIGHT");
    TT_FATAL(N % TILE_WIDTH == 0, "N must be divisible by TILE_WIDTH");

    TT_FATAL(a.size() == M * K, "A must have size M * K");
    TT_FATAL(b.size() == K * N, "B must have size K * N");
    TT_FATAL(output.size() == M * N, "Output must have size M * N");

    // Tile counts per dimension
    const uint32_t Mt = M / TILE_HEIGHT;  // number of tile rows in A / C
    const uint32_t Kt = K / TILE_HEIGHT;  // number of tile cols in A / tile rows in B
    const uint32_t Nt = N / TILE_WIDTH;   // number of tile cols in B / C

    // const uint32_t n_tiles_A = Mt * Kt;
    // const uint32_t n_tiles_B = Kt * Nt;
    const uint32_t n_tiles_C = Mt * Nt;

    // Create device tensors
    TensorLayout tile_layout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig(BufferType::DRAM));
    TensorSpec spec_A(Shape({M, K}), tile_layout);
    TensorSpec spec_B(Shape({K, N}), tile_layout);
    TensorSpec spec_C(Shape({M, N}), tile_layout);

    Tensor src0_tensor = Tensor::from_vector<bfloat16>(a, spec_A, prog_state.mesh_device.get());
    Tensor src1_tensor = Tensor::from_vector<bfloat16>(b, spec_B, prog_state.mesh_device.get());
    Tensor dst_tensor = create_device_tensor(spec_C, prog_state.mesh_device.get());

    // Circular buffers: 2 tiles each for double buffering
    create_cb(prog_state.program, prog_state.core, 2, CBIndex::c_0);   // A tiles
    create_cb(prog_state.program, prog_state.core, 2, CBIndex::c_1);   // B tiles
    create_cb(prog_state.program, prog_state.core, 2, CBIndex::c_16);  // C output tiles

    const auto& src0_mesh_buffer = src0_tensor.mesh_buffer();
    const auto& src1_mesh_buffer = src1_tensor.mesh_buffer();
    const auto& dst_mesh_buffer = dst_tensor.mesh_buffer();

    // Reader compile-time args: tensor layout info for A and B
    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(src0_mesh_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(src1_mesh_buffer).append_to(reader_compile_time_args);

    KernelHandle reader_id = tt_metal::CreateKernel(
        prog_state.program,
        OVERRIDE_KERNEL_PREFIX "ttnn/examples/lab1_matmul/kernels/dataflow/read_tiles.cpp",
        prog_state.core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(dst_mesh_buffer).append_to(writer_compile_time_args);
    KernelHandle writer_id = tt_metal::CreateKernel(
        prog_state.program,
        OVERRIDE_KERNEL_PREFIX "ttnn/examples/lab1_matmul/kernels/dataflow/write_tiles.cpp",
        prog_state.core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = writer_compile_time_args});

    // Compute compile-time args: tile counts needed for loop bounds
    std::vector<uint32_t> compute_compile_time_args = {Mt, Kt, Nt};
    tt_metal::CreateKernel(
        prog_state.program,
        OVERRIDE_KERNEL_PREFIX "ttnn/examples/lab1_matmul/kernels/compute/tiles_matmul.cpp",
        prog_state.core,
        tt_metal::ComputeConfig{.compile_args = compute_compile_time_args});

    // Runtime args
    uint32_t src0_addr = src0_mesh_buffer.address();
    uint32_t src1_addr = src1_mesh_buffer.address();
    uint32_t dst_addr = dst_mesh_buffer.address();

    // Reader needs: base addresses + tile dimension counts to generate indices
    tt_metal::SetRuntimeArgs(prog_state.program, reader_id, prog_state.core, {src0_addr, src1_addr, Mt, Kt, Nt});
    // Writer needs: base address + total output tiles
    tt_metal::SetRuntimeArgs(prog_state.program, writer_id, prog_state.core, {dst_addr, n_tiles_C});

    prog_state.workload.add_program(prog_state.device_range, std::move(prog_state.program));
    tt_metal::distributed::EnqueueMeshWorkload(prog_state.cq, prog_state.workload, true);

    output = dst_tensor.to_vector<bfloat16>();
}

int main() {
    bool pass = true;

    try {
        constexpr uint32_t M = 640;
        constexpr uint32_t K = 320;
        constexpr uint32_t N = 640;

        constexpr uint32_t rng_seed = 42;
        std::mt19937 rng(rng_seed);
        std::uniform_real_distribution<float> rng_dist(0.f, 1.0f);

        std::vector<bfloat16> src0_vec(M * K);
        for (bfloat16& v : src0_vec) {
            v = static_cast<bfloat16>(rng_dist(rng));
        }

        std::vector<bfloat16> src1_vec(K * N);
        for (bfloat16& v : src1_vec) {
            v = static_cast<bfloat16>(rng_dist(rng));
        }

        // Reference matmul on host
        std::vector<bfloat16> reference_result(M * N);
        reference_matmul(src0_vec, src1_vec, reference_result, M, K, N);

        // Tensix matmul
        std::vector<bfloat16> result_vec(M * N);
        ProgramState prog_state = init_program();
        matmul_tensix(src0_vec, src1_vec, result_vec, M, K, N, prog_state);

        log_info(tt::LogAlways, "Output vector of size {}", result_vec.size());

        // Validate
        TT_FATAL(result_vec.size() == reference_result.size(), "Result vector size mismatch");
        constexpr float RELTOL = 0.05f;  // slightly looser than eltwise add due to K=320 accumulation
        for (size_t i = 0; i < result_vec.size(); ++i) {
            const float expected = static_cast<float>(reference_result[i]);
            const float actual = static_cast<float>(result_vec[i]);
            float relative_error =
                (expected == 0.0f) ? std::abs(actual) : std::abs(actual - expected) / std::abs(expected);
            if (relative_error > RELTOL) {
                log_error(tt::LogAlways, "Mismatch at index {}: {} vs expected {}", i, actual, expected);
                log_error(
                    tt::LogAlways, "Expected relative tolerance: {} actual relative error: {}", RELTOL, relative_error);
                pass = false;
            }
        }

        pass &= prog_state.mesh_device->close();

    } catch (const std::exception& e) {
        log_error(tt::LogAlways, "Test failed with exception!");
        log_error(tt::LogAlways, "{}", e.what());
        throw;
    }

    if (pass) {
        log_info(tt::LogAlways, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}
