// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_splitk_c_shuffle.hpp"

#include "device_gemm_xdl_splitk_bf16_i8_bf16_mk_nk_mn_common.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_xdl_splitk_bf16_i8_bf16_mk_nk_mn_v1_interwave_instances(
    std::vector<std::unique_ptr<
        DeviceGemmSplitK<Row, Col, Row, BF16, I8, BF16, PassThrough, PassThrough, PassThrough>>>&
        instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_xdl_splitk_bf16_i8_bf16_mk_nk_mn_instances<GemmDefault,
                                                               PipelineVersion::v1,
                                                               LoopScheduler::Interwave>{});

    add_device_operation_instances(
        instances,
        device_gemm_xdl_splitk_bf16_i8_bf16_mk_nk_mn_instances<GemmMNPadding,
                                                               PipelineVersion::v1,
                                                               LoopScheduler::Interwave>{});

    add_device_operation_instances(
        instances,
        device_gemm_xdl_splitk_bf16_i8_bf16_mk_nk_mn_instances<GemmMNKPadding,
                                                               PipelineVersion::v1,
                                                               LoopScheduler::Interwave>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck