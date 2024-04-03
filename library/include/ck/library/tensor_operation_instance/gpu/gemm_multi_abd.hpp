// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <vector>
#include <memory>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_abd.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_abd_xdl_cshuffle.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using Scales      = ck::tensor_operation::element_wise::Scales;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using AddFastGelu = ck::tensor_operation::element_wise::AddFastGelu;

void add_device_gemm_xdl_multi_abd_bf16_i8_bf16_mk_kn_mn_bias_gelu_v1_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleABD<ck::Tuple<Row>,
                                                      ck::Tuple<Row, Row>,
                                                      ck::Tuple<Row>,
                                                      Row,
                                                      ck::Tuple<BF16>,
                                                      ck::Tuple<I8, BF16>,
                                                      ck::Tuple<BF16>,
                                                      BF16,
                                                      PassThrough,
                                                      Scales,
                                                      AddFastGelu>>>& instances);

// Multiply + GEMM + Add + Gelu
template <typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename ELayout,
          typename AsDataType,
          typename BsDataType,
          typename DsDataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGemmMultipleABD<AsLayout,
                                                        BsLayout,
                                                        DsLayout,
                                                        ELayout,
                                                        AsDataType,
                                                        BsDataType,
                                                        DsDataType,
                                                        EDataType,
                                                        PassThrough,
                                                        Scales,
                                                        AddFastGelu>>
{
    using DeviceOp = DeviceGemmMultipleABD<AsLayout,
                                           BsLayout,
                                           DsLayout,
                                           ELayout,
                                           AsDataType,
                                           BsDataType,
                                           DsDataType,
                                           EDataType,
                                           PassThrough,
                                           Scales,
                                           AddFastGelu>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(is_same_v<AsDataType, ck::Tuple<BF16>> &&
                     is_same_v<BsDataType, ck::Tuple<I8, BF16>> &&
                     is_same_v<DsDataType, ck::Tuple<BF16>> && is_same_v<EDataType, BF16>)
        {
            if constexpr(is_same_v<AsLayout, ck::Tuple<Row>> &&
                         is_same_v<BsLayout, ck::Tuple<Row, Row>> &&
                         is_same_v<DsLayout, ck::Tuple<Row>> && is_same_v<ELayout, Row>)
            {
                add_device_gemm_xdl_multi_abd_bf16_i8_bf16_mk_kn_mn_bias_gelu_v1_instances(op_ptrs);
            }
        }

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
