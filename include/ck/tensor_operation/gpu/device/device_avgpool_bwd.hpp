// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>

#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/utility/reduction_enums.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename DOutDataType, typename DInDataType>
struct DeviceAvgPoolBwd : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_dout,
                        void* p_din,
                        std::vector<ck::index_t> dout_n_k_wos_lengths,
                        std::vector<ck::index_t> dout_n_k_wos_strides,
                        std::vector<ck::index_t> din_n_k_wos_length,
                        std::vector<ck::index_t> din_n_k_wos_strides,
                        std::vector<ck::index_t> window_k_c_xs_lengths,
                        std::vector<ck::index_t> window_strides,
                        std::vector<ck::index_t> window_dilations,
                        std::vector<ck::index_t> input_left_pads,
                        std::vector<ck::index_t> input_right_pads) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck