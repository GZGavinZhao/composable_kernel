// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_dl.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#ifdef __int8__
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto GemmMNPadding = ck::tensor_operation::device::GemmSpecialization::MNPadding;

// Compilation parameters for a[m, k] * b[k, n] = c[m, n]
using device_gemm_dl_i8_i8_i8_mk_kn_mn_irregular_instances = std::tuple<
    // clang-format off
        // #########|  AData|   BData|   CData|    AccData| ALayout| BLayout| CLayout|           A|           B|           C|           GEMM| Block|  MPer|  NPer| K0Per| K1|      M1Per|      N1Per|   KPer|  M11N11Thread|  M11N11Thread|     ABlockTransfer|       ABlockTransfer| ABlockTransfer| ABlockTransfer|      ABlockTransfer|     ABlockTransfer|       ABlockTransfer|     BBlockTransfer|       BBlockTransfer| BBlockTransfer| BBlockTransfer|      BBlockTransfer|     BBlockTransfer|       BBlockTransfer|     CThreadTransfer| CThreadTransfer|    CThreadTransfer|
        // #########|   Type|    Type|    Type|       Type|        |        |        | Elementwise| Elementwise| Elementwise| Specialization|  Size| Block| Block| Block|   | ThreadM111| ThreadN111| Thread| ClusterM110Xs| ClusterN110Xs| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|      DstVectorTensor| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|      DstVectorTensor|        SrcDstAccess| SrcDstVectorDim| DstScalarPerVector|
        // #########|       |        |        |           |        |        |        |   Operation|   Operation|   Operation|               |      |      |      |      |   |           |           |       |              |              |        K0_M0_M1_K1|          K0_M0_M1_K1|   ArrangeOrder|          Order| Lengths_K0_M0_M1_K1| ContiguousDimOrder|  Lengths_K0_M0_M1_K1|        K0_N0_N1_K1|          K0_N0_N1_K1|   ArrangeOrder|          Order| Lengths_K0_N0_N1_K1| ContiguousDimOrder|  Lengths_K0_N0_N1_K1|               Order|                |                   |
        // #########|       |        |        |           |        |        |        |            |            |            |               |      |      |      |      |   |           |           |       |              |              |                   |                     |               |               |                    |                   |                     |                   |                     |               |               |                    |                   |                     |                    |                |                   |
        // MPerBlock=128, NPerBlock=128
        DeviceGemmDl< int8_t, int8_t, int8_t,      int32_t,     Row,     Row,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,   128,   128,   128,    16,  4,          4,          8,      1,       S<8, 2>,       S<4, 2>,      S<8, 1, 2, 4>,       S<2, 1, 64, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 4>,      S<1, 2, 0, 3>,        S<1, 1, 1, 4>,      S<2, 1, 8, 4>,       S<8, 1, 16, 1>,  S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<1, 1, 4, 1>,      S<0, 3, 1, 2>,        S<1, 1, 4, 4>, S<0, 1, 2, 3, 4, 5>,                5,                  1>,
        DeviceGemmDl< int8_t, int8_t, int8_t,      int32_t,     Row,     Row,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,   128,   128,   128,    16,  4,          4,          8,      1,       S<4, 4>,       S<4, 2>,      S<8, 1, 2, 4>,       S<2, 1, 64, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 4>,      S<1, 2, 0, 3>,        S<1, 1, 1, 4>,      S<2, 1, 8, 4>,       S<8, 1, 16, 1>,  S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<1, 1, 4, 1>,      S<0, 3, 1, 2>,        S<1, 1, 4, 4>, S<0, 1, 2, 3, 4, 5>,                5,                  1>,
        DeviceGemmDl< int8_t, int8_t, int8_t,      int32_t,     Row,     Row,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,   128,   128,   128,    16,  4,          4,          8,      1,       S<2, 8>,       S<2, 4>,      S<8, 1, 2, 4>,       S<2, 1, 64, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 4>,      S<1, 2, 0, 3>,        S<1, 1, 1, 4>,      S<2, 1, 8, 4>,       S<8, 1, 16, 1>,  S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<1, 1, 4, 1>,      S<0, 3, 1, 2>,        S<1, 1, 4, 4>, S<0, 1, 2, 3, 4, 5>,                5,                  1>,
        // MPerBlock=128, NPerBlock=64
        DeviceGemmDl< int8_t, int8_t, int8_t,      int32_t,     Row,     Row,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,   256,   128,    64,    16,  4,          4,          2,      1,       S<4, 4>,       S<4, 4>,      S<8, 1, 1, 4>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 4>,      S<1, 2, 0, 3>,        S<1, 1, 1, 4>,      S<2, 1, 4, 4>,       S<8, 1, 16, 1>,  S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<1, 1, 4, 1>,      S<0, 3, 1, 2>,        S<1, 1, 4, 4>, S<0, 1, 2, 3, 4, 5>,                5,                  1>,
        DeviceGemmDl< int8_t, int8_t, int8_t,      int32_t,     Row,     Row,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,   256,   128,    64,    16,  4,          4,          2,      1,       S<2, 8>,       S<2, 8>,      S<8, 1, 1, 4>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 4>,      S<1, 2, 0, 3>,        S<1, 1, 1, 4>,      S<2, 1, 4, 4>,       S<8, 1, 16, 1>,  S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<1, 1, 4, 1>,      S<0, 3, 1, 2>,        S<1, 1, 4, 4>, S<0, 1, 2, 3, 4, 5>,                5,                  1>,
        // MPerBlock=64, NPerBlock=128
        DeviceGemmDl< int8_t, int8_t, int8_t,      int32_t,     Row,     Row,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,   256,    64,   128,    16,  4,          2,          4,      1,       S<4, 4>,       S<4, 4>,      S<8, 1, 1, 4>,       S<2, 1, 64, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 4>,      S<1, 2, 0, 3>,        S<1, 1, 1, 4>,      S<2, 1, 4, 4>,       S<8, 1, 32, 1>,  S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<1, 1, 4, 1>,      S<0, 3, 1, 2>,        S<1, 1, 4, 4>, S<0, 1, 2, 3, 4, 5>,                5,                  1>,
        DeviceGemmDl< int8_t, int8_t, int8_t,      int32_t,     Row,     Row,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,   256,    64,   128,    16,  4,          2,          4,      1,       S<2, 8>,       S<2, 8>,      S<8, 1, 1, 4>,       S<2, 1, 64, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 4>,      S<1, 2, 0, 3>,        S<1, 1, 1, 4>,      S<2, 1, 4, 4>,       S<8, 1, 32, 1>,  S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<1, 1, 4, 1>,      S<0, 3, 1, 2>,        S<1, 1, 4, 4>, S<0, 1, 2, 3, 4, 5>,                5,                  1>,
        // MPerBlock=64, NPerBlock=64
        DeviceGemmDl< int8_t, int8_t, int8_t,      int32_t,     Row,     Row,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,    64,    64,    64,     8,  4,          4,          4,      1,       S<4, 2>,       S<4, 2>,      S<4, 1, 2, 4>,       S<2, 1, 32, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 4>,      S<1, 2, 0, 3>,        S<1, 1, 1, 4>,      S<2, 1, 4, 4>,       S<4, 1, 16, 1>,  S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<1, 1, 4, 1>,      S<0, 3, 1, 2>,        S<1, 1, 4, 4>, S<0, 1, 2, 3, 4, 5>,                5,                  1>,
        DeviceGemmDl< int8_t, int8_t, int8_t,      int32_t,     Row,     Row,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,    64,    64,    64,     8,  4,          4,          4,      1,       S<2, 4>,       S<2, 4>,      S<4, 1, 2, 4>,       S<2, 1, 32, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 4>,      S<1, 2, 0, 3>,        S<1, 1, 1, 4>,      S<2, 1, 4, 4>,       S<4, 1, 16, 1>,  S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<1, 1, 4, 1>,      S<0, 3, 1, 2>,        S<1, 1, 4, 4>, S<0, 1, 2, 3, 4, 5>,                5,                  1>,
        DeviceGemmDl< int8_t, int8_t, int8_t,      int32_t,     Row,     Row,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,    64,    64,    64,     8,  4,          4,          4,      1,       S<8, 1>,       S<4, 2>,      S<4, 1, 2, 4>,       S<2, 1, 32, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 4>,      S<1, 2, 0, 3>,        S<1, 1, 1, 4>,      S<2, 1, 4, 4>,       S<4, 1, 16, 1>,  S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<1, 1, 4, 1>,      S<0, 3, 1, 2>,        S<1, 1, 4, 4>, S<0, 1, 2, 3, 4, 5>,                5,                  1>,
        DeviceGemmDl< int8_t, int8_t, int8_t,      int32_t,     Row,     Row,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,    64,    64,    64,     8,  4,          4,          4,      1,       S<4, 2>,       S<8, 1>,      S<4, 1, 2, 4>,       S<2, 1, 32, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 4>,      S<1, 2, 0, 3>,        S<1, 1, 1, 4>,      S<2, 1, 4, 4>,       S<4, 1, 16, 1>,  S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<1, 1, 4, 1>,      S<0, 3, 1, 2>,        S<1, 1, 4, 4>, S<0, 1, 2, 3, 4, 5>,                5,                  1>,
        // MPerBlock=32, NPerBlock=32
        DeviceGemmDl< int8_t, int8_t, int8_t,      int32_t,     Row,     Row,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,    32,    32,    32,     8,  4,          2,          4,      1,       S<4, 2>,       S<2, 2>,      S<4, 1, 2, 4>,       S<2, 1, 16, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 4>,      S<1, 2, 0, 3>,        S<1, 1, 1, 4>,      S<2, 1, 4, 4>,       S<4, 1, 8, 1>,   S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<1, 1, 4, 1>,      S<0, 3, 1, 2>,        S<1, 1, 4, 4>, S<0, 1, 2, 3, 4, 5>,                5,                  1>,
        DeviceGemmDl< int8_t, int8_t, int8_t,      int32_t,     Row,     Row,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,    32,    32,    32,     8,  4,          4,          2,      1,       S<2, 2>,       S<4, 2>,      S<4, 1, 2, 4>,       S<2, 1, 16, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 4>,      S<1, 2, 0, 3>,        S<1, 1, 1, 4>,      S<2, 1, 4, 4>,       S<4, 1, 8, 1>,   S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<1, 1, 4, 1>,      S<0, 3, 1, 2>,        S<1, 1, 4, 4>, S<0, 1, 2, 3, 4, 5>,                5,                  1>,
        DeviceGemmDl< int8_t, int8_t, int8_t,      int32_t,     Row,     Row,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,    32,    32,    32,     8,  4,          4,          2,      1,       S<2, 2>,       S<2, 4>,      S<4, 1, 2, 4>,       S<2, 1, 16, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 4>,      S<1, 2, 0, 3>,        S<1, 1, 1, 4>,      S<2, 1, 4, 4>,       S<4, 1, 8, 1>,   S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<1, 1, 4, 1>,      S<0, 3, 1, 2>,        S<1, 1, 4, 4>, S<0, 1, 2, 3, 4, 5>,                5,                  1>,
        // MPerBlock=16, NPerBlock=16
        DeviceGemmDl< int8_t, int8_t, int8_t,      int32_t,     Row,     Row,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,    16,    16,    16,    16,  2,          2,          2,      1,       S<2, 2>,       S<2, 2>,      S<4, 1, 4, 2>,        S<4, 1, 4, 1>,  S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<4, 1, 1, 2>,      S<0, 3, 1, 2>,        S<1, 1, 1, 2>,      S<4, 1, 4, 2>,        S<4, 1, 4, 1>,   S<0, 3, 1, 2>,  S<0, 3, 1, 2>,      S<1, 1, 4, 1>,      S<0, 3, 1, 2>,        S<1, 1, 4, 2>, S<0, 1, 2, 3, 4, 5>,                5,                  1>,
        DeviceGemmDl< int8_t, int8_t, int8_t,      int32_t,     Row,     Row,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,    16,    16,    16,    16,  2,          2,          2,      1,       S<4, 1>,       S<4, 1>,      S<4, 1, 4, 2>,        S<4, 1, 4, 1>,  S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<4, 1, 1, 2>,      S<0, 3, 1, 2>,        S<1, 1, 1, 2>,      S<4, 1, 4, 2>,        S<4, 1, 4, 1>,   S<0, 3, 1, 2>,  S<0, 3, 1, 2>,      S<1, 1, 4, 1>,      S<0, 3, 1, 2>,        S<1, 1, 4, 2>, S<0, 1, 2, 3, 4, 5>,                5,                  1>,
        // MPerBlock=8, NPerBlock=64
        DeviceGemmDl< int8_t, int8_t, int8_t,      int32_t,     Row,     Row,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,    64,     8,    64,    32,  2,          1,          2,      1,       S<2, 2>,       S<8, 2>,      S<4, 1, 1, 2>,        S<8, 1, 8, 1>,  S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<4, 1, 1, 2>,      S<0, 3, 1, 2>,        S<1, 1, 1, 2>,      S<8, 1, 4, 2>,       S<4, 1, 16, 1>,   S<0, 3, 1, 2>,  S<0, 3, 1, 2>,      S<1, 1, 4, 1>,      S<0, 3, 1, 2>,        S<1, 1, 4, 2>, S<0, 1, 2, 3, 4, 5>,                5,                  1>,
        // MPerBlock=64, NPerBlock=8
        DeviceGemmDl< int8_t, int8_t, int8_t,      int32_t,     Row,     Row,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,    64,    64,     8,    32,  2,          2,          1,      1,       S<8, 2>,       S<2, 2>,      S<8, 1, 4, 2>,       S<4, 1, 16, 1>,  S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<4, 1, 1, 2>,      S<0, 3, 1, 2>,        S<1, 1, 1, 2>,      S<1, 1, 4, 2>,       S<32, 1, 2, 1>,   S<0, 3, 1, 2>,  S<0, 3, 1, 2>,      S<1, 1, 4, 1>,      S<0, 3, 1, 2>,        S<1, 1, 4, 2>, S<0, 1, 2, 3, 4, 5>,                5,                  1>,
        // MPerBlock=8, NPerBlock=8
        DeviceGemmDl< int8_t, int8_t, int8_t,      int32_t,     Row,     Row,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,     8,     8,     8,     4,  2,          1,          2,      1,       S<4, 1>,       S<2, 1>,      S<4, 1, 1, 2>,        S<1, 1, 8, 1>,  S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<4, 1, 1, 2>,      S<0, 3, 1, 2>,        S<1, 1, 1, 2>,      S<1, 1, 4, 2>,        S<4, 1, 2, 1>,   S<0, 3, 1, 2>,  S<0, 3, 1, 2>,      S<1, 1, 4, 1>,      S<0, 3, 1, 2>,        S<1, 1, 4, 2>, S<0, 1, 2, 3, 4, 5>,                5,                  1>,
        DeviceGemmDl< int8_t, int8_t, int8_t,      int32_t,     Row,     Row,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,     8,     8,     8,     4,  2,          1,          2,      1,       S<1, 4>,       S<1, 2>,      S<4, 1, 1, 2>,        S<1, 1, 8, 1>,  S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<4, 1, 1, 2>,      S<0, 3, 1, 2>,        S<1, 1, 1, 2>,      S<1, 1, 4, 2>,        S<4, 1, 2, 1>,   S<0, 3, 1, 2>,  S<0, 3, 1, 2>,      S<1, 1, 4, 1>,      S<0, 3, 1, 2>,        S<1, 1, 4, 2>, S<0, 1, 2, 3, 4, 5>,                5,                  1>,
        DeviceGemmDl< int8_t, int8_t, int8_t,      int32_t,     Row,     Row,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,     8,     8,     8,     4,  2,          2,          1,      1,       S<2, 1>,       S<4, 1>,      S<4, 1, 1, 2>,        S<1, 1, 8, 1>,  S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<4, 1, 1, 2>,      S<0, 3, 1, 2>,        S<1, 1, 1, 2>,      S<1, 1, 4, 2>,        S<4, 1, 2, 1>,   S<0, 3, 1, 2>,  S<0, 3, 1, 2>,      S<1, 1, 4, 1>,      S<0, 3, 1, 2>,        S<1, 1, 4, 2>, S<0, 1, 2, 3, 4, 5>,                5,                  1>,
        DeviceGemmDl< int8_t, int8_t, int8_t,      int32_t,     Row,     Row,     Row, PassThrough, PassThrough, PassThrough,  GemmMNPadding,     8,     8,     8,     4,  2,          2,          1,      1,       S<1, 2>,       S<1, 4>,      S<4, 1, 1, 2>,        S<1, 1, 8, 1>,  S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<4, 1, 1, 2>,      S<0, 3, 1, 2>,        S<1, 1, 1, 2>,      S<1, 1, 4, 2>,        S<4, 1, 2, 1>,   S<0, 3, 1, 2>,  S<0, 3, 1, 2>,      S<1, 1, 4, 1>,      S<0, 3, 1, 2>,        S<1, 1, 4, 2>, S<0, 1, 2, 3, 4, 5>,                5,                  1>
    // clang-format on
    >;

void add_device_gemm_dl_i8_i8_i8_mk_kn_mn_irregular_instances(
    std::vector<std::unique_ptr<
        DeviceGemm<Row, Row, Row, int8_t, int8_t, int8_t, PassThrough, PassThrough, PassThrough>>>&
        instances)
{
    add_device_operation_instances(instances,
                                   device_gemm_dl_i8_i8_i8_mk_kn_mn_irregular_instances{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
