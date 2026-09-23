/* -------------------------------------------------------------------------
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is part of the MindStudio project.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *    http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
*/

#ifndef MSPTI_CBID_H
#define MSPTI_CBID_H

/**
 * @brief Definitions of indices for Runtime API functions, unique across entire API
 */
typedef enum {
    MSPTI_CBID_RUNTIME_INVALID                              = 0,
    MSPTI_CBID_RUNTIME_DEVICE_SET                           = 1,
    MSPTI_CBID_RUNTIME_DEVICE_RESET                         = 2,
    MSPTI_CBID_RUNTIME_DEVICE_SET_EX                        = 3,
    MSPTI_CBID_RUNTIME_CONTEXT_CREATED_EX                   = 4,
    MSPTI_CBID_RUNTIME_CONTEXT_CREATED                      = 5,
    MSPTI_CBID_RUNTIME_CONTEXT_DESTROY                      = 6,
    MSPTI_CBID_RUNTIME_STREAM_CREATED                       = 7,
    MSPTI_CBID_RUNTIME_STREAM_DESTROY                       = 8,
    MSPTI_CBID_RUNTIME_STREAM_SYNCHRONIZED                  = 9,
    MSPTI_CBID_RUNTIME_LAUNCH                               = 10,
    MSPTI_CBID_RUNTIME_CPU_LAUNCH                           = 11,
    MSPTI_CBID_RUNTIME_AICPU_LAUNCH                         = 12,
    MSPTI_CBID_RUNTIME_AIV_LAUNCH                           = 13,
    MSPTI_CBID_RUNTIME_FFTS_LAUNCH                          = 14,
    MSPTI_CBID_RUNTIME_MALLOC                               = 15,
    MSPTI_CBID_RUNTIME_FREE                                 = 16,
    MSPTI_CBID_RUNTIME_MALLOC_HOST                          = 17,
    MSPTI_CBID_RUNTIME_FREE_HOST                            = 18,
    MSPTI_CBID_RUNTIME_MALLOC_CACHED                        = 19,
    MSPTI_CBID_RUNTIME_FLUSH_CACHE                          = 20,
    MSPTI_CBID_RUNTIME_INVALID_CACHE                        = 21,
    MSPTI_CBID_RUNTIME_MEMCPY                               = 22,
    MSPTI_CBID_RUNTIME_MEMCPY_HOST                          = 23,
    MSPTI_CBID_RUNTIME_MEMCPY_ASYNC                         = 24,
    MSPTI_CBID_RUNTIME_MEM_CPY2D                            = 25,
    MSPTI_CBID_RUNTIME_MEM_CPY2D_ASYNC                      = 26,
    MSPTI_CBID_RUNTIME_MEM_SET                              = 27,
    MSPTI_CBID_RUNTIME_MEM_SET_ASYNC                        = 28,
    MSPTI_CBID_RUNTIME_MEM_GET_INFO                         = 29,
    MSPTI_CBID_RUNTIME_RESERVE_MEM_ADDRESS                  = 30,
    MSPTI_CBID_RUNTIME_RELEASE_MEM_ADDRESS                  = 31,
    MSPTI_CBID_RUNTIME_MALLOC_PHYSICAL                      = 32,
    MSPTI_CBID_RUNTIME_FREE_PHYSICAL                        = 33,
    MSPTI_CBID_RUNTIME_MEM_EXPORT_TO_SHAREABLE_HANDLE       = 34,
    MSPTI_CBID_RUNTIME_MEM_IMPORT_FROM_SHAREABLE_HANDLE     = 35,
    MSPTI_CBID_RUNTIME_MEM_SET_PID_TO_SHAREABLE_HANDLE      = 36,
    MSPTI_CBID_RUNTIME_SIZE,
    MSPTI_CBID_RUNTIME_FORCE_INT                            = 0x7fffffff
} msptiCallbackIdRuntime;

/**
 * @brief Definitions of indices for hccl API functions
 */
typedef enum {
    MSPTI_CBID_HCCL_INVALID                                 = 0,
    MSPTI_CBID_HCCL_ALLREDUCE                               = 1,
    MSPTI_CBID_HCCL_BROADCAST                               = 2,
    MSPTI_CBID_HCCL_ALLGATHER                               = 3,
    MSPTI_CBID_HCCL_REDUCE_SCATTER                          = 4,
    MSPTI_CBID_HCCL_REDUCE                                  = 5,
    MSPTI_CBID_HCCL_ALL_TO_ALL                              = 6,
    MSPTI_CBID_HCCL_ALL_TO_ALLV                             = 7,
    MSPTI_CBID_HCCL_BARRIER                                 = 8,
    MSPTI_CBID_HCCL_SCATTER                                 = 9,
    MSPTI_CBID_HCCL_SEND                                    = 10,
    MSPTI_CBID_HCCL_RECV                                    = 11,
    MSPTI_CBID_HCCL_SENDRECV                                = 12,
    MSPTI_CBID_HCCL_SIZE,
    MSPTI_CBID_HCCL_FORCE_INT                            = 0x7fffffff
} msptiCallbackIdHccl;

#endif
