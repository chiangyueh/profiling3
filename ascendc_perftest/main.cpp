/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>  
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <getopt.h>
#include <fstream>

#include "acl/acl.h"
#include "utils.h"
#include "perftest_common_types.h"
#include "mte_perftest_common.h"

int g_npus = 2;
int ub_size = 16;
const char *ipport;
static const char *fuc_data_type;
int f_rank = 0;
int f_npu = 0;
extern "C" void device_perftest_demo(uint32_t block_dim, void *stream, uint8_t *d1_ptr, uint8_t *d0_ptr, uint8_t *time_gva, int elements, int test_type, int loop_count, int data_type, int ub_size_kb);

static perftest::ascendc_mode_t get_test_type(const char *test_type_str) {
    if (strcmp(test_type_str, "put") == 0) return perftest::TEST_MODE_ASCENDC_PUT;
    else if (strcmp(test_type_str, "get") == 0) return perftest::TEST_MODE_ASCENDC_GET;
    else if (strcmp(test_type_str, "ub2gm_local") == 0) return perftest::TEST_MODE_UB2GM_LOCAL;
    else if (strcmp(test_type_str, "ub2gm_remote") == 0) return perftest::TEST_MODE_UB2GM_REMOTE;
    else if (strcmp(test_type_str, "gm2ub_local") == 0) return perftest::TEST_MODE_GM2UB_LOCAL;
    else if (strcmp(test_type_str, "gm2ub_remote") == 0) return perftest::TEST_MODE_GM2UB_REMOTE;
    return perftest::TEST_MODE_GM2UB_REMOTE;
}

static void fill_data_by_type(std::vector<uint8_t>& data, uint64_t index, uint64_t value, const char *data_type) {
    size_t datatype_size = get_datatype_size(data_type);
    uint64_t offset_bytes = index *datatype_size;
    
    #define FILL_OP(type) \
        do { \
            type val = static_cast<type>(value); \
            if constexpr (std::is_same_v<type, int8_t>) val = static_cast<int8_t>(value % 128); \
            else if constexpr (std::is_same_v<type, uint8_t>) val = static_cast<uint8_t>(value % 256); \
            else if constexpr (std::is_same_v<type, int16_t>) val = static_cast<int16_t>(value % 32768); \
            else if constexpr (std::is_same_v<type, uint16_t>) val = static_cast<uint16_t>(value % 65536); \
            else if constexpr (std::is_same_v<type, char>) val = static_cast<char>(value % 128); \
            std::copy_n(reinterpret_cast<const char*>(&val), sizeof(type), reinterpret_cast<char*>(&data[offset_bytes])); \
        } while(0)
    
    DISPATCH_BY_TYPE(data_type, FILL_OP);
    
    #undef FILL_OP
}


static void print_last_element(uint8_t *data, const char *data_type, uint64_t all_size, const char *label, void *device_ptr) {
    size_t datatype_size = get_datatype_size(data_type);
    uint64_t last_offset = all_size - datatype_size;
    std::cout << "[" << label << "] last element: ";
    
    #define PRINT_OP(type) \
        do { \
            type val; \
            std::copy_n(reinterpret_cast<const char*>(&data[last_offset]), sizeof(type), reinterpret_cast<char*>(&val)); \
            if constexpr (std::is_same_v<type, int8_t> || std::is_same_v<type, uint8_t> || std::is_same_v<type, char>) { \
                std::cout << static_cast<int>(val); \
            } else { \
                std::cout << val; \
            } \
        } while(0)
    
    DISPATCH_BY_TYPE(data_type, PRINT_OP);
    
    #undef PRINT_OP
    
    std::cout << " address: " << std::hex << device_ptr << std::dec << std::endl;
}


static int run_rma_performance_test(const char *test_type, const std::vector<int> &block_sizes,
                                    int min_exponent, int max_exponent, int device1, int device2,
                                    int loop_count, const char *data_type)
{
    perftest::ascendc_mode_t test_type_enum = get_test_type(test_type);
    perftest::perf_data_type_t data_type_enum = get_data_type(data_type);
    int status = 0;
    aclrtStream stream_0 = nullptr;
    void *d0_ptr = nullptr;
    void *d1_ptr = nullptr;
    void *time_ptr = nullptr;
    uint8_t *src_host = nullptr;
    uint8_t *dst_host = nullptr;
    float *time_host = nullptr;

    status = aclInit(nullptr);
    
    status = aclrtSetDevice(device1);
    aclrtDeviceEnablePeerAccess(device2,0);
    status = aclrtCreateStream(&stream_0);
    status = aclrtSetDevice(device2);
    aclrtDeviceEnablePeerAccess(device1,0);
    
    
    std::vector<std::vector<std::string>> csv_data = {
        {"DataSize/B", "Npus", "Blocks", "UBsize/KB", "Bandwidth/GB/s", "CoreMaxTime/us", "SingleCoreTime/us"},
    };
    std::vector<int> powers_of_two;

    for (int exponent = min_exponent; exponent <= max_exponent; ++exponent) {
        int value = std::pow(2, exponent);
        powers_of_two.push_back(value);
    }
    size_t datatype_size = get_datatype_size(data_type);

     for (int block_size : block_sizes) {
        for (int datasize : powers_of_two) {
            std::cout << "\n========== [LOOP START] block_size=" << block_size << ", datasize=" << datasize << " ==========" << std::endl;
            
            d0_ptr = nullptr;
            d1_ptr = nullptr;
            time_ptr = nullptr;
            src_host = nullptr;
            dst_host = nullptr;
            time_host = nullptr;
            
            status = aclrtSetDevice(device2);
            int trans_size = datasize * block_size;
            int tile_length = datasize / datatype_size;
            int block_size_bytes = tile_length * datatype_size;
            int offset = (block_size_bytes < 512) ? 512/datatype_size : block_size_bytes/datatype_size;
            uint64_t all_size = offset * block_size * datatype_size;
            status = aclrtMalloc((void **) &(d1_ptr), all_size, ACL_MEM_MALLOC_HUGE_FIRST);

            status = aclrtSetDevice(device1);
            status = aclrtMalloc((void **) &(d0_ptr), all_size, ACL_MEM_MALLOC_HUGE_FIRST);
            status = aclrtMalloc((void **) &(time_ptr), sizeof(float) * 2048, ACL_MEM_MALLOC_HUGE_FIRST);

            std::cout << " block_size: " << block_size << " size: " << datasize << std::endl;

            std::vector<uint8_t> input(all_size, 0);
            for (uint64_t i = 0; i < all_size / datatype_size; i++) {
                fill_data_by_type(input, i, i + datasize, data_type);
            }

            status = aclrtMemcpy(d0_ptr, all_size, input.data(), all_size, ACL_MEMCPY_HOST_TO_DEVICE);
            
            std::vector<uint8_t> input_d1(all_size, 0);
            for (uint64_t i = 0; i < all_size / datatype_size; i++) {
                fill_data_by_type(input_d1, i, i + datasize + 1, data_type);
            }
            status = aclrtMemcpy(d1_ptr, all_size, input_d1.data(), all_size, ACL_MEMCPY_HOST_TO_DEVICE);
            status = aclrtMallocHost(reinterpret_cast<void**>(&src_host), all_size);
            status = aclrtMallocHost(reinterpret_cast<void**>(&dst_host), all_size);

            
            status = aclrtMemcpy(src_host, all_size, d0_ptr, all_size, ACL_MEMCPY_DEVICE_TO_HOST);
            status = aclrtMemcpy(dst_host, all_size, d1_ptr, all_size, ACL_MEMCPY_DEVICE_TO_HOST);
            
            std::vector<uint8_t> src_before(all_size);
            std::vector<uint8_t> dst_before(all_size);
            std::copy_n(reinterpret_cast<const char*>(src_host), all_size, reinterpret_cast<char*>(src_before.data()));
            std::copy_n(reinterpret_cast<const char*>(dst_host), all_size, reinterpret_cast<char*>(dst_before.data()));
            
            print_last_element(src_host, data_type, all_size, "start src", d0_ptr);
            print_last_element(dst_host, data_type, all_size, "start  dst", d1_ptr);

            device_perftest_demo(block_size, stream_0, (uint8_t *)d1_ptr, (uint8_t *)d0_ptr,  (uint8_t *)time_ptr, trans_size/datatype_size, static_cast<int>(test_type_enum), loop_count, static_cast<int>(data_type_enum), ub_size);
            status = aclrtSynchronizeStream(stream_0);
            
            status = aclrtMallocHost(reinterpret_cast<void**>(&time_host), sizeof(float)* 2048);
            status = aclrtMemcpy(time_host, sizeof(float)*2048, time_ptr, sizeof(float)*2048, ACL_MEMCPY_DEVICE_TO_HOST);
            float *usetime = (float*)time_host;
            
            std::cout << " time: " << *usetime << " size: " << datatype_size * trans_size << std::endl;

            status = aclrtMemcpy(src_host, all_size, d0_ptr, all_size, ACL_MEMCPY_DEVICE_TO_HOST);
            status = aclrtMemcpy(dst_host, all_size, d1_ptr, all_size, ACL_MEMCPY_DEVICE_TO_HOST);
            
            bool verify_success = true;
            
            auto compare_values = [&](uint8_t *ptr1, uint64_t offset1, uint8_t *ptr2, uint64_t offset2, const char *label1, const char *label2) -> bool {
                #define COMPARE_OP(type) \
                    do { \
                        type val1, val2; \
                        std::copy_n(reinterpret_cast<const char*>(&ptr1[offset1 * datatype_size]), sizeof(type), reinterpret_cast<char*>(&val1)); \
                        std::copy_n(reinterpret_cast<const char*>(&ptr2[offset2 * datatype_size]), sizeof(type), reinterpret_cast<char*>(&val2)); \
                        if (val1 != val2) { \
                            std::cout << "  [ERROR] Mismatch at index " << offset1 << ": " << label1 << "=" << static_cast<double>(val1) << ", " << label2 << "=" << static_cast<double>(val2) << std::endl; \
                            return false; \
                        } \
                        return true; \
                    } while(0)
                
                DISPATCH_BY_TYPE(data_type, COMPARE_OP);
                
                #undef COMPARE_OP
                
                return false;
            };

            if (strcmp(test_type, "put") == 0) {
                std::cout << "\n[Verification] put operation: Checking first value of each core..." << std::endl;
                std::cout << "  Block size: " << block_size << ", Tile length: " << tile_length << ", Offset: " << offset << std::endl;
                
                for (int block_idx = 0; block_idx < block_size; ++block_idx) {
                    uint64_t src_start = block_idx * offset;
                    uint64_t dst_start = block_idx * offset;
                    
                    if (src_start * datatype_size < all_size && dst_start * datatype_size < all_size) {
                        if (!compare_values(dst_host, dst_start, src_before.data(), src_start, "dst", "src_before")) {
                            verify_success = false;
                            std::cout << "  [ERROR] Block " << block_idx << ": values do not match!" << std::endl;
                            break;
                        }
                    }
                }
            } else if (strcmp(test_type, "get") == 0) {
                std::cout << "\n[Verification] get operation: Checking first value of each core..." << std::endl;
                std::cout << "  Block size: " << block_size << ", Tile length: " << tile_length << ", Offset: " << offset << std::endl;
                
                for (int block_idx = 0; block_idx < block_size; ++block_idx) {
                    uint64_t src_start = block_idx * offset;
                    uint64_t dst_start = block_idx * offset;
                    
                    if (src_start * datatype_size < all_size && dst_start * datatype_size < all_size) {
                        if (!compare_values(src_host, src_start, dst_before.data(), dst_start, "src", "dst_before")) {
                            verify_success = false;
                            std::cout << "  [ERROR] Block " << block_idx << ": values do not match!" << std::endl;
                            break;
                        }
                    }
                }
            } else {
                std::cout << "\n[Verification] " << test_type << " operation: Verification skipped for this test type." << std::endl;
            }

            if (verify_success) {
                std::cout << "[Verification] SUCCESS: All cores' first values transferred correctly!" << std::endl;
            } else {
                std::cout << "[Verification] FAILED: At least one core's first value was not transferred correctly!" << std::endl;
            }
            
            std::cout << "" << std::endl;

            float max_core_time = 0.0f;
            float mean_core_time = 0.0f;
            
            for (int i = 0; i < block_size; i++) {
                if (*(usetime+i*16) > max_core_time){
                    max_core_time = *(usetime+i*16);
                }
            }
            
            double bandwidth = 0.0;
            if (max_core_time > 0) {
                bandwidth = (double)datasize * (double)block_size / (double)max_core_time * 1000000.0 / 1024.0 / 1024.0 / 1024.0;
            }
            
            std::vector<std::string> sub_data = {
                int_to_string(datasize), 
                int_to_string(g_npus), 
                int_to_string(block_size), 
                int_to_string(ub_size), 
                float_to_string(bandwidth),
                float_to_string(max_core_time)
            };

            for (int i = 0; i < block_size; i++) {
                mean_core_time += *(usetime+i*16);
                sub_data.push_back(float_to_string(*(usetime+i*16)));
            }
            
            csv_data.push_back(sub_data);

            
            status = aclrtFreeHost(src_host);
            status = aclrtFreeHost(dst_host);
            status = aclrtFreeHost(time_host);
            status = aclrtSetDevice(device2);
            status = aclrtFree(d1_ptr);
            status = aclrtSetDevice(device1);
            status = aclrtFree(time_ptr);
            status = aclrtFree(d0_ptr);
            status = aclrtSynchronizeStream(stream_0);
            
            std::cout << "========== [LOOP END] block_size=" << block_size << ", datasize=" << datasize << " ==========\n" << std::endl;
        }
     }


    status = aclrtDestroyStream(stream_0);
    status = aclrtResetDevice(device1);
    status = aclrtResetDevice(device2);
    status = aclFinalize();

    
    write_csv("output/" + int_to_string(0) + "_" + std::string(test_type) + "_" + std::string(data_type) + ".csv", csv_data);

    return 0;
}

int main(int argc, char *argv[])
{
    int status = 0;
    const char *test_type = "gm2ub_remote";
    const char *data_type = "float";
    int min_block_size = 32;
    int max_block_size = 32;
    int min_exponent = 3;
    int max_exponent = 20;
    int device1 = 1;
    int device2 = 2;
    int loop_count = 50;
    int ub_size_kb = 16;
    std::vector<int> block_sizes;
    
    static struct option long_options[] = {
        {"test-type", required_argument, 0, 't'},
        {"datatype", required_argument, 0, 'd'},
        {"block-size", required_argument, 0, 'b'},
        {"block-range", required_argument, 0, 0},
        {"block-list", required_argument, 0, 0},
        {"exponent", required_argument, 0, 'e'},
        {"exponent-range", required_argument, 0, 0},
        {"device1", required_argument, 0, 0},
        {"device2", required_argument, 0, 0},
        {"loop-count", required_argument, 0, 0},
        {"ub-size", required_argument, 0, 0},
        {0, 0, 0, 0}
    };
    
    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "t:d:b:e:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 't':
                test_type = optarg;
                break;
            case 'd':
                data_type = optarg;
                break;
            case 'b':
                min_block_size = max_block_size = std::atoi(optarg);
                break;
            case 'e':
                min_exponent = max_exponent = std::atoi(optarg);
                break;
            case 0:
                if (strcmp(long_options[option_index].name, "block-range") == 0) {
                    min_block_size = std::atoi(optarg);
                    if (optind < argc) {
                        max_block_size = std::atoi(argv[optind]);
                        optind++;
                    }
                } else if (strcmp(long_options[option_index].name, "block-list") == 0) {
                    block_sizes = parse_block_list(optarg);
                    if (block_sizes.empty()) {
                        std::cerr << "错误: --block-list 格式无效，示例: --block-list 2,4,6,8" << std::endl;
                        return 1;
                    }
                } else if (strcmp(long_options[option_index].name, "exponent-range") == 0) {
                    min_exponent = std::atoi(optarg);
                    if (optind < argc) {
                        max_exponent = std::atoi(argv[optind]);
                        optind++;
                    }
                } else if (strcmp(long_options[option_index].name, "device1") == 0) {
                    device1 = std::atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "device2") == 0) {
                    device2 = std::atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "loop-count") == 0) {
                    loop_count = std::atoi(optarg);
                } else if (strcmp(long_options[option_index].name, "ub-size") == 0) {
                    ub_size_kb = std::atoi(optarg);
                }
                break;
            default:
                std::cerr << "错误: 未知参数" << std::endl;
                std::cerr << "使用方法: " << argv[0] << " [-t <put|get|ub2gm_local|ub2gm_remote|gm2ub_local|gm2ub_remote>] [-d <float|int8|int16|int32|int64|uint8|uint16|uint32|uint64|char>] [-b <block_size>] [-e <exponent>] [--block-range <min> <max>] [--block-list <b1,b2,...>] [--exponent-range <min> <max>] [--device1 <device_id>] [--device2 <device_id>] [--loop-count <count>]" << std::endl;
                return 1;
        }
    }

    if (block_sizes.empty()) {
        if (min_block_size <= 0 || max_block_size < min_block_size) {
            std::cerr << "错误: block-range 无效，min 必须 > 0 且 max >= min" << std::endl;
            return 1;
        }
        for (int block_size = min_block_size; block_size <= max_block_size; block_size++) {
            block_sizes.push_back(block_size);
        }
    }

    if (strcmp(test_type, "put") != 0 && strcmp(test_type, "get") != 0 && 
        strcmp(test_type, "ub2gm_local") != 0 && strcmp(test_type, "ub2gm_remote") != 0 &&
        strcmp(test_type, "gm2ub_local") != 0 && strcmp(test_type, "gm2ub_remote") != 0) {
        std::cerr << "错误: 测试类型必须是 'put'、'get'、'ub2gm_local'、'ub2gm_remote'、'gm2ub_local' 或 'gm2ub_remote'" << std::endl;
        std::cerr << "使用方法: " << argv[0] << " [-t <put|get|ub2gm_local|ub2gm_remote|gm2ub_local|gm2ub_remote>] [-d <float|int8|int16|int32|int64|uint8|uint16|uint32|uint64|char>] [-b <block_size>] [-e <exponent>] [--block-range <min> <max>] [--block-list <b1,b2,...>] [--exponent-range <min> <max>] [--device1 <device_id>] [--device2 <device_id>] [--loop-count <count>]" << std::endl;
        return 1;
    }
    
    if (strcmp(data_type, "float") != 0 && strcmp(data_type, "int8") != 0 && 
        strcmp(data_type, "int16") != 0 && strcmp(data_type, "int32") != 0 && 
        strcmp(data_type, "int64") != 0 && strcmp(data_type, "uint8") != 0 && 
        strcmp(data_type, "uint16") != 0 && strcmp(data_type, "uint32") != 0 && 
        strcmp(data_type, "uint64") != 0 && strcmp(data_type, "char") != 0) {
        std::cerr << "错误: 数据类型必须是 'float'、'int8'、'int16'、'int32'、'int64'、'uint8'、'uint16'、'uint32'、'uint64' 或 'char'" << std::endl;
        std::cerr << "使用方法: " << argv[0] << " [-t <put|get|ub2gm_local|ub2gm_remote|gm2ub_local|gm2ub_remote>] [-d <float|int8|int16|int32|int64|uint8|uint16|uint32|uint64|char>] [-b <block_size>] [-e <exponent>] [--block-range <min> <max>] [--block-list <b1,b2,...>] [--exponent-range <min> <max>] [--device1 <device_id>] [--device2 <device_id>] [--loop-count <count>]" << std::endl;
        return 1;
    }
    
    std::cout << "[SUCCESS] demo run start in rank, test type: " << test_type << ", data type: " << data_type << std::endl;
    print_block_sizes(block_sizes);
    std::cout << "幂数范围: " << min_exponent << "-" << max_exponent << std::endl;
    std::cout << "设备ID: " << device1 << ", " << device2 << std::endl;
    std::cout << "循环次数: " << loop_count << std::endl;
    std::cout << "UB size (KB): " << ub_size_kb << std::endl;
    ub_size = ub_size_kb;
    status = run_rma_performance_test(test_type, block_sizes, min_exponent, max_exponent, device1, device2, loop_count, data_type);
    std::cout << "[SUCCESS] demo run success in rank " << std::endl;
    
    return 0;
}
