cmake_minimum_required(VERSION 3.16)
project(Ascend_c)

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend910B3" CACHE STRING "system on chip type")
set(KERNEL_TYPE "all" CACHE STRING "base/singlecoreSplitK/deterministicSplitK/all")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "ASCEND CANN package installation directory"
)
set(MMV3_OP_KERNEL_DIR "/root/bench_run/matmul_v3/op_kernel"
    CACHE STRING "path to mat_mul_v3 op_kernel sources"
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    # install-префикс зависит от RUN_MODE -> out/cpu, out/npu, out/sim.
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out/${RUN_MODE}"
        CACHE STRING "path for install()" FORCE)
endif()

# --- Имена артефактов в зависимости от RUN_MODE ---
set(EXECUTABLE_NAME "matmul_${RUN_MODE}")
set(KERNEL_LIB_NAME "ascendc_kernels_${RUN_MODE}")

# --- Выбор ядер ---------------------------------------------------------
# Каждое ядро = отдельный launch-файл + дефайн для хоста, чтобы
# ACLRT_LAUNCH_KERNEL не ссылался на несобранный символ.
set(MMV3_ALL_KERNELS base singlecoreSplitK deterministicSplitK)

if("${KERNEL_TYPE}" STREQUAL "all")
    set(MMV3_SELECTED ${MMV3_ALL_KERNELS})
else()
    if(NOT "${KERNEL_TYPE}" IN_LIST MMV3_ALL_KERNELS)
        message(FATAL_ERROR "KERNEL_TYPE must be one of: ${MMV3_ALL_KERNELS} all")
    endif()
    set(MMV3_SELECTED ${KERNEL_TYPE})
endif()

set(KERNEL_FILES "")
foreach(k IN LISTS MMV3_SELECTED)
    set(_src ${CMAKE_CURRENT_SOURCE_DIR}/matmulV3_launch_${k}.cpp)
    if(NOT EXISTS ${_src})
        message(FATAL_ERROR "kernel source not found: ${_src}")
    endif()
    list(APPEND KERNEL_FILES ${_src})
endforeach()

set(MMV3_KERNEL_DEFS "")
if("base" IN_LIST MMV3_SELECTED)
    list(APPEND MMV3_KERNEL_DEFS MM_HAVE_BASE)
endif()
if("singlecoreSplitK" IN_LIST MMV3_SELECTED)
    list(APPEND MMV3_KERNEL_DEFS MM_HAVE_SC_SPLITK)
endif()
if("deterministicSplitK" IN_LIST MMV3_SELECTED)
    list(APPEND MMV3_KERNEL_DEFS MM_HAVE_DET_SPLITK)
endif()

message(STATUS "[mmv3] kernels: ${MMV3_SELECTED}")
message(STATUS "[mmv3] defs:    ${MMV3_KERNEL_DEFS}")

set(CUSTOM_ASCEND310P_LIST "Ascend310P1" "Ascend310P3")

if("${RUN_MODE}" STREQUAL "cpu")
    include(cmake/cpu_lib.cmake)
elseif("${RUN_MODE}" STREQUAL "sim" OR "${RUN_MODE}" STREQUAL "npu")
    include(cmake/npu_lib.cmake)
else()
    message(FATAL_ERROR "invalid RUN_MODE: ${RUN_MODE}")
endif()

# Хост: тайлинг заполняется прямо в main_v3.cpp, отдельного tiling.cpp нет.
add_executable(ascendc_kernels_bbit
    ${CMAKE_CURRENT_SOURCE_DIR}/main_v3.cpp
)

set_target_properties(ascendc_kernels_bbit PROPERTIES
    OUTPUT_NAME "${EXECUTABLE_NAME}"
)

# Хосту нужен заголовок со структурой MatmulTilingData
target_include_directories(ascendc_kernels_bbit PRIVATE
    ${MMV3_OP_KERNEL_DIR}
)

target_compile_options(ascendc_kernels_bbit PRIVATE
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:-g>>
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall
)

target_compile_definitions(ascendc_kernels_bbit PRIVATE
    $<$<BOOL:$<IN_LIST:${SOC_VERSION},${CUSTOM_ASCEND310P_LIST}>>:CUSTOM_ASCEND310P>
    SOC_VERSION="${SOC_VERSION}"
    MM_KERNEL=0
    ${MMV3_KERNEL_DEFS}
)

target_link_libraries(ascendc_kernels_bbit PRIVATE
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:host_intf_pub>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:ascendcl>>
    ${KERNEL_LIB_NAME}
    tiling_api
    register
    platform
    ascendalog
    dl
)

install(TARGETS ascendc_kernels_bbit
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
