cmake_minimum_required(VERSION 3.22)
project(minicpm_v47_android LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
get_filename_component(OMNI_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../vendor/llama.cpp-omni-v47" ABSOLUTE)
if(NOT EXISTS "${OMNI_ROOT}/tools/mtmd/mtmd.h")
    message(FATAL_ERROR "Run python3 scripts/prepare_v47.py first")
endif()
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
# Vendor reference configuration for this fork on arm64 Android
# (OpenBMB MiniCPM-V-demo-Android CMakeLists): KleidiAI + OpenMP on.
if(ANDROID_ABI STREQUAL "arm64-v8a")
    set(GGML_CPU_KLEIDIAI ON CACHE BOOL "" FORCE)
    set(GGML_OPENMP ON CACHE BOOL "" FORCE)
endif()
add_subdirectory("${OMNI_ROOT}" upstream EXCLUDE_FROM_ALL)
# tools/mtmd/CMakeLists.txt reads LLAMA_INSTALL_VERSION, which the fork's
# root CMakeLists sets as a normal variable in its own directory scope.
# Upstream builds add mtmd from inside that scope; we add it from here, so
# the variable is not visible and set_target_properties() gets an empty
# VERSION (configure error at mtmd/CMakeLists.txt:48). Define it here,
# mirroring the root computation (0.0.<build number>).
if(NOT DEFINED LLAMA_INSTALL_VERSION)
    set(LLAMA_INSTALL_VERSION 0.0.0)
endif()
add_subdirectory("${OMNI_ROOT}/tools/mtmd" mtmd EXCLUDE_FROM_ALL)
add_library(minicpm_v47_bench SHARED benchmark.cpp)
target_include_directories(minicpm_v47_bench PRIVATE
    "${OMNI_ROOT}" "${OMNI_ROOT}/common" "${OMNI_ROOT}/include"
    "${OMNI_ROOT}/ggml/include" "${OMNI_ROOT}/tools/mtmd" "${OMNI_ROOT}/vendor")
target_link_libraries(minicpm_v47_bench PRIVATE mtmd llama android log)
target_link_options(minicpm_v47_bench PRIVATE "-Wl,-z,max-page-size=16384")
