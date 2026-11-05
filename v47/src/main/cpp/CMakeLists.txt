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
    # Build 22: arch floor for ggml-cpu. With GGML_NATIVE off, the
    # fork's ggml-cpu CMake compiles the whole backend at plain
    # armv8-a unless GGML_CPU_ARM_ARCH is set, so the compile-time
    # feature macros the fork's KleidiAI selection reads
    # (__ARM_FEATURE_DOTPROD / __ARM_FEATURE_MATMUL_INT8 in
    # ggml-cpu.c) all read 0 and the loader log prints
    # "kleidiai: no compatible q4 kernels found for CPU features
    # mask 0" - the kernels never engage even on silicon that has
    # the features (the Fold6's Cortex-X4/A720 do). Setting the
    # floor to armv8.2-a+dotprod+i8mm makes this the ONLY -march
    # flag on the ggml-cpu target (the fork appends exactly
    # -march=${GGML_CPU_ARM_ARCH} in the non-native path).
    # CAVEAT: the ggml-cpu objects will SIGILL on a pre-armv8.2
    # phone. Acceptable for this single-device trial (Snapdragon
    # 8 Gen 3 is armv9.2-class); noted in the module README.
    set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+i8mm" CACHE STRING "" FORCE)
endif()
add_subdirectory("${OMNI_ROOT}" upstream EXCLUDE_FROM_ALL)
if(ANDROID)
    # Portability shim for the fork's LES slot runtime: llama-moe-slot.cpp
    # toggles the Darwin-only F_NOCACHE fcntl command on its experts fds.
    # That toggle is an advisory page-cache hint only (failures are logged,
    # reads do not depend on it) and it is reached only when the
    # LLAMA_MOE_F_NOCACHE env toggle is explicitly enabled. Bionic does not
    # define F_NOCACHE, so the fork does not compile for Android as pinned.
    # Darwin's value (48) is an unassigned fcntl command on Linux, so the
    # call fails cleanly with EINVAL if ever reached. Data is unaffected.
    target_compile_definitions(llama PRIVATE F_NOCACHE=48)
endif()
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
# Build 15: llama-common for the fork's speculative-decoding machinery
# (common/speculative.cpp: draft-dspark implementation, sampler and
# seq_rm probe helpers). LLAMA_BUILD_COMMON has always built it; the
# trial simply never linked it before.
target_link_libraries(minicpm_v47_bench PRIVATE mtmd llama llama-common android log)
target_link_options(minicpm_v47_bench PRIVATE "-Wl,-z,max-page-size=16384")
