# web/refs: the web test's references on the host (ggml-cpu, all cores).
#   cmake -S web/refs -B <build> -G Ninja && cmake --build <build>   (web/build.nu runs it)
cmake_minimum_required(VERSION 3.19)
project(ggml_rd_web_refs C CXX)

set(CMAKE_CXX_STANDARD 20)
get_filename_component(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
if(NOT DEFINED GGML_ROOT)
	get_filename_component(GGML_ROOT "${ROOT}/../ggml" ABSOLUTE)
endif()

# Not native, and on arm64 plain armv8-a (Apple's default target has f16
# vector arithmetic): ggml then accumulates f16 products in f32, as the
# oracle's precise mode does.
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
	set(GGML_CPU_ARM_ARCH "armv8-a" CACHE STRING "" FORCE)
endif()
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${GGML_ROOT} ggml EXCLUDE_FROM_ALL)

add_executable(ggml_rd_web_refs
	refs.cpp
	${ROOT}/guest/ggml_test/graph_nets.cpp
	${ROOT}/guest/ggml_test/app_graphs/app_graphs.cpp)
target_include_directories(ggml_rd_web_refs PRIVATE ${ROOT}/guest/ggml_test)
target_link_libraries(ggml_rd_web_refs PRIVATE ggml ggml-cpu ggml-base)
