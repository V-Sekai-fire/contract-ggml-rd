# web: ggml-rd on WebGPU, built with emscripten into one ES module (no
# pthreads, so a page needs no cross-origin isolation). The device layer is
# web/rd_compute_web.cpp in place of guest-runtime's RenderingDevice one; the
# kernels, packers and graph code are guest/ggml-rd unchanged.
#
#   emcmake cmake -S web -B <build> && cmake --build <build>   (web/build.nu does both and runs the test)
cmake_minimum_required(VERSION 3.19)
project(ggml_rd_web C CXX)

set(CMAKE_CXX_STANDARD 20)
get_filename_component(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
if(NOT DEFINED GGML_ROOT)
	get_filename_component(GGML_ROOT "${ROOT}/../ggml" ABSOLUTE)
endif()
if(NOT DEFINED GUEST_RUNTIME_ROOT)
	get_filename_component(GUEST_RUNTIME_ROOT "${ROOT}/../guest-runtime" ABSOLUTE)
endif()

set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${GGML_ROOT} ggml EXCLUDE_FROM_ALL)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_program(SLANGC slangc REQUIRED)

# The kernels as WGSL, in the table shape the SPIR-V one has: the web device's
# shader_from_spirv takes WGSL text.
file(GLOB SLANG CONFIGURE_DEPENDS ${ROOT}/kernels/ggml/slang/*.slang)
set(WGSL_DIR ${CMAKE_BINARY_DIR}/wgsl-ggml)
set(WGSL_OUT)
foreach(S ${SLANG})
	get_filename_component(K ${S} NAME_WE)
	add_custom_command(OUTPUT ${WGSL_DIR}/${K}.wgsl
		COMMAND ${CMAKE_COMMAND} -E make_directory ${WGSL_DIR}
		COMMAND ${SLANGC} ${S} -target wgsl -entry main -stage compute -o ${WGSL_DIR}/${K}.wgsl
		DEPENDS ${S})
	list(APPEND WGSL_OUT ${WGSL_DIR}/${K}.wgsl)
endforeach()
set(KERNELS_INC ${CMAKE_BINARY_DIR}/ggml_kernels.inc)
add_custom_command(OUTPUT ${KERNELS_INC}
	COMMAND ${CMAKE_COMMAND} -DSRC=${WGSL_DIR} -DOUT=${KERNELS_INC} -P ${CMAKE_CURRENT_SOURCE_DIR}/embed_wgsl.cmake
	DEPENDS ${WGSL_OUT} ${CMAKE_CURRENT_SOURCE_DIR}/embed_wgsl.cmake COMMENT "WGSL kernels into ggml_kernels.inc")

# The CPU fallback's runners (the same generator the host L2 uses).
set(HOST_KERNELS ${CMAKE_BINARY_DIR}/host_kernels.cpp)
file(GLOB EMITS CONFIGURE_DEPENDS ${ROOT}/kernels/ggml/cpp/*_emit.cpp)
add_custom_command(OUTPUT ${HOST_KERNELS}
	COMMAND ${Python3_EXECUTABLE} ${ROOT}/tests/ggml_rd_kernels/gen_host_kernels.py
		${ROOT}/kernels/ggml/kernels.txt ${HOST_KERNELS} ${ROOT}/kernels/ggml/cpp_siblings.txt
	DEPENDS ${ROOT}/tests/ggml_rd_kernels/gen_host_kernels.py ${ROOT}/kernels/ggml/kernels.txt ${EMITS})

file(GLOB OPS CONFIGURE_DEPENDS ${ROOT}/guest/ggml-rd/ops/*.cpp)
# OBJECT: the op packers register through static initializers, which a
# static archive would drop.
add_library(ggml_rd_web OBJECT
	rd_compute_web.cpp
	${ROOT}/guest/ggml-rd/ggml-rd.cpp
	${ROOT}/guest/ggml-rd/rd_graph.cpp
	${ROOT}/guest/ggml-rd/rd_kernels.cpp
	${ROOT}/guest/ggml-rd/rd_cpu.cpp
	${ROOT}/guest/ggml-rd/rd_pack.cpp
	${OPS}
	${HOST_KERNELS}
	${KERNELS_INC}
)
# web/ first: its rd_compute.h and api.hpp stand in for the sandbox's.
target_include_directories(ggml_rd_web PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}
	${ROOT}/guest/ggml-rd
	${ROOT}/kernels/ggml
	${ROOT}/kernels/ggml/cpp
	${ROOT}/tests/ggml_rd_kernels
	${CMAKE_BINARY_DIR}
	${GUEST_RUNTIME_ROOT}/guest/avbd/slang-rt
	${GGML_ROOT}/src
	${GUEST_RUNTIME_ROOT}/guest
)
set_source_files_properties(${HOST_KERNELS} PROPERTIES COMPILE_OPTIONS "-include;${CMAKE_CURRENT_SOURCE_DIR}/slang_platform_web.h")
target_link_libraries(ggml_rd_web PUBLIC ggml ggml-cpu ggml-base)
target_compile_options(ggml_rd_web PRIVATE -msimd128 -fexceptions)
target_compile_definitions(ggml_rd_web PUBLIC GGML_RD_WEB=1)

set(WEB_LINK
	-sASYNCIFY -sASYNCIFY_STACK_SIZE=1048576
	-sALLOW_MEMORY_GROWTH -sMAXIMUM_MEMORY=4GB -sSTACK_SIZE=4MB
	-sMODULARIZE -sEXPORT_ES6 -sENVIRONMENT=web,worker
	-sEXPORTED_RUNTIME_METHODS=ccall,HEAPF32,ENV
	-sDEFAULT_LIBRARY_FUNCS_TO_INCLUDE=$stringToNewUTF8,$UTF8ToString
	--pre-js ${CMAKE_CURRENT_SOURCE_DIR}/rd_webgpu.js)

add_executable(ggml_rd_web_test
	test/graph_web.cpp
	${ROOT}/guest/ggml_test/graph_nets.cpp
	${ROOT}/guest/ggml_test/app_graphs/app_graphs.cpp)
target_include_directories(ggml_rd_web_test PRIVATE ${ROOT}/guest/ggml_test)
target_compile_options(ggml_rd_web_test PRIVATE -msimd128 -fexceptions)
target_link_libraries(ggml_rd_web_test PRIVATE ggml_rd_web)
target_link_options(ggml_rd_web_test PRIVATE ${WEB_LINK} -sEXPORTED_FUNCTIONS=_malloc,_free,_web_main -fexceptions)
set_target_properties(ggml_rd_web_test PROPERTIES SUFFIX ".mjs")
