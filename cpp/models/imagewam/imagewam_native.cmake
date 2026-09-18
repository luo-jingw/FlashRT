# ImageWAM native runtime (libflashrt_imagewam_native.so), included from the
# root CMakeLists.txt so it shares GPU_ARCH, GPU_GENCODE and the
# flash_rt_kernels compile flags.
#
# EXCLUDE_FROM_ALL: build it explicitly,
#   cmake --build build --target flashrt_imagewam_native
# It exports only the frt_imagewam_* C API (c_api.h); the model-runtime
# ABI headers are used for their struct definitions only (no link to
# libflashrt_runtime / libflashrt_exec).
#
# The csrc kernels the native pipeline launches are compiled into this
# library from the same sources with the same CUDA flags as flash_rt_kernels,
# so both pipelines run identical device code. On SM100-class builds the
# flash_rt_fp4 objects (fp4_kernels_obj) are linked in for NVFP4 linears.

set(_imagewam_native_dir ${CMAKE_CURRENT_LIST_DIR})

add_library(flashrt_imagewam_native SHARED EXCLUDE_FROM_ALL
  ${_imagewam_native_dir}/src/c_api.cpp
  ${_imagewam_native_dir}/src/native_runtime.cpp
  ${_imagewam_native_dir}/src/native_schema.cpp
  ${_imagewam_native_dir}/src/native_pipeline.cpp
  ${_imagewam_native_dir}/src/fp4_linear.cpp
  ${_imagewam_native_dir}/src/io_transforms.cpp
  ${_imagewam_native_dir}/src/proprio_projection.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm/gemm_runner.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/norm.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/rope.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/activation.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/decoder_fused.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/elementwise.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/softmax.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels/attention_cublas.cu
)
# Host IO transforms must round like the float32 torch kernels they mirror:
# no fused multiply-add contraction.
set_source_files_properties(${_imagewam_native_dir}/src/io_transforms.cpp
  PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
target_include_directories(flashrt_imagewam_native PRIVATE
  ${_imagewam_native_dir}/include
  ${_imagewam_native_dir}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
  ${CMAKE_CURRENT_SOURCE_DIR}/exec/include
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gemm
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels
)
target_compile_options(flashrt_imagewam_native PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr -O3
    --ftz=true --prec-div=false --prec-sqrt=false
    ${GPU_GENCODE}
  >
)
target_link_libraries(flashrt_imagewam_native PRIVATE
  CUDA::cublas
  CUDA::cublasLt
  CUDA::cudart
)
if(ENABLE_SM100_CUTLASS)
  target_sources(flashrt_imagewam_native PRIVATE $<TARGET_OBJECTS:fp4_kernels_obj>)
  target_compile_definitions(flashrt_imagewam_native PRIVATE FLASHRT_IMAGEWAM_NATIVE_NVFP4=1)
endif()
set_target_properties(flashrt_imagewam_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt
  CXX_VISIBILITY_PRESET hidden
  CUDA_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  CUDA_STANDARD 17
  POSITION_INDEPENDENT_CODE ON
)
