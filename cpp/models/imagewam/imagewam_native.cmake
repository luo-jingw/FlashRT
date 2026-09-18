# ImageWAM native runtime (libflashrt_imagewam_native.so), included from the
# root CMakeLists.txt so it shares GPU_ARCH, GPU_GENCODE and the
# flash_rt_kernels compile flags.
#
# EXCLUDE_FROM_ALL: build it explicitly,
#   cmake --build build --target flashrt_imagewam_native
# It exports only the frt_imagewam_* C API (c_api.h); the model-runtime
# ABI headers are used for their struct definitions only (no link to
# libflashrt_runtime / libflashrt_exec).

set(_imagewam_native_dir ${CMAKE_CURRENT_LIST_DIR})

add_library(flashrt_imagewam_native SHARED EXCLUDE_FROM_ALL
  ${_imagewam_native_dir}/src/c_api.cpp
  ${_imagewam_native_dir}/src/native_runtime.cpp
  ${_imagewam_native_dir}/src/native_schema.cpp
  ${_imagewam_native_dir}/src/io_transforms.cpp
  ${_imagewam_native_dir}/src/proprio_projection.cpp
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
)
target_link_libraries(flashrt_imagewam_native PRIVATE
  CUDA::cudart
  CUDA::cublasLt
)
set_target_properties(flashrt_imagewam_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/flash_rt
  CXX_VISIBILITY_PRESET hidden
  CUDA_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  CUDA_STANDARD 17
  POSITION_INDEPENDENT_CODE ON
)
