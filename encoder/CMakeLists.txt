# Copyright (c) the JPEG XL Project Authors.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

add_library(jxl_tiny STATIC EXCLUDE_FROM_ALL
  base/cache_aligned.cc
  base/data_parallel.cc
  base/padded_bytes.cc
  dct_scales.cc
  enc_ac_strategy.cc
  enc_adaptive_quantization.cc
  enc_bit_writer.cc
  enc_chroma_from_luma.cc
  enc_cluster.cc
  enc_entropy_code.cc
  enc_file.cc
  enc_frame.cc
  enc_group.cc
  enc_huffman_tree.cc
  enc_xyb.cc
  image.cc
  quant_weights.cc
  read_pfm.cc
  trace.cc
)
target_compile_options(jxl_tiny PUBLIC "${JPEGXL_INTERNAL_FLAGS}")
target_include_directories(jxl_tiny PUBLIC "${PROJECT_SOURCE_DIR}")
target_link_libraries(jxl_tiny hwy pthread)


add_executable(cjxl_tiny cjxl_main.cc)
target_link_libraries(cjxl_tiny jxl_tiny)

add_executable(jxl_tiny_trace jxl_tiny_trace_main.cc)
target_link_libraries(jxl_tiny_trace jxl_tiny)

if(BUILD_TESTING)
  find_program(PYTHON3_EXECUTABLE python3)
  if(PYTHON3_EXECUTABLE)
    execute_process(
      COMMAND ${PYTHON3_EXECUTABLE} -c "import numpy"
      RESULT_VARIABLE JPEGXL_TINY_NUMPY_RESULT
      OUTPUT_QUIET
      ERROR_QUIET)
    if(JPEGXL_TINY_NUMPY_RESULT EQUAL 0)
      add_test(
        NAME trace_smoke_test
        COMMAND ${PYTHON3_EXECUTABLE}
          ${PROJECT_SOURCE_DIR}/tools/trace_smoke_test.py
          --cjxl $<TARGET_FILE:cjxl_tiny>
          --trace $<TARGET_FILE:jxl_tiny_trace>
          --work-dir ${CMAKE_CURRENT_BINARY_DIR}/trace_smoke_test)
      add_test(
        NAME xyb_trace_parity_test
        COMMAND ${PYTHON3_EXECUTABLE}
          ${PROJECT_SOURCE_DIR}/tools/xyb_trace_parity_test.py
          --trace $<TARGET_FILE:jxl_tiny_trace>
          --work-dir ${CMAKE_CURRENT_BINARY_DIR}/xyb_trace_parity_test)
      add_test(
        NAME aq_trace_parity_test
        COMMAND ${PYTHON3_EXECUTABLE}
          ${PROJECT_SOURCE_DIR}/tools/aq_trace_parity_test.py
          --trace $<TARGET_FILE:jxl_tiny_trace>
          --work-dir ${CMAKE_CURRENT_BINARY_DIR}/aq_trace_parity_test)
      add_test(
        NAME group_trace_parity_test
        COMMAND ${PYTHON3_EXECUTABLE}
          ${PROJECT_SOURCE_DIR}/tools/group_trace_parity_test.py
          --trace $<TARGET_FILE:jxl_tiny_trace>
          --work-dir ${CMAKE_CURRENT_BINARY_DIR}/group_trace_parity_test)
      add_test(
        NAME token_trace_parity_test
        COMMAND ${PYTHON3_EXECUTABLE}
          ${PROJECT_SOURCE_DIR}/tools/token_trace_parity_test.py
          --trace $<TARGET_FILE:jxl_tiny_trace>
          --work-dir ${CMAKE_CURRENT_BINARY_DIR}/token_trace_parity_test)
      add_test(
        NAME entropy_code_trace_parity_test
        COMMAND ${PYTHON3_EXECUTABLE}
          ${PROJECT_SOURCE_DIR}/tools/entropy_code_trace_parity_test.py
          --trace $<TARGET_FILE:jxl_tiny_trace>
          --work-dir ${CMAKE_CURRENT_BINARY_DIR}/entropy_code_trace_parity_test
          --fixture-matrix)
      add_test(
        NAME section_trace_parity_test
        COMMAND ${PYTHON3_EXECUTABLE}
          ${PROJECT_SOURCE_DIR}/tools/section_trace_parity_test.py
          --trace $<TARGET_FILE:jxl_tiny_trace>
          --work-dir ${CMAKE_CURRENT_BINARY_DIR}/section_trace_parity_test
          --fixture-matrix)
      add_test(
        NAME codestream_trace_parity_test
        COMMAND ${PYTHON3_EXECUTABLE}
          ${PROJECT_SOURCE_DIR}/tools/codestream_trace_parity_test.py
          --trace $<TARGET_FILE:jxl_tiny_trace>
          --work-dir ${CMAKE_CURRENT_BINARY_DIR}/codestream_trace_parity_test
          --fixture-matrix)
      add_test(
        NAME encode_trace_parity_test
        COMMAND ${PYTHON3_EXECUTABLE}
          ${PROJECT_SOURCE_DIR}/tools/encode_trace_parity_test.py
          --trace $<TARGET_FILE:jxl_tiny_trace>
          --work-dir ${CMAKE_CURRENT_BINARY_DIR}/encode_trace_parity_test
          --fixture-matrix)
    else()
      message(STATUS "NumPy not found; skipping trace_smoke_test")
    endif()
  else()
    message(STATUS "python3 not found; skipping trace_smoke_test")
  endif()
endif()
