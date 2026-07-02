cmake_minimum_required(VERSION 3.14)

project(llama_cpp_ggml_sys C CXX ASM)

if(NOT DEFINED GGML_SOURCE_DIR)
    message(FATAL_ERROR "GGML_SOURCE_DIR must point at the vendored llama.cpp/ggml directory")
endif()

add_subdirectory(${GGML_SOURCE_DIR} ggml-build)
