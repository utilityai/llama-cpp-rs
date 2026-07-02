#pragma once

#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum llama_rs_compute_tool_call_haystack_status {
    LLAMA_RS_COMPUTE_TOOL_CALL_HAYSTACK_OK = 0,
    LLAMA_RS_COMPUTE_TOOL_CALL_HAYSTACK_NULL_MODEL_ARG,
    LLAMA_RS_COMPUTE_TOOL_CALL_HAYSTACK_NULL_OUT_HAYSTACK_ARG,
    LLAMA_RS_COMPUTE_TOOL_CALL_HAYSTACK_NULL_OUT_ERROR_ARG,
    LLAMA_RS_COMPUTE_TOOL_CALL_HAYSTACK_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_COMPUTE_TOOL_CALL_HAYSTACK_THREW_CXX_EXCEPTION,
} llama_rs_compute_tool_call_haystack_status;

llama_rs_compute_tool_call_haystack_status llama_rs_compute_tool_call_haystack(
    const struct llama_model * model,
    char ** out_haystack,
    char ** out_error);

typedef enum llama_rs_diagnose_tool_call_synthetic_renders_status {
    LLAMA_RS_DIAGNOSE_TOOL_CALL_SYNTHETIC_RENDERS_OK = 0,
    LLAMA_RS_DIAGNOSE_TOOL_CALL_SYNTHETIC_RENDERS_NULL_MODEL_ARG,
    LLAMA_RS_DIAGNOSE_TOOL_CALL_SYNTHETIC_RENDERS_NULL_OUT_NO_TOOLS_ARG,
    LLAMA_RS_DIAGNOSE_TOOL_CALL_SYNTHETIC_RENDERS_NULL_OUT_WITH_TOOLS_ARG,
    LLAMA_RS_DIAGNOSE_TOOL_CALL_SYNTHETIC_RENDERS_NULL_OUT_ERROR_ARG,
    LLAMA_RS_DIAGNOSE_TOOL_CALL_SYNTHETIC_RENDERS_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_DIAGNOSE_TOOL_CALL_SYNTHETIC_RENDERS_THREW_CXX_EXCEPTION,
} llama_rs_diagnose_tool_call_synthetic_renders_status;

llama_rs_diagnose_tool_call_synthetic_renders_status llama_rs_diagnose_tool_call_synthetic_renders(
    const struct llama_model * model,
    char ** out_no_tools,
    char ** out_with_tools,
    char ** out_error);

#ifdef __cplusplus
}
#endif
