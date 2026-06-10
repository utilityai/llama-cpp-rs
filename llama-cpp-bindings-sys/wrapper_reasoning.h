#pragma once

#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum llama_rs_detect_reasoning_markers_status {
    LLAMA_RS_DETECT_REASONING_MARKERS_OK = 0,
    LLAMA_RS_DETECT_REASONING_MARKERS_NULL_MODEL_ARG,
    LLAMA_RS_DETECT_REASONING_MARKERS_NULL_OUT_OPEN_ARG,
    LLAMA_RS_DETECT_REASONING_MARKERS_NULL_OUT_CLOSE_ARG,
    LLAMA_RS_DETECT_REASONING_MARKERS_NULL_OUT_ERROR_ARG,
    LLAMA_RS_DETECT_REASONING_MARKERS_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_DETECT_REASONING_MARKERS_VENDORED_THREW_CXX_EXCEPTION,
} llama_rs_detect_reasoning_markers_status;

llama_rs_detect_reasoning_markers_status llama_rs_detect_reasoning_markers(
    const struct llama_model * model,
    char ** out_open,
    char ** out_close,
    char ** out_error);

typedef enum llama_rs_render_chat_template_status {
    LLAMA_RS_RENDER_CHAT_TEMPLATE_OK = 0,
    LLAMA_RS_RENDER_CHAT_TEMPLATE_NULL_MODEL_ARG,
    LLAMA_RS_RENDER_CHAT_TEMPLATE_NULL_MESSAGES_ARG,
    LLAMA_RS_RENDER_CHAT_TEMPLATE_NULL_OUT_RENDERED_ARG,
    LLAMA_RS_RENDER_CHAT_TEMPLATE_NULL_OUT_ERROR_ARG,
    LLAMA_RS_RENDER_CHAT_TEMPLATE_MODEL_HAS_NO_CHAT_TEMPLATE,
    LLAMA_RS_RENDER_CHAT_TEMPLATE_MODEL_HAS_NO_VOCAB,
    LLAMA_RS_RENDER_CHAT_TEMPLATE_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_RENDER_CHAT_TEMPLATE_VENDORED_THREW_CXX_EXCEPTION,
} llama_rs_render_chat_template_status;

llama_rs_render_chat_template_status llama_rs_render_chat_template(
    const struct llama_model * model,
    const char * messages_json,
    int add_generation_prompt,
    int enable_thinking,
    char ** out_rendered,
    char ** out_error);

#ifdef __cplusplus
}
#endif
