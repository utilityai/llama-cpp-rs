#pragma once

#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum llama_rs_apply_chat_template_status {
    LLAMA_RS_APPLY_CHAT_TEMPLATE_OK = 0,
    LLAMA_RS_APPLY_CHAT_TEMPLATE_NULL_MODEL_ARG,
    LLAMA_RS_APPLY_CHAT_TEMPLATE_NULL_TEMPLATE_ARG,
    LLAMA_RS_APPLY_CHAT_TEMPLATE_NULL_MESSAGES_ARG,
    LLAMA_RS_APPLY_CHAT_TEMPLATE_NULL_OUT_STRING_ARG,
    LLAMA_RS_APPLY_CHAT_TEMPLATE_NULL_OUT_ERROR_ARG,
    LLAMA_RS_APPLY_CHAT_TEMPLATE_MODEL_HAS_NO_VOCAB,
    LLAMA_RS_APPLY_CHAT_TEMPLATE_TEMPLATE_APPLICATION_FAILED,
    LLAMA_RS_APPLY_CHAT_TEMPLATE_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_APPLY_CHAT_TEMPLATE_VENDORED_THREW_CXX_EXCEPTION,
} llama_rs_apply_chat_template_status;

llama_rs_apply_chat_template_status llama_rs_apply_chat_template(
    const struct llama_model * model,
    const char * template_src,
    const char * const * roles,
    const char * const * contents,
    size_t n_messages,
    int add_generation_prompt,
    char ** out_string,
    char ** out_error);

#ifdef __cplusplus
}
#endif
