#pragma once

#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Detect the reasoning open/close marker strings for a model by analyzing its
 * Jinja chat template via llama.cpp's autoparser.
 *
 * On success (LLAMA_RS_STATUS_OK):
 *   - If the model has detected reasoning markers, *out_open and *out_close are
 *     set to heap-allocated null-terminated strings owned by the caller. Free
 *     each via llama_rs_string_free.
 *   - If no reasoning markers were detected, *out_open and *out_close are left
 *     as nullptr.
 *
 * On LLAMA_RS_STATUS_EXCEPTION, *out_error is set to a heap-allocated message;
 * free via llama_rs_string_free.
 */
llama_rs_status llama_rs_detect_reasoning_markers(
    const struct llama_model * model,
    char ** out_open,
    char ** out_close,
    char ** out_error);

#ifdef __cplusplus
}
#endif
