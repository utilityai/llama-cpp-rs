#pragma once

#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Detect the tool-call section open/close marker strings for a model by
 * analyzing its Jinja chat template via llama.cpp's autoparser.
 *
 * On success (LLAMA_RS_STATUS_OK):
 *   - If the model has detected tool-call section markers, *out_open and
 *     *out_close are set to heap-allocated null-terminated strings owned by
 *     the caller. Free each via llama_rs_string_free.
 *   - If the model declares no tool-call markers (or an empty pair),
 *     *out_open and *out_close are left as nullptr.
 *
 * On LLAMA_RS_STATUS_EXCEPTION, *out_error is set to a heap-allocated message;
 * free via llama_rs_string_free.
 */
llama_rs_status llama_rs_detect_tool_call_markers(
    const struct llama_model * model,
    char ** out_open,
    char ** out_close,
    char ** out_error);

/**
 * Render the model's chat template with the autoparser's standard synthetic
 * inputs (assistant_no_tools vs assistant_with_tools). Useful for diagnosing
 * why marker detection fails.
 *
 * On success (LLAMA_RS_STATUS_OK):
 *   - *out_no_tools and *out_with_tools point to heap-allocated rendered
 *     outputs (free via llama_rs_string_free). Either can be empty when the
 *     template throws during rendering.
 *
 * On LLAMA_RS_STATUS_EXCEPTION, *out_error is set.
 */
llama_rs_status llama_rs_diagnose_tool_call_synthetic_renders(
    const struct llama_model * model,
    char ** out_no_tools,
    char ** out_with_tools,
    char ** out_error);

#ifdef __cplusplus
}
#endif
