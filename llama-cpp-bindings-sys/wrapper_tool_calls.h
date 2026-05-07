#pragma once

#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Render the model's chat template with the autoparser's standard tool-call
 * vs. plain-assistant synthetic turns and return the diff slice that surrounds
 * the tool-call payload. The returned haystack is the text that lives between
 * the model's tool-call open/close markers (with any reasoning prelude
 * stripped). Marker extraction from the haystack is performed in Rust.
 *
 * On success (LLAMA_RS_STATUS_OK):
 *   - If the model declares no tool-call markers (or an empty haystack),
 *     *out_haystack is left as nullptr.
 *   - Otherwise *out_haystack is a heap-allocated null-terminated string owned
 *     by the caller. Free via llama_rs_string_free.
 *
 * On LLAMA_RS_STATUS_EXCEPTION, *out_error is set to a heap-allocated message;
 * free via llama_rs_string_free.
 */
llama_rs_status llama_rs_compute_tool_call_haystack(
    const struct llama_model * model,
    char ** out_haystack,
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
