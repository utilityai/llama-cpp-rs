#pragma once

#include "llama.cpp/ggml/include/ggml.h"
#include "llama.cpp/include/llama.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum llama_rs_fit_status {
    LLAMA_RS_FIT_STATUS_SUCCESS = 0,
    LLAMA_RS_FIT_STATUS_FAILURE = 1,
    LLAMA_RS_FIT_STATUS_ERROR   = 2,
} llama_rs_fit_status;

llama_rs_fit_status llama_rs_fit_params(
    const char * path_model,
    struct llama_model_params * mparams,
    struct llama_context_params * cparams,
    float * tensor_split,
    struct llama_model_tensor_buft_override * tensor_buft_overrides,
    size_t * margins,
    uint32_t n_ctx_min,
    enum ggml_log_level log_level);

#ifdef __cplusplus
}
#endif
