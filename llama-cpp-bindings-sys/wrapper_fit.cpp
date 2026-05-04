#include "wrapper_fit.h"

#include <exception>

#include "llama.cpp/common/fit.h"

extern "C" llama_rs_fit_status llama_rs_fit_params(
    const char * path_model,
    struct llama_model_params * mparams,
    struct llama_context_params * cparams,
    float * tensor_split,
    struct llama_model_tensor_buft_override * tensor_buft_overrides,
    size_t * margins,
    uint32_t n_ctx_min,
    enum ggml_log_level log_level) {
    try {
        const common_params_fit_status status = common_fit_params(
            path_model, mparams, cparams, tensor_split, tensor_buft_overrides,
            margins, n_ctx_min, log_level);
        switch (status) {
            case COMMON_PARAMS_FIT_STATUS_SUCCESS:
                return LLAMA_RS_FIT_STATUS_SUCCESS;
            case COMMON_PARAMS_FIT_STATUS_FAILURE:
                return LLAMA_RS_FIT_STATUS_FAILURE;
            case COMMON_PARAMS_FIT_STATUS_ERROR:
                return LLAMA_RS_FIT_STATUS_ERROR;
        }
        return LLAMA_RS_FIT_STATUS_ERROR;
    } catch (const std::exception &) {
        return LLAMA_RS_FIT_STATUS_ERROR;
    }
}
