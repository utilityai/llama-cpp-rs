#ifndef LLAVA_SAMPLING_H
#define LLAVA_SAMPLING_H

struct llama_sampling_params;
struct llama_sampling_context;

#ifdef __cplusplus
extern "C"
{
#endif
    struct llama_sampling_params *llama_sampling_params_default();
    void llama_sampling_params_free(struct llama_sampling_params *params);

    struct llama_sampling_context *llama_sampling_context_init(const struct llama_sampling_params *params);
    void llama_sampling_context_free(struct llama_sampling_context *ctx);

    const char *llava_sample(struct llama_sampling_context *ctx_sampling,
                             struct llama_context *ctx_llama,
                             int *n_past);

#ifdef __cplusplus
}
#endif
#endif // LLAVA_SAMPLING_H