#include "llava_sampling.h"
#include "sampling.h"

struct llama_sampling_params *llama_sampling_params_default()
{
    return new llama_sampling_params;
}
void llama_sampling_params_free(struct llama_sampling_params *params)
{
    if (params)
    {
        delete params;
    }
}

struct llama_sampling_context *llama_sampling_context_init(const struct llama_sampling_params *params)
{
    return llama_sampling_init(*params);
}

void llama_sampling_context_free(struct llama_sampling_context *ctx)
{
    llama_sampling_free(ctx);
}

static bool eval_tokens(struct llama_context *ctx_llama, std::vector<llama_token> tokens, int n_batch, int *n_past)
{
    int N = (int)tokens.size();
    for (int i = 0; i < N; i += n_batch)
    {
        int n_eval = (int)tokens.size() - i;
        if (n_eval > n_batch)
        {
            n_eval = n_batch;
        }
        if (llama_decode(ctx_llama, llama_batch_get_one(&tokens[i], n_eval, *n_past, 0)))
        {
            fprintf(stderr, "%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, N, n_batch, *n_past);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

static bool eval_id(struct llama_context *ctx_llama, int id, int *n_past)
{
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens(ctx_llama, tokens, 1, n_past);
}

const char *llava_sample(struct llama_sampling_context *ctx_sampling,
                         struct llama_context *ctx_llama,
                         int *n_past)
{
    const llama_token id = llama_sampling_sample(ctx_sampling, ctx_llama, NULL);
    llama_sampling_accept(ctx_sampling, ctx_llama, id, true);
    static std::string ret;
    if (id == llama_token_eos(llama_get_model(ctx_llama)))
    {
        ret = "</s>";
    }
    else
    {
        ret = llama_token_to_piece(ctx_llama, id);
    }
    eval_id(ctx_llama, id, n_past);
    return ret.c_str();
}
