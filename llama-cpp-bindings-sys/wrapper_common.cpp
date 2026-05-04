#include "wrapper_common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <stdint.h>

#include "llama.cpp/common/common.h"
#include "llama.cpp/common/json-schema-to-grammar.h"
#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#include <nlohmann/json.hpp>

extern "C" llama_rs_status llama_rs_json_schema_to_grammar(
    const char * schema_json,
    bool force_gbnf,
    char ** out_grammar,
    char ** out_error) {
    if (!schema_json || !out_grammar || !out_error) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    *out_grammar = nullptr;
    *out_error = nullptr;

    try {
        const auto schema = nlohmann::ordered_json::parse(schema_json);
        const auto grammar = json_schema_to_grammar(schema, force_gbnf);
        *out_grammar = llama_rs_dup_string(grammar);

        return *out_grammar ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        fprintf(stderr, "%s: C++ exception: %s\n", __func__, err.what());
        *out_error = llama_rs_dup_string(err.what());

        return LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        fprintf(stderr, "%s: unknown C++ exception\n", __func__);
        *out_error = llama_rs_dup_string("unknown C++ exception");

        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" void llama_rs_string_free(char * ptr) {
    if (ptr) {
        std::free(ptr);
    }
}

extern "C" struct llama_sampler * llama_rs_sampler_init_grammar(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    char ** out_error) {
    if (!out_error) {
        return nullptr;
    }

    *out_error = nullptr;

    try {
        return llama_sampler_init_grammar(vocab, grammar_str, grammar_root);
    } catch (const std::exception & err) {
        fprintf(stderr, "%s: C++ exception: %s\n", __func__, err.what());
        *out_error = llama_rs_dup_string(err.what());

        return nullptr;
    } catch (...) {
        fprintf(stderr, "%s: unknown C++ exception\n", __func__);
        *out_error = llama_rs_dup_string("unknown C++ exception");

        return nullptr;
    }
}

extern "C" struct llama_sampler * llama_rs_sampler_init_grammar_lazy(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_words,
    size_t num_trigger_words,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens,
    char ** out_error) {
    if (!out_error) {
        return nullptr;
    }

    *out_error = nullptr;

    try {
        std::vector<std::string> trigger_patterns;
        trigger_patterns.reserve(num_trigger_words);
        for (size_t i = 0; i < num_trigger_words; ++i) {
            const char * word = trigger_words ? trigger_words[i] : nullptr;
            if (word && word[0] != '\0') {
                trigger_patterns.push_back(regex_escape(word));
            }
        }
        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & pattern : trigger_patterns) {
            trigger_patterns_c.push_back(pattern.c_str());
        }

        return llama_sampler_init_grammar_lazy_patterns(
            vocab,
            grammar_str,
            grammar_root,
            trigger_patterns_c.data(),
            trigger_patterns_c.size(),
            trigger_tokens,
            num_trigger_tokens);
    } catch (const std::exception & err) {
        fprintf(stderr, "%s: C++ exception: %s\n", __func__, err.what());
        *out_error = llama_rs_dup_string(err.what());

        return nullptr;
    } catch (...) {
        fprintf(stderr, "%s: unknown C++ exception\n", __func__);
        *out_error = llama_rs_dup_string("unknown C++ exception");

        return nullptr;
    }
}

extern "C" struct llama_sampler * llama_rs_sampler_init_grammar_lazy_patterns(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_patterns,
    size_t num_trigger_patterns,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens,
    char ** out_error) {
    if (!out_error) {
        return nullptr;
    }

    *out_error = nullptr;

    try {
        return llama_sampler_init_grammar_lazy_patterns(
            vocab,
            grammar_str,
            grammar_root,
            trigger_patterns,
            num_trigger_patterns,
            trigger_tokens,
            num_trigger_tokens);
    } catch (const std::exception & err) {
        fprintf(stderr, "%s: C++ exception: %s\n", __func__, err.what());
        *out_error = llama_rs_dup_string(err.what());

        return nullptr;
    } catch (...) {
        fprintf(stderr, "%s: unknown C++ exception\n", __func__);
        *out_error = llama_rs_dup_string("unknown C++ exception");

        return nullptr;
    }
}

extern "C" llama_pos llama_rs_memory_seq_pos_max(
    struct llama_context * ctx,
    llama_seq_id seq_id) {
    if (!ctx) {
        return -1;
    }
    auto * mem = llama_get_memory(ctx);
    if (!mem) {
        return -1;
    }
    uint32_t n_seq_max = llama_n_seq_max(ctx);
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
        return -1;
    }

    return llama_memory_seq_pos_max(mem, seq_id);
}

extern "C" llama_rs_status llama_rs_encode(
    struct llama_context * ctx,
    struct llama_batch batch) {
    if (!ctx) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    const auto * model = llama_get_model(ctx);
    if (!llama_model_has_encoder(model)) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    int32_t result = llama_encode(ctx, batch);
    if (result != 0) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }

    return LLAMA_RS_STATUS_OK;
}

extern "C" llama_rs_status llama_rs_memory_seq_add(
    struct llama_context * ctx,
    llama_seq_id seq_id,
    llama_pos p0,
    llama_pos p1,
    llama_pos shift) {
    if (!ctx) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    const auto * model = llama_get_model(ctx);
    const auto rope = llama_model_rope_type(model);
    if (rope == LLAMA_ROPE_TYPE_MROPE || rope == LLAMA_ROPE_TYPE_VISION || rope == LLAMA_ROPE_TYPE_IMROPE) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    auto * mem = llama_get_memory(ctx);
    if (!mem) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    llama_memory_seq_add(mem, seq_id, p0, p1, shift);

    return LLAMA_RS_STATUS_OK;
}

extern "C" llama_rs_status llama_rs_memory_seq_div(
    struct llama_context * ctx,
    llama_seq_id seq_id,
    llama_pos p0,
    llama_pos p1,
    int d) {
    if (!ctx) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    const auto * model = llama_get_model(ctx);
    const auto rope = llama_model_rope_type(model);
    if (rope == LLAMA_ROPE_TYPE_MROPE || rope == LLAMA_ROPE_TYPE_VISION || rope == LLAMA_ROPE_TYPE_IMROPE) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    auto * mem = llama_get_memory(ctx);
    if (!mem) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    llama_memory_seq_div(mem, seq_id, p0, p1, d);

    return LLAMA_RS_STATUS_OK;
}

extern "C" llama_rs_status llama_rs_sampler_sample(
    struct llama_sampler * sampler,
    struct llama_context * ctx,
    int32_t idx,
    llama_token * out_token,
    char ** out_error) {
    if (!sampler || !ctx || !out_token || !out_error) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    *out_error = nullptr;

    try {
        *out_token = llama_sampler_sample(sampler, ctx, idx);

        return LLAMA_RS_STATUS_OK;
    } catch (const std::exception & err) {
        fprintf(stderr, "%s: C++ exception: %s\n", __func__, err.what());
        *out_error = llama_rs_dup_string(err.what());

        return LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        fprintf(stderr, "%s: unknown C++ exception\n", __func__);
        *out_error = llama_rs_dup_string("unknown C++ exception");

        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_sampler_accept(
    struct llama_sampler * sampler,
    llama_token token,
    char ** out_error) {
    if (!sampler || !out_error) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    *out_error = nullptr;

    try {
        llama_sampler_accept(sampler, token);

        return LLAMA_RS_STATUS_OK;
    } catch (const std::exception & err) {
        fprintf(stderr, "%s: C++ exception: %s\n", __func__, err.what());
        *out_error = llama_rs_dup_string(err.what());

        return LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        fprintf(stderr, "%s: unknown C++ exception\n", __func__);
        *out_error = llama_rs_dup_string("unknown C++ exception");

        return LLAMA_RS_STATUS_EXCEPTION;
    }
}
