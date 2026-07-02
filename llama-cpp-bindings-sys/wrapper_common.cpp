#include "wrapper_common.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <gsl/span>
#include <memory>
#include <new>
#include <regex>
#include <stdexcept>
#include <string>
#include <cstdint>

#include "llama.cpp/common/common.h"
#include "llama.cpp/common/json-schema-to-grammar.h"
#include "llama.cpp/include/llama.h"
#include <nlohmann/json.hpp> // IWYU pragma: keep
#include <nlohmann/json_fwd.hpp>
#include "wrapper_utils.h"

#include <vector>

extern "C" auto llama_rs_json_schema_to_grammar(
    const char * schema_json,
    bool force_gbnf,
    char ** out_grammar,
    char ** out_error) -> llama_rs_json_schema_to_grammar_status {
    if (out_grammar != nullptr) {
        *out_grammar = nullptr;
    }
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (schema_json == nullptr) {
        return LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_NULL_SCHEMA_JSON_ARG;
    }
    if (out_grammar == nullptr) {
        return LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_NULL_OUT_GRAMMAR_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_NULL_OUT_ERROR_ARG;
    }

    try {
        const auto schema = nlohmann::ordered_json::parse(schema_json);
        const auto grammar = json_schema_to_grammar(schema, force_gbnf);
        *out_grammar = llama_rs_dup_string(grammar);
        if (*out_grammar == nullptr) {
            return LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_OK;
    } catch (const std::invalid_argument & error) {
        return llama_rs_capture_message(out_error, error.what())
                   ? LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_INVALID_SCHEMA
                   : LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_ERROR_STRING_ALLOCATION_FAILED;
    } catch (...) {
        return llama_rs_capture_exception(
            out_error,
            LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_ERROR_STRING_ALLOCATION_FAILED,
            LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_THREW_CXX_EXCEPTION);
    }
}

extern "C" void llama_rs_string_free(char * ptr) {
    const std::unique_ptr<char[]> reclaimed(ptr);
}

extern "C" auto llama_rs_sampler_init_grammar(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    struct llama_sampler ** out_sampler,
    char ** out_error) -> llama_rs_sampler_init_grammar_status {
    if (out_sampler != nullptr) {
        *out_sampler = nullptr;
    }
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (out_sampler == nullptr) {
        return LLAMA_RS_SAMPLER_INIT_GRAMMAR_NULL_OUT_SAMPLER_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_SAMPLER_INIT_GRAMMAR_NULL_OUT_ERROR_ARG;
    }
    try {
        *out_sampler = llama_sampler_init_grammar(vocab, grammar_str, grammar_root);
        if (*out_sampler == nullptr) {
            return LLAMA_RS_SAMPLER_INIT_GRAMMAR_COMPILATION_FAILED;
        }
        return LLAMA_RS_SAMPLER_INIT_GRAMMAR_OK;
    } catch (...) {
        return llama_rs_capture_exception(
            out_error,
            LLAMA_RS_SAMPLER_INIT_GRAMMAR_ERROR_STRING_ALLOCATION_FAILED,
            LLAMA_RS_SAMPLER_INIT_GRAMMAR_THREW_CXX_EXCEPTION);
    }
}

extern "C" auto llama_rs_sampler_init_grammar_lazy(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_words,
    size_t num_trigger_words,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens,
    struct llama_sampler ** out_sampler,
    char ** out_error) -> llama_rs_sampler_init_grammar_lazy_status {
    if (out_sampler != nullptr) {
        *out_sampler = nullptr;
    }
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (out_sampler == nullptr) {
        return LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_NULL_OUT_SAMPLER_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_NULL_OUT_ERROR_ARG;
    }
    try {
        std::vector<std::string> trigger_patterns;
        trigger_patterns.reserve(num_trigger_words);
        const gsl::span<const char *> words(
            trigger_words, trigger_words != nullptr ? num_trigger_words : 0);
        for (const char * const word : words) {
            if ((word != nullptr) && *word != '\0') {
                trigger_patterns.push_back(regex_escape(word));
            }
        }
        std::vector<const char *> trigger_patterns_c(trigger_patterns.size());
        std::transform(
            trigger_patterns.begin(),
            trigger_patterns.end(),
            trigger_patterns_c.begin(),
            [](const std::string & pattern) -> const char * { return pattern.c_str(); });

        *out_sampler = llama_sampler_init_grammar_lazy_patterns(
            vocab,
            grammar_str,
            grammar_root,
            trigger_patterns_c.data(),
            trigger_patterns_c.size(),
            trigger_tokens,
            num_trigger_tokens);
        if (*out_sampler == nullptr) {
            return LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_COMPILATION_FAILED;
        }
        return LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_OK;
    } catch (...) {
        return llama_rs_capture_exception(
            out_error,
            LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_ERROR_STRING_ALLOCATION_FAILED,
            LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_THREW_CXX_EXCEPTION);
    }
}

extern "C" auto llama_rs_sampler_init_grammar_lazy_patterns(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_patterns,
    size_t num_trigger_patterns,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens,
    struct llama_sampler ** out_sampler,
    char ** out_error) -> llama_rs_sampler_init_grammar_lazy_patterns_status {
    if (out_sampler != nullptr) {
        *out_sampler = nullptr;
    }
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (out_sampler == nullptr) {
        return LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_NULL_OUT_SAMPLER_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_NULL_OUT_ERROR_ARG;
    }
    try {
        *out_sampler = llama_sampler_init_grammar_lazy_patterns(
            vocab,
            grammar_str,
            grammar_root,
            trigger_patterns,
            num_trigger_patterns,
            trigger_tokens,
            num_trigger_tokens);
        if (*out_sampler == nullptr) {
            return LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_COMPILATION_FAILED;
        }
        return LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_OK;
    } catch (const std::regex_error & error) {
        return llama_rs_capture_message(out_error, error.what())
                   ? LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_INVALID_TRIGGER_PATTERN
                   : LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_ERROR_STRING_ALLOCATION_FAILED;
    } catch (...) {
        return llama_rs_capture_exception(
            out_error,
            LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_ERROR_STRING_ALLOCATION_FAILED,
            LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_THREW_CXX_EXCEPTION);
    }
}

extern "C" auto llama_rs_memory_seq_pos_max(
    const struct llama_context * ctx,
    llama_seq_id seq_id) noexcept -> llama_pos {
    if (ctx == nullptr) {
        return -1;
    }
    auto * mem = llama_get_memory(ctx);
    if (mem == nullptr) {
        return -1;
    }
    uint32_t const n_seq_max = llama_n_seq_max(ctx);
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
        return -1;
    }

    return llama_memory_seq_pos_max(mem, seq_id);
}

extern "C" auto llama_rs_encode(
    struct llama_context * ctx,
    struct llama_batch batch,
    int32_t * out_return_code,
    char ** out_error) -> llama_rs_encode_status {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (out_return_code != nullptr) {
        *out_return_code = 0;
    }
    if (ctx == nullptr) {
        return LLAMA_RS_ENCODE_NULL_CTX_ARG;
    }
    try {
        const auto * model = llama_get_model(ctx);
        if (!llama_model_has_encoder(model)) {
            return LLAMA_RS_ENCODE_MODEL_HAS_NO_ENCODER;
        }
        int32_t const result = llama_encode(ctx, batch);
        if (result != 0) {
            if (out_return_code != nullptr) {
                *out_return_code = result;
            }
            if (result == -2) {
                return LLAMA_RS_ENCODE_OUT_OF_MEMORY;
            }
            if (result == -3) {
                return LLAMA_RS_ENCODE_COMPUTE_FAILED;
            }
            return LLAMA_RS_ENCODE_RETURNED_ERROR_CODE;
        }
        return LLAMA_RS_ENCODE_OK;
    } catch (...) {
        return llama_rs_capture_exception(
            out_error,
            LLAMA_RS_ENCODE_ERROR_STRING_ALLOCATION_FAILED,
            LLAMA_RS_ENCODE_THREW_CXX_EXCEPTION);
    }
}

extern "C" auto llama_rs_memory_seq_add(
    const struct llama_context * ctx,
    llama_seq_id seq_id,
    llama_pos pos_start,
    llama_pos pos_end,
    llama_pos shift,
    char ** out_error) -> llama_rs_memory_seq_add_status {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (ctx == nullptr) {
        return LLAMA_RS_MEMORY_SEQ_ADD_NULL_CTX_ARG;
    }
    try {
        const auto * model = llama_get_model(ctx);
        const auto rope = llama_model_rope_type(model);
        if (rope == LLAMA_ROPE_TYPE_MROPE || rope == LLAMA_ROPE_TYPE_VISION || rope == LLAMA_ROPE_TYPE_IMROPE) {
            return LLAMA_RS_MEMORY_SEQ_ADD_INCOMPATIBLE_ROPE_TYPE;
        }
        auto * mem = llama_get_memory(ctx);
        if (mem == nullptr) {
            return LLAMA_RS_MEMORY_SEQ_ADD_NULL_MEM;
        }
        llama_memory_seq_add(mem, seq_id, pos_start, pos_end, shift);
        return LLAMA_RS_MEMORY_SEQ_ADD_OK;
    } catch (...) {
        return llama_rs_capture_exception(
            out_error,
            LLAMA_RS_MEMORY_SEQ_ADD_ERROR_STRING_ALLOCATION_FAILED,
            LLAMA_RS_MEMORY_SEQ_ADD_THREW_CXX_EXCEPTION);
    }
}

extern "C" auto llama_rs_memory_seq_div(
    const struct llama_context * ctx,
    llama_seq_id seq_id,
    llama_pos pos_start,
    llama_pos pos_end,
    int divisor,
    char ** out_error) -> llama_rs_memory_seq_div_status {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (ctx == nullptr) {
        return LLAMA_RS_MEMORY_SEQ_DIV_NULL_CTX_ARG;
    }
    try {
        const auto * model = llama_get_model(ctx);
        const auto rope = llama_model_rope_type(model);
        if (rope == LLAMA_ROPE_TYPE_MROPE || rope == LLAMA_ROPE_TYPE_VISION || rope == LLAMA_ROPE_TYPE_IMROPE) {
            return LLAMA_RS_MEMORY_SEQ_DIV_INCOMPATIBLE_ROPE_TYPE;
        }
        auto * mem = llama_get_memory(ctx);
        if (mem == nullptr) {
            return LLAMA_RS_MEMORY_SEQ_DIV_NULL_MEM;
        }
        llama_memory_seq_div(mem, seq_id, pos_start, pos_end, divisor);
        return LLAMA_RS_MEMORY_SEQ_DIV_OK;
    } catch (...) {
        return llama_rs_capture_exception(
            out_error,
            LLAMA_RS_MEMORY_SEQ_DIV_ERROR_STRING_ALLOCATION_FAILED,
            LLAMA_RS_MEMORY_SEQ_DIV_THREW_CXX_EXCEPTION);
    }
}

extern "C" auto llama_rs_sampler_sample(
    struct llama_sampler * sampler,
    struct llama_context * ctx,
    int32_t idx,
    llama_token * out_token,
    char ** out_error) -> llama_rs_sampler_sample_status {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (sampler == nullptr) {
        return LLAMA_RS_SAMPLER_SAMPLE_NULL_SAMPLER_ARG;
    }
    if (ctx == nullptr) {
        return LLAMA_RS_SAMPLER_SAMPLE_NULL_CTX_ARG;
    }
    if (out_token == nullptr) {
        return LLAMA_RS_SAMPLER_SAMPLE_NULL_OUT_TOKEN_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_SAMPLER_SAMPLE_NULL_OUT_ERROR_ARG;
    }
    try {
        *out_token = llama_sampler_sample(sampler, ctx, idx);
        return LLAMA_RS_SAMPLER_SAMPLE_OK;
    } catch (...) {
        return llama_rs_capture_exception(
            out_error,
            LLAMA_RS_SAMPLER_SAMPLE_ERROR_STRING_ALLOCATION_FAILED,
            LLAMA_RS_SAMPLER_SAMPLE_THREW_CXX_EXCEPTION);
    }
}

extern "C" auto llama_rs_sampler_accept(
    struct llama_sampler * sampler,
    llama_token token,
    char ** out_error) -> llama_rs_sampler_accept_status {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (sampler == nullptr) {
        return LLAMA_RS_SAMPLER_ACCEPT_NULL_SAMPLER_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_SAMPLER_ACCEPT_NULL_OUT_ERROR_ARG;
    }
    try {
        llama_sampler_accept(sampler, token);
        return LLAMA_RS_SAMPLER_ACCEPT_OK;
    } catch (...) {
        return llama_rs_capture_exception(
            out_error,
            LLAMA_RS_SAMPLER_ACCEPT_ERROR_STRING_ALLOCATION_FAILED,
            LLAMA_RS_SAMPLER_ACCEPT_THREW_CXX_EXCEPTION);
    }
}

extern "C" auto llama_rs_load_model_from_file(
    const char * path,
    struct llama_model_params params,
    struct llama_model ** out_model,
    char ** out_error) -> llama_rs_load_model_from_file_status {
    if (out_model != nullptr) {
        *out_model = nullptr;
    }
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (path == nullptr) {
        return LLAMA_RS_LOAD_MODEL_FROM_FILE_NULL_PATH_ARG;
    }
    if (out_model == nullptr) {
        return LLAMA_RS_LOAD_MODEL_FROM_FILE_NULL_OUT_MODEL_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_LOAD_MODEL_FROM_FILE_NULL_OUT_ERROR_ARG;
    }
    try {
        *out_model = llama_model_load_from_file(path, params);
        if (*out_model == nullptr) {
            return LLAMA_RS_LOAD_MODEL_FROM_FILE_LOAD_FAILED;
        }
        return LLAMA_RS_LOAD_MODEL_FROM_FILE_OK;
    } catch (...) {
        return llama_rs_capture_exception(
            out_error,
            LLAMA_RS_LOAD_MODEL_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED,
            LLAMA_RS_LOAD_MODEL_FROM_FILE_THREW_CXX_EXCEPTION);
    }
}

extern "C" auto llama_rs_new_context_with_model(
    struct llama_model * model,
    struct llama_context_params params,
    struct llama_context ** out_ctx,
    char ** out_error) -> llama_rs_new_context_with_model_status {
    if (out_ctx != nullptr) {
        *out_ctx = nullptr;
    }
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (model == nullptr) {
        return LLAMA_RS_NEW_CONTEXT_WITH_MODEL_NULL_MODEL_ARG;
    }
    if (out_ctx == nullptr) {
        return LLAMA_RS_NEW_CONTEXT_WITH_MODEL_NULL_OUT_CTX_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_NEW_CONTEXT_WITH_MODEL_NULL_OUT_ERROR_ARG;
    }
    try {
        *out_ctx = llama_init_from_model(model, params);
        if (*out_ctx == nullptr) {
            return LLAMA_RS_NEW_CONTEXT_WITH_MODEL_CREATION_FAILED;
        }
        return LLAMA_RS_NEW_CONTEXT_WITH_MODEL_OK;
    } catch (...) {
        return llama_rs_capture_exception(
            out_error,
            LLAMA_RS_NEW_CONTEXT_WITH_MODEL_ERROR_STRING_ALLOCATION_FAILED,
            LLAMA_RS_NEW_CONTEXT_WITH_MODEL_THREW_CXX_EXCEPTION);
    }
}

extern "C" auto llama_rs_decode(
    struct llama_context * ctx,
    struct llama_batch batch,
    int32_t * out_return_code,
    char ** out_error) -> llama_rs_decode_status {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (out_return_code != nullptr) {
        *out_return_code = 0;
    }
    if (ctx == nullptr) {
        return LLAMA_RS_DECODE_NULL_CTX_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_DECODE_NULL_OUT_ERROR_ARG;
    }
    try {
        int32_t const result = llama_decode(ctx, batch);
        if (result != 0) {
            if (out_return_code != nullptr) {
                *out_return_code = result;
            }
            if (result == -2) {
                return LLAMA_RS_DECODE_OUT_OF_MEMORY;
            }
            if (result == -3) {
                return LLAMA_RS_DECODE_COMPUTE_FAILED;
            }
            return LLAMA_RS_DECODE_RETURNED_ERROR_CODE;
        }
        return LLAMA_RS_DECODE_OK;
    } catch (...) {
        return llama_rs_capture_exception(
            out_error,
            LLAMA_RS_DECODE_ERROR_STRING_ALLOCATION_FAILED,
            LLAMA_RS_DECODE_THREW_CXX_EXCEPTION);
    }
}

extern "C" auto llama_rs_tokenize(
    const struct llama_vocab * vocab,
    const char * text,
    int32_t text_len,
    llama_token * tokens,
    int32_t n_tokens_max,
    bool add_special,
    bool parse_special,
    int32_t * out_returned_count,
    char ** out_error) -> llama_rs_tokenize_status {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (out_returned_count != nullptr) {
        *out_returned_count = 0;
    }
    if (vocab == nullptr) {
        return LLAMA_RS_TOKENIZE_NULL_VOCAB_ARG;
    }
    if (text == nullptr) {
        return LLAMA_RS_TOKENIZE_NULL_TEXT_ARG;
    }
    if (out_returned_count == nullptr) {
        return LLAMA_RS_TOKENIZE_NULL_OUT_RETURNED_COUNT_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_TOKENIZE_NULL_OUT_ERROR_ARG;
    }
    try {
        int32_t const count = llama_tokenize(
            vocab, text, text_len, tokens, n_tokens_max, add_special, parse_special);
        *out_returned_count = count;
        return LLAMA_RS_TOKENIZE_OK;
    } catch (...) {
        return llama_rs_capture_exception(
            out_error,
            LLAMA_RS_TOKENIZE_ERROR_STRING_ALLOCATION_FAILED,
            LLAMA_RS_TOKENIZE_THREW_CXX_EXCEPTION);
    }
}

extern "C" auto llama_rs_sampler_apply(
    struct llama_sampler * sampler,
    struct llama_token_data_array * data_array,
    char ** out_error) -> llama_rs_sampler_apply_status {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (sampler == nullptr) {
        return LLAMA_RS_SAMPLER_APPLY_NULL_SAMPLER_ARG;
    }
    if (data_array == nullptr) {
        return LLAMA_RS_SAMPLER_APPLY_NULL_DATA_ARRAY_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_SAMPLER_APPLY_NULL_OUT_ERROR_ARG;
    }
    try {
        llama_sampler_apply(sampler, data_array);
        return LLAMA_RS_SAMPLER_APPLY_OK;
    } catch (...) {
        return llama_rs_capture_exception(
            out_error,
            LLAMA_RS_SAMPLER_APPLY_ERROR_STRING_ALLOCATION_FAILED,
            LLAMA_RS_SAMPLER_APPLY_THREW_CXX_EXCEPTION);
    }
}
