#include "wrapper_common.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <new>
#include <string>
#include <stdint.h>

#include "llama.cpp/common/common.h"
#include "llama.cpp/common/json-schema-to-grammar.h"
#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#include <nlohmann/json.hpp>

extern "C" llama_rs_json_schema_to_grammar_status llama_rs_json_schema_to_grammar(
    const char * schema_json,
    bool force_gbnf,
    char ** out_grammar,
    char ** out_error) {
    if (out_grammar) {
        *out_grammar = nullptr;
    }
    if (out_error) {
        *out_error = nullptr;
    }
    if (!schema_json) {
        return LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_NULL_SCHEMA_JSON_ARG;
    }
    if (!out_grammar) {
        return LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_NULL_OUT_GRAMMAR_ARG;
    }
    if (!out_error) {
        return LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_NULL_OUT_ERROR_ARG;
    }

    try {
        const auto schema = nlohmann::ordered_json::parse(schema_json);
        const auto grammar = json_schema_to_grammar(schema, force_gbnf);
        *out_grammar = llama_rs_dup_string(grammar);
        if (!*out_grammar) {
            return LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        *out_error = llama_rs_dup_string(err.what());
        if (!*out_error) {
            return LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        *out_error = llama_rs_dup_string("unknown c++ exception");
        if (!*out_error) {
            return LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_VENDORED_THREW_CXX_EXCEPTION;
    }
}

extern "C" void llama_rs_string_free(char * ptr) {
    if (ptr) {
        std::free(ptr);
    }
}

extern "C" llama_rs_sampler_init_grammar_status llama_rs_sampler_init_grammar(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    struct llama_sampler ** out_sampler,
    char ** out_error) {
    if (out_sampler) {
        *out_sampler = nullptr;
    }
    if (out_error) {
        *out_error = nullptr;
    }
    if (!out_sampler) {
        return LLAMA_RS_SAMPLER_INIT_GRAMMAR_NULL_OUT_SAMPLER_ARG;
    }
    if (!out_error) {
        return LLAMA_RS_SAMPLER_INIT_GRAMMAR_NULL_OUT_ERROR_ARG;
    }
    try {
        *out_sampler = llama_sampler_init_grammar(vocab, grammar_str, grammar_root);
        if (!*out_sampler) {
            return LLAMA_RS_SAMPLER_INIT_GRAMMAR_VENDORED_RETURNED_NULL;
        }
        return LLAMA_RS_SAMPLER_INIT_GRAMMAR_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_SAMPLER_INIT_GRAMMAR_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        *out_error = llama_rs_dup_string(err.what());
        if (!*out_error) {
            return LLAMA_RS_SAMPLER_INIT_GRAMMAR_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_SAMPLER_INIT_GRAMMAR_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        *out_error = llama_rs_dup_string("unknown c++ exception");
        if (!*out_error) {
            return LLAMA_RS_SAMPLER_INIT_GRAMMAR_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_SAMPLER_INIT_GRAMMAR_VENDORED_THREW_CXX_EXCEPTION;
    }
}

extern "C" llama_rs_sampler_init_grammar_lazy_status llama_rs_sampler_init_grammar_lazy(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_words,
    size_t num_trigger_words,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens,
    struct llama_sampler ** out_sampler,
    char ** out_error) {
    if (out_sampler) {
        *out_sampler = nullptr;
    }
    if (out_error) {
        *out_error = nullptr;
    }
    if (!out_sampler) {
        return LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_NULL_OUT_SAMPLER_ARG;
    }
    if (!out_error) {
        return LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_NULL_OUT_ERROR_ARG;
    }
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

        *out_sampler = llama_sampler_init_grammar_lazy_patterns(
            vocab,
            grammar_str,
            grammar_root,
            trigger_patterns_c.data(),
            trigger_patterns_c.size(),
            trigger_tokens,
            num_trigger_tokens);
        if (!*out_sampler) {
            return LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_VENDORED_RETURNED_NULL;
        }
        return LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        *out_error = llama_rs_dup_string(err.what());
        if (!*out_error) {
            return LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        *out_error = llama_rs_dup_string("unknown c++ exception");
        if (!*out_error) {
            return LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_VENDORED_THREW_CXX_EXCEPTION;
    }
}

extern "C" llama_rs_sampler_init_grammar_lazy_patterns_status llama_rs_sampler_init_grammar_lazy_patterns(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_patterns,
    size_t num_trigger_patterns,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens,
    struct llama_sampler ** out_sampler,
    char ** out_error) {
    if (out_sampler) {
        *out_sampler = nullptr;
    }
    if (out_error) {
        *out_error = nullptr;
    }
    if (!out_sampler) {
        return LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_NULL_OUT_SAMPLER_ARG;
    }
    if (!out_error) {
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
        if (!*out_sampler) {
            return LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_VENDORED_RETURNED_NULL;
        }
        return LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        *out_error = llama_rs_dup_string(err.what());
        if (!*out_error) {
            return LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        *out_error = llama_rs_dup_string("unknown c++ exception");
        if (!*out_error) {
            return LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_VENDORED_THREW_CXX_EXCEPTION;
    }
}

extern "C" llama_pos llama_rs_memory_seq_pos_max(
    struct llama_context * ctx,
    llama_seq_id seq_id) {
    if (!ctx) {
        return -1;
    }
    try {
        auto * mem = llama_get_memory(ctx);
        if (!mem) {
            return -1;
        }
        uint32_t n_seq_max = llama_n_seq_max(ctx);
        if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
            return -1;
        }

        return llama_memory_seq_pos_max(mem, seq_id);
    } catch (...) {
        return -1;
    }
}

extern "C" llama_rs_encode_status llama_rs_encode(
    struct llama_context * ctx,
    struct llama_batch batch,
    int32_t * out_vendored_return_code,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (out_vendored_return_code) {
        *out_vendored_return_code = 0;
    }
    if (!ctx) {
        return LLAMA_RS_ENCODE_NULL_CTX_ARG;
    }
    try {
        const auto * model = llama_get_model(ctx);
        if (!llama_model_has_encoder(model)) {
            return LLAMA_RS_ENCODE_MODEL_HAS_NO_ENCODER;
        }
        int32_t result = llama_encode(ctx, batch);
        if (result != 0) {
            if (out_vendored_return_code) {
                *out_vendored_return_code = result;
            }
            return LLAMA_RS_ENCODE_VENDORED_RETURNED_NONZERO_CODE;
        }
        return LLAMA_RS_ENCODE_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_ENCODE_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        if (out_error) {
            *out_error = llama_rs_dup_string(err.what());
            if (!*out_error) {
                return LLAMA_RS_ENCODE_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_ENCODE_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        if (out_error) {
            *out_error = llama_rs_dup_string("unknown c++ exception");
            if (!*out_error) {
                return LLAMA_RS_ENCODE_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_ENCODE_VENDORED_THREW_CXX_EXCEPTION;
    }
}

extern "C" llama_rs_memory_seq_add_status llama_rs_memory_seq_add(
    struct llama_context * ctx,
    llama_seq_id seq_id,
    llama_pos p0,
    llama_pos p1,
    llama_pos shift,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!ctx) {
        return LLAMA_RS_MEMORY_SEQ_ADD_NULL_CTX_ARG;
    }
    try {
        const auto * model = llama_get_model(ctx);
        const auto rope = llama_model_rope_type(model);
        if (rope == LLAMA_ROPE_TYPE_MROPE || rope == LLAMA_ROPE_TYPE_VISION || rope == LLAMA_ROPE_TYPE_IMROPE) {
            return LLAMA_RS_MEMORY_SEQ_ADD_INCOMPATIBLE_ROPE_TYPE;
        }
        auto * mem = llama_get_memory(ctx);
        if (!mem) {
            return LLAMA_RS_MEMORY_SEQ_ADD_NULL_MEM;
        }
        llama_memory_seq_add(mem, seq_id, p0, p1, shift);
        return LLAMA_RS_MEMORY_SEQ_ADD_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_MEMORY_SEQ_ADD_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        if (out_error) {
            *out_error = llama_rs_dup_string(err.what());
            if (!*out_error) {
                return LLAMA_RS_MEMORY_SEQ_ADD_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_MEMORY_SEQ_ADD_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        if (out_error) {
            *out_error = llama_rs_dup_string("unknown c++ exception");
            if (!*out_error) {
                return LLAMA_RS_MEMORY_SEQ_ADD_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_MEMORY_SEQ_ADD_VENDORED_THREW_CXX_EXCEPTION;
    }
}

extern "C" llama_rs_memory_seq_div_status llama_rs_memory_seq_div(
    struct llama_context * ctx,
    llama_seq_id seq_id,
    llama_pos p0,
    llama_pos p1,
    int d,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!ctx) {
        return LLAMA_RS_MEMORY_SEQ_DIV_NULL_CTX_ARG;
    }
    try {
        const auto * model = llama_get_model(ctx);
        const auto rope = llama_model_rope_type(model);
        if (rope == LLAMA_ROPE_TYPE_MROPE || rope == LLAMA_ROPE_TYPE_VISION || rope == LLAMA_ROPE_TYPE_IMROPE) {
            return LLAMA_RS_MEMORY_SEQ_DIV_INCOMPATIBLE_ROPE_TYPE;
        }
        auto * mem = llama_get_memory(ctx);
        if (!mem) {
            return LLAMA_RS_MEMORY_SEQ_DIV_NULL_MEM;
        }
        llama_memory_seq_div(mem, seq_id, p0, p1, d);
        return LLAMA_RS_MEMORY_SEQ_DIV_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_MEMORY_SEQ_DIV_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        if (out_error) {
            *out_error = llama_rs_dup_string(err.what());
            if (!*out_error) {
                return LLAMA_RS_MEMORY_SEQ_DIV_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_MEMORY_SEQ_DIV_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        if (out_error) {
            *out_error = llama_rs_dup_string("unknown c++ exception");
            if (!*out_error) {
                return LLAMA_RS_MEMORY_SEQ_DIV_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_MEMORY_SEQ_DIV_VENDORED_THREW_CXX_EXCEPTION;
    }
}

extern "C" llama_rs_sampler_sample_status llama_rs_sampler_sample(
    struct llama_sampler * sampler,
    struct llama_context * ctx,
    int32_t idx,
    llama_token * out_token,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!sampler) {
        return LLAMA_RS_SAMPLER_SAMPLE_NULL_SAMPLER_ARG;
    }
    if (!ctx) {
        return LLAMA_RS_SAMPLER_SAMPLE_NULL_CTX_ARG;
    }
    if (!out_token) {
        return LLAMA_RS_SAMPLER_SAMPLE_NULL_OUT_TOKEN_ARG;
    }
    if (!out_error) {
        return LLAMA_RS_SAMPLER_SAMPLE_NULL_OUT_ERROR_ARG;
    }
    try {
        *out_token = llama_sampler_sample(sampler, ctx, idx);
        return LLAMA_RS_SAMPLER_SAMPLE_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_SAMPLER_SAMPLE_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        *out_error = llama_rs_dup_string(err.what());
        if (!*out_error) {
            return LLAMA_RS_SAMPLER_SAMPLE_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_SAMPLER_SAMPLE_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        *out_error = llama_rs_dup_string("unknown c++ exception");
        if (!*out_error) {
            return LLAMA_RS_SAMPLER_SAMPLE_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_SAMPLER_SAMPLE_VENDORED_THREW_CXX_EXCEPTION;
    }
}

extern "C" llama_rs_sampler_accept_status llama_rs_sampler_accept(
    struct llama_sampler * sampler,
    llama_token token,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!sampler) {
        return LLAMA_RS_SAMPLER_ACCEPT_NULL_SAMPLER_ARG;
    }
    if (!out_error) {
        return LLAMA_RS_SAMPLER_ACCEPT_NULL_OUT_ERROR_ARG;
    }
    try {
        llama_sampler_accept(sampler, token);
        return LLAMA_RS_SAMPLER_ACCEPT_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_SAMPLER_ACCEPT_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        *out_error = llama_rs_dup_string(err.what());
        if (!*out_error) {
            return LLAMA_RS_SAMPLER_ACCEPT_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_SAMPLER_ACCEPT_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        *out_error = llama_rs_dup_string("unknown c++ exception");
        if (!*out_error) {
            return LLAMA_RS_SAMPLER_ACCEPT_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_SAMPLER_ACCEPT_VENDORED_THREW_CXX_EXCEPTION;
    }
}
