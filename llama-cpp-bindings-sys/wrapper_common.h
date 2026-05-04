#pragma once

#include "llama.cpp/include/llama.h"

#include <stdbool.h>
#include <stddef.h>

struct llama_model;
struct llama_sampler;
struct llama_vocab;

#include "wrapper_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

llama_rs_status llama_rs_json_schema_to_grammar(
    const char * schema_json,
    bool force_gbnf,
    char ** out_grammar,
    char ** out_error);

struct llama_sampler * llama_rs_sampler_init_grammar(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    char ** out_error);

struct llama_sampler * llama_rs_sampler_init_grammar_lazy(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_words,
    size_t num_trigger_words,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens,
    char ** out_error);

struct llama_sampler * llama_rs_sampler_init_grammar_lazy_patterns(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_patterns,
    size_t num_trigger_patterns,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens,
    char ** out_error);

llama_rs_status llama_rs_sampler_accept(
    struct llama_sampler * sampler,
    llama_token token,
    char ** out_error);

llama_rs_status llama_rs_sampler_sample(
    struct llama_sampler * sampler,
    struct llama_context * ctx,
    int32_t idx,
    llama_token * out_token,
    char ** out_error);

void llama_rs_string_free(char * ptr);

llama_pos llama_rs_memory_seq_pos_max(
    struct llama_context * ctx,
    llama_seq_id seq_id);

llama_rs_status llama_rs_encode(
    struct llama_context * ctx,
    struct llama_batch batch);

llama_rs_status llama_rs_memory_seq_add(
    struct llama_context * ctx,
    llama_seq_id seq_id,
    llama_pos p0,
    llama_pos p1,
    llama_pos shift);

llama_rs_status llama_rs_memory_seq_div(
    struct llama_context * ctx,
    llama_seq_id seq_id,
    llama_pos p0,
    llama_pos p1,
    int d);

#ifdef __cplusplus
}
#endif
