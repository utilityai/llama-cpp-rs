#include "wrapper_common.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <random>
#include <string>
#include <stdint.h>
#include <vector>

#include "llama.cpp/common/chat.h"
#include "llama.cpp/common/common.h"
#include "llama.cpp/common/fit.h"
#include "llama.cpp/common/json-schema-to-grammar.h"
#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

extern "C" llama_rs_status llama_rs_json_schema_to_grammar(
    const char * schema_json,
    bool force_gbnf,
    char ** out_grammar) {
    if (!schema_json || !out_grammar) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    *out_grammar = nullptr;
    try {
        const auto schema = nlohmann::ordered_json::parse(schema_json);
        const auto grammar = json_schema_to_grammar(schema, force_gbnf);
        *out_grammar = llama_rs_dup_string(grammar);
        return *out_grammar ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_ALLOCATION_FAILED;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" void llama_rs_string_free(char * ptr) {
    if (ptr) {
        std::free(ptr);
    }
}

extern "C" void llama_rs_chat_template_result_free(struct llama_rs_chat_template_result * result) {
    if (!result) {
        return;
    }
    std::free(result->prompt);
    std::free(result->grammar);
    std::free(result->parser);
    std::free(result->generation_prompt);
    if (result->grammar_triggers) {
        for (size_t i = 0; i < result->grammar_triggers_count; ++i) {
            std::free(result->grammar_triggers[i].value);
        }
        std::free(result->grammar_triggers);
    }
    if (result->preserved_tokens) {
        for (size_t i = 0; i < result->preserved_tokens_count; ++i) {
            std::free(result->preserved_tokens[i]);
        }
        std::free(result->preserved_tokens);
    }
    if (result->additional_stops) {
        for (size_t i = 0; i < result->additional_stops_count; ++i) {
            std::free(result->additional_stops[i]);
        }
        std::free(result->additional_stops);
    }
    result->prompt = nullptr;
    result->grammar = nullptr;
    result->parser = nullptr;
    result->generation_prompt = nullptr;
    result->chat_format = 0;
    result->grammar_lazy = false;
    result->grammar_triggers = nullptr;
    result->grammar_triggers_count = 0;
    result->preserved_tokens = nullptr;
    result->preserved_tokens_count = 0;
    result->additional_stops = nullptr;
    result->additional_stops_count = 0;
}

static std::string random_string(size_t length = 32) {
    static constexpr char chars[] =
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    static constexpr size_t chars_len = sizeof(chars) - 1;
    static thread_local std::mt19937 generator([]() {
        std::random_device rd;
        std::seed_seq seed{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
        return std::mt19937(seed);
    }());

    std::uniform_int_distribution<size_t> distribution(0, chars_len - 1);
    std::string result(length, '\0');
    for (size_t i = 0; i < length; ++i) {
        result[i] = chars[distribution(generator)];
    }
    return result;
}

struct llama_rs_chat_parse_state_oaicompat {
    common_chat_parser_params syntax;
    common_chat_msg chat_msg;
    std::string generated_text;
    std::vector<std::string> generated_tool_call_ids;

    explicit llama_rs_chat_parse_state_oaicompat(common_chat_parser_params syntax_in)
        : syntax(std::move(syntax_in)) {}
};

static void init_chat_msg(struct llama_rs_chat_msg_oaicompat * out_msg) {
    if (!out_msg) {
        return;
    }
    out_msg->role = nullptr;
    out_msg->content = nullptr;
    out_msg->content_parts = nullptr;
    out_msg->content_parts_count = 0;
    out_msg->reasoning_content = nullptr;
    out_msg->tool_name = nullptr;
    out_msg->tool_call_id = nullptr;
    out_msg->tool_calls = nullptr;
    out_msg->tool_calls_count = 0;
}

static llama_rs_status dup_string_array(
    const std::vector<std::string> & values,
    char *** out_items,
    size_t * out_count) {
    if (!out_items || !out_count) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *out_items = nullptr;
    *out_count = 0;
    if (values.empty()) {
        return LLAMA_RS_STATUS_OK;
    }

    auto ** items = static_cast<char **>(std::calloc(values.size(), sizeof(char *)));
    if (!items) {
        return LLAMA_RS_STATUS_ALLOCATION_FAILED;
    }
    for (size_t i = 0; i < values.size(); ++i) {
        items[i] = llama_rs_dup_string(values[i]);
        if (!items[i] && !values[i].empty()) {
            for (size_t j = 0; j <= i; ++j) {
                std::free(items[j]);
            }
            std::free(items);
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
    }
    *out_items = items;
    *out_count = values.size();
    return LLAMA_RS_STATUS_OK;
}

static llama_rs_status dup_trigger_array(
    const std::vector<common_grammar_trigger> & triggers,
    struct llama_rs_grammar_trigger ** out_items,
    size_t * out_count) {
    if (!out_items || !out_count) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *out_items = nullptr;
    *out_count = 0;
    if (triggers.empty()) {
        return LLAMA_RS_STATUS_OK;
    }

    auto * items = static_cast<struct llama_rs_grammar_trigger *>(
        std::calloc(triggers.size(), sizeof(struct llama_rs_grammar_trigger)));
    if (!items) {
        return LLAMA_RS_STATUS_ALLOCATION_FAILED;
    }
    for (size_t i = 0; i < triggers.size(); ++i) {
        items[i].type = static_cast<int>(triggers[i].type);
        items[i].token = triggers[i].token;
        items[i].value = llama_rs_dup_string(triggers[i].value);
        if (!items[i].value && !triggers[i].value.empty()) {
            for (size_t j = 0; j <= i; ++j) {
                std::free(items[j].value);
            }
            std::free(items);
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
    }
    *out_items = items;
    *out_count = triggers.size();
    return LLAMA_RS_STATUS_OK;
}

static llama_rs_status fill_template_result(
    const common_chat_params & params,
    struct llama_rs_chat_template_result * out_result) {
    out_result->prompt = llama_rs_dup_string(params.prompt);
    if (!out_result->prompt) {
        return LLAMA_RS_STATUS_ALLOCATION_FAILED;
    }
    if (!params.grammar.empty()) {
        out_result->grammar = llama_rs_dup_string(params.grammar);
        if (!out_result->grammar) {
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
    }
    if (!params.parser.empty()) {
        out_result->parser = llama_rs_dup_string(params.parser);
        if (!out_result->parser) {
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
    }
    if (!params.generation_prompt.empty()) {
        out_result->generation_prompt = llama_rs_dup_string(params.generation_prompt);
        if (!out_result->generation_prompt) {
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
    }
    out_result->chat_format = static_cast<int>(params.format);
    out_result->grammar_lazy = params.grammar_lazy;

    auto status = dup_trigger_array(
        params.grammar_triggers,
        &out_result->grammar_triggers,
        &out_result->grammar_triggers_count);
    if (status != LLAMA_RS_STATUS_OK) {
        return status;
    }
    status = dup_string_array(
        params.preserved_tokens,
        &out_result->preserved_tokens,
        &out_result->preserved_tokens_count);
    if (status != LLAMA_RS_STATUS_OK) {
        return status;
    }
    return dup_string_array(
        params.additional_stops,
        &out_result->additional_stops,
        &out_result->additional_stops_count);
}

extern "C" struct llama_sampler * llama_rs_sampler_init_grammar(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root) {
    try {
        return llama_sampler_init_grammar(vocab, grammar_str, grammar_root);
    } catch (...) {
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
    size_t num_trigger_tokens) {
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
    } catch (...) {
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
    size_t num_trigger_tokens) {
    try {
        return llama_sampler_init_grammar_lazy_patterns(
            vocab,
            grammar_str,
            grammar_root,
            trigger_patterns,
            num_trigger_patterns,
            trigger_tokens,
            num_trigger_tokens);
    } catch (...) {
        return nullptr;
    }
}

extern "C" llama_rs_status llama_rs_sampler_accept(struct llama_sampler * sampler, llama_token token) {
    if (!sampler) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    try {
        llama_sampler_accept(sampler, token);
        return LLAMA_RS_STATUS_OK;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

static llama_rs_status dup_content_parts(
    const std::vector<common_chat_msg_content_part> & parts,
    struct llama_rs_chat_msg_content_part_oaicompat ** out_items,
    size_t * out_count) {
    *out_items = nullptr;
    *out_count = 0;
    if (parts.empty()) {
        return LLAMA_RS_STATUS_OK;
    }

    auto * items = static_cast<struct llama_rs_chat_msg_content_part_oaicompat *>(
        std::calloc(parts.size(), sizeof(struct llama_rs_chat_msg_content_part_oaicompat)));
    if (!items) {
        return LLAMA_RS_STATUS_ALLOCATION_FAILED;
    }
    for (size_t i = 0; i < parts.size(); ++i) {
        items[i].type = llama_rs_dup_string(parts[i].type);
        items[i].text = llama_rs_dup_string(parts[i].text);
        if ((!items[i].type && !parts[i].type.empty()) || (!items[i].text && !parts[i].text.empty())) {
            for (size_t j = 0; j <= i; ++j) {
                std::free(items[j].type);
                std::free(items[j].text);
            }
            std::free(items);
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
    }
    *out_items = items;
    *out_count = parts.size();
    return LLAMA_RS_STATUS_OK;
}

static llama_rs_status fill_chat_msg(
    const common_chat_msg & msg,
    struct llama_rs_chat_msg_oaicompat * out_msg) {
    init_chat_msg(out_msg);
    if (!msg.role.empty()) {
        out_msg->role = llama_rs_dup_string(msg.role);
        if (!out_msg->role) {
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
    }
    if (!msg.content.empty()) {
        out_msg->content = llama_rs_dup_string(msg.content);
        if (!out_msg->content) {
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
    }
    if (!msg.content_parts.empty()) {
        auto status = dup_content_parts(msg.content_parts, &out_msg->content_parts, &out_msg->content_parts_count);
        if (status != LLAMA_RS_STATUS_OK) {
            return status;
        }
    }
    if (!msg.reasoning_content.empty()) {
        out_msg->reasoning_content = llama_rs_dup_string(msg.reasoning_content);
        if (!out_msg->reasoning_content) {
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
    }
    if (!msg.tool_name.empty()) {
        out_msg->tool_name = llama_rs_dup_string(msg.tool_name);
        if (!out_msg->tool_name) {
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
    }
    if (!msg.tool_call_id.empty()) {
        out_msg->tool_call_id = llama_rs_dup_string(msg.tool_call_id);
        if (!out_msg->tool_call_id) {
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
    }
    if (!msg.tool_calls.empty()) {
        auto * calls = static_cast<struct llama_rs_tool_call_oaicompat *>(
            std::calloc(msg.tool_calls.size(), sizeof(struct llama_rs_tool_call_oaicompat)));
        if (!calls) {
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
        for (size_t i = 0; i < msg.tool_calls.size(); ++i) {
            calls[i].name = llama_rs_dup_string(msg.tool_calls[i].name);
            calls[i].arguments = llama_rs_dup_string(msg.tool_calls[i].arguments);
            calls[i].id = llama_rs_dup_string(msg.tool_calls[i].id);
            if ((!calls[i].name && !msg.tool_calls[i].name.empty()) ||
                (!calls[i].arguments && !msg.tool_calls[i].arguments.empty()) ||
                (!calls[i].id && !msg.tool_calls[i].id.empty())) {
                for (size_t j = 0; j <= i; ++j) {
                    std::free(calls[j].name);
                    std::free(calls[j].arguments);
                    std::free(calls[j].id);
                }
                std::free(calls);
                return LLAMA_RS_STATUS_ALLOCATION_FAILED;
            }
        }
        out_msg->tool_calls = calls;
        out_msg->tool_calls_count = msg.tool_calls.size();
    }
    return LLAMA_RS_STATUS_OK;
}

extern "C" llama_rs_status llama_rs_apply_chat_template_oaicompat(
    const struct llama_model * model,
    const char * chat_template,
    const struct llama_rs_chat_template_oaicompat_params * params,
    struct llama_rs_chat_template_result * out_result) {
    if (!chat_template || !params || !params->messages || !out_result) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    out_result->prompt = nullptr;
    out_result->grammar = nullptr;
    out_result->parser = nullptr;
    out_result->generation_prompt = nullptr;
    out_result->chat_format = 0;
    out_result->grammar_lazy = false;
    out_result->grammar_triggers = nullptr;
    out_result->grammar_triggers_count = 0;
    out_result->preserved_tokens = nullptr;
    out_result->preserved_tokens_count = 0;
    out_result->additional_stops = nullptr;
    out_result->additional_stops_count = 0;

    try {
        auto tmpls = common_chat_templates_init(model, chat_template);
        common_chat_templates_inputs inputs;
        inputs.add_generation_prompt = params->add_generation_prompt;
        inputs.use_jinja = params->use_jinja;
        inputs.parallel_tool_calls = params->parallel_tool_calls;
        inputs.enable_thinking = params->enable_thinking;
        inputs.add_bos = params->add_bos;
        inputs.add_eos = params->add_eos;

        inputs.messages = common_chat_msgs_parse_oaicompat(json::parse(params->messages));
        if (params->tools && std::strlen(params->tools) > 0) {
            inputs.tools = common_chat_tools_parse_oaicompat(json::parse(params->tools));
        }
        if (params->tool_choice && std::strlen(params->tool_choice) > 0) {
            inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(params->tool_choice);
        }
        if (params->json_schema && std::strlen(params->json_schema) > 0) {
            inputs.json_schema = params->json_schema;
        }
        if (params->grammar && std::strlen(params->grammar) > 0) {
            inputs.grammar = params->grammar;
        }
        if (params->reasoning_format && std::strlen(params->reasoning_format) > 0) {
            inputs.reasoning_format = common_reasoning_format_from_name(params->reasoning_format);
        }
        if (params->chat_template_kwargs && std::strlen(params->chat_template_kwargs) > 0) {
            auto kwargs = json::parse(params->chat_template_kwargs);
            for (const auto & item : kwargs.items()) {
                inputs.chat_template_kwargs[item.key()] = item.value().dump();
            }
        }

        auto params_out = common_chat_templates_apply(tmpls.get(), inputs);
        auto status = fill_template_result(params_out, out_result);
        if (status != LLAMA_RS_STATUS_OK) {
            llama_rs_chat_template_result_free(out_result);
        }
        return status;
    } catch (const std::exception &) {
        llama_rs_chat_template_result_free(out_result);
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_chat_parse_to_oaicompat(
    const char * input,
    bool is_partial,
    int chat_format,
    bool parse_tool_calls,
    const char * parser_data,
    const char * generation_prompt,
    char ** out_json) {
    if (!input || !out_json) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *out_json = nullptr;

    try {
        common_chat_parser_params syntax;
        syntax.format = static_cast<common_chat_format>(chat_format);
        syntax.parse_tool_calls = parse_tool_calls;
        if (generation_prompt && std::strlen(generation_prompt) > 0) {
            syntax.generation_prompt = generation_prompt;
        }
        if (parser_data && std::strlen(parser_data) > 0) {
            syntax.parser.load(parser_data);
        }

        auto msg = common_chat_parse(input, is_partial, syntax);
        std::vector<std::string> ids_cache;
        msg.set_tool_call_ids(ids_cache, []() { return random_string(); });
        auto json_msg = msg.to_json_oaicompat().dump();
        *out_json = llama_rs_dup_string(json_msg);
        return *out_json ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_ALLOCATION_FAILED;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" struct llama_rs_chat_parse_state_oaicompat * llama_rs_chat_parse_state_init_oaicompat(
    int chat_format,
    bool parse_tool_calls,
    const char * parser_data,
    const char * generation_prompt) {
    try {
        common_chat_parser_params syntax;
        syntax.format = static_cast<common_chat_format>(chat_format);
        syntax.parse_tool_calls = parse_tool_calls;
        if (generation_prompt && std::strlen(generation_prompt) > 0) {
            syntax.generation_prompt = generation_prompt;
        }
        if (parser_data && std::strlen(parser_data) > 0) {
            syntax.parser.load(parser_data);
        }
        return new llama_rs_chat_parse_state_oaicompat(std::move(syntax));
    } catch (const std::exception &) {
        return nullptr;
    }
}

extern "C" llama_rs_status llama_rs_chat_parse_state_update_oaicompat(
    struct llama_rs_chat_parse_state_oaicompat * state,
    const char * text_added,
    bool is_partial,
    struct llama_rs_chat_msg_oaicompat * out_msg,
    struct llama_rs_chat_msg_diff_oaicompat ** out_diffs,
    size_t * out_diffs_count) {
    if (!state || !out_msg || !out_diffs || !out_diffs_count) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *out_diffs = nullptr;
    *out_diffs_count = 0;
    init_chat_msg(out_msg);

    try {
        if (text_added && text_added[0] != '\0') {
            state->generated_text += text_added;
        }
        auto msg_prv_copy = state->chat_msg;
        auto new_msg = common_chat_parse(state->generated_text, is_partial, state->syntax);
        std::vector<common_chat_msg_diff> diffs;
        if (!new_msg.empty()) {
            new_msg.set_tool_call_ids(state->generated_tool_call_ids, []() { return random_string(); });
            state->chat_msg = new_msg;
            diffs = common_chat_msg_diff::compute_diffs(msg_prv_copy, state->chat_msg);
        }

        auto status = fill_chat_msg(state->chat_msg, out_msg);
        if (status != LLAMA_RS_STATUS_OK) {
            llama_rs_chat_msg_free_oaicompat(out_msg);
            return status;
        }

        if (!diffs.empty()) {
            auto * diff_arr = static_cast<struct llama_rs_chat_msg_diff_oaicompat *>(
                std::calloc(diffs.size(), sizeof(struct llama_rs_chat_msg_diff_oaicompat)));
            if (!diff_arr) {
                llama_rs_chat_msg_free_oaicompat(out_msg);
                return LLAMA_RS_STATUS_ALLOCATION_FAILED;
            }
            for (size_t i = 0; i < diffs.size(); ++i) {
                diff_arr[i].tool_call_index = diffs[i].tool_call_index;
                if (!diffs[i].reasoning_content_delta.empty()) {
                    diff_arr[i].reasoning_content_delta = llama_rs_dup_string(diffs[i].reasoning_content_delta);
                }
                if (!diffs[i].content_delta.empty()) {
                    diff_arr[i].content_delta = llama_rs_dup_string(diffs[i].content_delta);
                }
                if (diffs[i].tool_call_index != std::string::npos) {
                    if (!diffs[i].tool_call_delta.name.empty()) {
                        diff_arr[i].tool_call_delta.name = llama_rs_dup_string(diffs[i].tool_call_delta.name);
                    }
                    if (!diffs[i].tool_call_delta.arguments.empty()) {
                        diff_arr[i].tool_call_delta.arguments = llama_rs_dup_string(diffs[i].tool_call_delta.arguments);
                    }
                    if (!diffs[i].tool_call_delta.id.empty()) {
                        diff_arr[i].tool_call_delta.id = llama_rs_dup_string(diffs[i].tool_call_delta.id);
                    }
                }
            }
            *out_diffs = diff_arr;
            *out_diffs_count = diffs.size();
        }
        return LLAMA_RS_STATUS_OK;
    } catch (const std::exception &) {
        llama_rs_chat_msg_free_oaicompat(out_msg);
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_chat_msg_diff_to_oaicompat_json(
    const struct llama_rs_chat_msg_diff_oaicompat * diff,
    char ** out_json) {
    if (!diff || !out_json) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *out_json = nullptr;

    try {
        json delta = json::object();
        if (diff->reasoning_content_delta && diff->reasoning_content_delta[0] != '\0') {
            delta["reasoning_content"] = diff->reasoning_content_delta;
        }
        if (diff->content_delta && diff->content_delta[0] != '\0') {
            delta["content"] = diff->content_delta;
        }
        if (diff->tool_call_index != std::string::npos) {
            auto tool_call = json::object();
            tool_call["index"] = diff->tool_call_index;
            if (diff->tool_call_delta.id && diff->tool_call_delta.id[0] != '\0') {
                tool_call["id"] = diff->tool_call_delta.id;
            }
            tool_call["type"] = "function";
            auto function = json::object();
            if (diff->tool_call_delta.name && diff->tool_call_delta.name[0] != '\0') {
                function["name"] = diff->tool_call_delta.name;
            }
            if (diff->tool_call_delta.arguments && diff->tool_call_delta.arguments[0] != '\0') {
                function["arguments"] = diff->tool_call_delta.arguments;
            }
            tool_call["function"] = function;
            delta["tool_calls"] = json::array({tool_call});
        }
        *out_json = llama_rs_dup_string(delta.dump());
        return *out_json ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_ALLOCATION_FAILED;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" void llama_rs_chat_msg_free_oaicompat(struct llama_rs_chat_msg_oaicompat * msg) {
    if (!msg) {
        return;
    }
    std::free(msg->role);
    std::free(msg->content);
    if (msg->content_parts) {
        for (size_t i = 0; i < msg->content_parts_count; ++i) {
            std::free(msg->content_parts[i].type);
            std::free(msg->content_parts[i].text);
        }
        std::free(msg->content_parts);
    }
    std::free(msg->reasoning_content);
    std::free(msg->tool_name);
    std::free(msg->tool_call_id);
    if (msg->tool_calls) {
        for (size_t i = 0; i < msg->tool_calls_count; ++i) {
            std::free(msg->tool_calls[i].name);
            std::free(msg->tool_calls[i].arguments);
            std::free(msg->tool_calls[i].id);
        }
        std::free(msg->tool_calls);
    }
    init_chat_msg(msg);
}

extern "C" void llama_rs_chat_msg_diff_free_oaicompat(
    struct llama_rs_chat_msg_diff_oaicompat * diffs,
    size_t count) {
    if (!diffs) {
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        std::free(diffs[i].reasoning_content_delta);
        std::free(diffs[i].content_delta);
        std::free(diffs[i].tool_call_delta.name);
        std::free(diffs[i].tool_call_delta.arguments);
        std::free(diffs[i].tool_call_delta.id);
    }
    std::free(diffs);
}

extern "C" void llama_rs_chat_parse_state_free_oaicompat(struct llama_rs_chat_parse_state_oaicompat * state) {
    delete state;
}

// Thin pass-through to llama.cpp's common_fit_params (a C++ symbol in libcommon).
// Returns common_params_fit_status as an int: 0 = success, 1 = failure, 2 = error.
extern "C" int llama_rs_fit_params(
    const char * path_model,
    struct llama_model_params * mparams,
    struct llama_context_params * cparams,
    float * tensor_split,
    struct llama_model_tensor_buft_override * tensor_buft_overrides,
    size_t * margins,
    uint32_t n_ctx_min,
    enum ggml_log_level log_level) {
    return static_cast<int>(common_fit_params(
        path_model,
        mparams,
        cparams,
        tensor_split,
        tensor_buft_overrides,
        margins,
        n_ctx_min,
        log_level));
}

extern "C" void llama_rs_memory_breakdown_print(const struct llama_context * ctx) {
    common_memory_breakdown_print(ctx);
}
