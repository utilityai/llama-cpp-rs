#include "wrapper_oai.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <random>
#include <string>
#include <vector>

#include "llama.cpp/common/chat.h"
#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

struct llama_rs_chat_parse_state_oaicompat {
    common_chat_syntax syntax;
    common_chat_msg chat_msg;
    std::string generated_text;
    std::vector<std::string> generated_tool_call_ids;

    explicit llama_rs_chat_parse_state_oaicompat(common_chat_syntax syntax_in)
        : syntax(std::move(syntax_in)) {}
};

static std::string random_string(size_t length = 32) {
    static constexpr char chars[] =
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    static constexpr size_t chars_len = sizeof(chars) - 1;
    if (length == 0) {
        return std::string();
    }

    static thread_local std::mt19937 generator([]() {
        std::random_device rd;
        std::seed_seq seed{
            rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd(),
        };
        return std::mt19937(seed);
    }());

    std::uniform_int_distribution<size_t> distribution(0, chars_len - 1);
    std::string result(length, '\0');
    for (size_t i = 0; i < length; ++i) {
        result[i] = chars[distribution(generator)];
    }
    return result;
}

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

static llama_rs_status dup_content_parts(
    const std::vector<common_chat_msg_content_part> & parts,
    struct llama_rs_chat_msg_content_part_oaicompat ** out_items,
    size_t * out_count) {
    if (!out_items || !out_count) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *out_items = nullptr;
    *out_count = 0;
    if (parts.empty()) {
        return LLAMA_RS_STATUS_OK;
    }

    auto * items = static_cast<struct llama_rs_chat_msg_content_part_oaicompat *>(
        std::malloc(sizeof(struct llama_rs_chat_msg_content_part_oaicompat) * parts.size()));
    if (!items) {
        return LLAMA_RS_STATUS_ALLOCATION_FAILED;
    }
    for (size_t i = 0; i < parts.size(); ++i) {
        items[i].type = llama_rs_dup_string(parts[i].type);
        items[i].text = llama_rs_dup_string(parts[i].text);
        if ((!items[i].type && !parts[i].type.empty())
            || (!items[i].text && !parts[i].text.empty())) {
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
    if (!out_msg) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
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
        const auto status = dup_content_parts(
            msg.content_parts,
            &out_msg->content_parts,
            &out_msg->content_parts_count);
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
            std::malloc(sizeof(struct llama_rs_tool_call_oaicompat) * msg.tool_calls.size()));
        if (!calls) {
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
        for (size_t i = 0; i < msg.tool_calls.size(); ++i) {
            calls[i].name = llama_rs_dup_string(msg.tool_calls[i].name);
            calls[i].arguments = llama_rs_dup_string(msg.tool_calls[i].arguments);
            calls[i].id = llama_rs_dup_string(msg.tool_calls[i].id);
            if ((!calls[i].name && !msg.tool_calls[i].name.empty())
                || (!calls[i].arguments && !msg.tool_calls[i].arguments.empty())
                || (!calls[i].id && !msg.tool_calls[i].id.empty())) {
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

static llama_rs_status to_common_chat_msg(
    const struct llama_rs_chat_msg_oaicompat & msg,
    common_chat_msg & out_msg) {
    if (!msg.role) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    out_msg.role = msg.role;
    if (msg.content) {
        out_msg.content = msg.content;
    }
    if (msg.content_parts_count > 0) {
        if (!msg.content_parts) {
            return LLAMA_RS_STATUS_INVALID_ARGUMENT;
        }
        out_msg.content_parts.reserve(msg.content_parts_count);
        for (size_t i = 0; i < msg.content_parts_count; ++i) {
            const auto & part = msg.content_parts[i];
            if (!part.type || !part.text) {
                return LLAMA_RS_STATUS_INVALID_ARGUMENT;
            }
            common_chat_msg_content_part item;
            item.type = part.type;
            item.text = part.text;
            out_msg.content_parts.push_back(std::move(item));
        }
    }
    if (msg.reasoning_content) {
        out_msg.reasoning_content = msg.reasoning_content;
    }
    if (msg.tool_name) {
        out_msg.tool_name = msg.tool_name;
    }
    if (msg.tool_call_id) {
        out_msg.tool_call_id = msg.tool_call_id;
    }
    if (msg.tool_calls_count > 0) {
        if (!msg.tool_calls) {
            return LLAMA_RS_STATUS_INVALID_ARGUMENT;
        }
        out_msg.tool_calls.reserve(msg.tool_calls_count);
        for (size_t i = 0; i < msg.tool_calls_count; ++i) {
            const auto & call = msg.tool_calls[i];
            if (!call.name || !call.arguments) {
                return LLAMA_RS_STATUS_INVALID_ARGUMENT;
            }
            common_chat_tool_call tool_call;
            tool_call.name = call.name;
            tool_call.arguments = call.arguments;
            tool_call.id = call.id ? call.id : "";
            out_msg.tool_calls.push_back(std::move(tool_call));
        }
    }
    return LLAMA_RS_STATUS_OK;
}

extern "C" llama_rs_status llama_rs_apply_chat_template_with_tools_oaicompat(
    const struct llama_model * model,
    const char * chat_template,
    const struct llama_chat_message * messages,
    size_t message_count,
    const char * tools_json,
    const char * json_schema,
    bool add_generation_prompt,
    struct llama_rs_chat_template_result * out_result) {
    if (!chat_template || !out_result) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    out_result->prompt = nullptr;
    out_result->grammar = nullptr;
    out_result->parser = nullptr;
    out_result->chat_format = 0;
    out_result->thinking_forced_open = false;
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
        inputs.add_generation_prompt = add_generation_prompt;
        inputs.use_jinja = true;

        inputs.messages.reserve(message_count);
        for (size_t i = 0; i < message_count; ++i) {
            common_chat_msg msg;
            msg.role = messages[i].role ? messages[i].role : "";
            msg.content = messages[i].content ? messages[i].content : "";
            inputs.messages.push_back(std::move(msg));
        }

        if (tools_json && std::strlen(tools_json) > 0) {
            inputs.tools = common_chat_tools_parse_oaicompat<std::string>(tools_json);
        }
        if (json_schema && std::strlen(json_schema) > 0) {
            inputs.json_schema = json_schema;
        }

        auto params = common_chat_templates_apply(tmpls.get(), inputs);
        out_result->prompt = llama_rs_dup_string(params.prompt);
        if (!params.grammar.empty()) {
            out_result->grammar = llama_rs_dup_string(params.grammar);
        }
        if (!params.parser.empty()) {
            out_result->parser = llama_rs_dup_string(params.parser);
        }
        out_result->chat_format = static_cast<int>(params.format);
        out_result->thinking_forced_open = params.thinking_forced_open;
        out_result->grammar_lazy = params.grammar_lazy;
        const auto status_triggers = dup_trigger_array(
            params.grammar_triggers,
            &out_result->grammar_triggers,
            &out_result->grammar_triggers_count);
        if (status_triggers != LLAMA_RS_STATUS_OK) {
            llama_rs_chat_template_result_free(out_result);
            return status_triggers;
        }
        const auto status_tokens = dup_string_array(
            params.preserved_tokens,
            &out_result->preserved_tokens,
            &out_result->preserved_tokens_count);
        if (status_tokens != LLAMA_RS_STATUS_OK) {
            llama_rs_chat_template_result_free(out_result);
            return status_tokens;
        }
        const auto status_stops = dup_string_array(
            params.additional_stops,
            &out_result->additional_stops,
            &out_result->additional_stops_count);
        if (status_stops != LLAMA_RS_STATUS_OK) {
            llama_rs_chat_template_result_free(out_result);
            return status_stops;
        }
        if (!out_result->prompt) {
            llama_rs_chat_template_result_free(out_result);
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
        return LLAMA_RS_STATUS_OK;
    } catch (const std::exception &) {
        llama_rs_chat_template_result_free(out_result);
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_apply_chat_template_oaicompat(
    const struct llama_model * model,
    const char * chat_template,
    const struct llama_rs_chat_template_oaicompat_params * params,
    struct llama_rs_chat_template_result * out_result) {
    if (!chat_template || !params || !out_result) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    if (!params->messages) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    out_result->prompt = nullptr;
    out_result->grammar = nullptr;
    out_result->parser = nullptr;
    out_result->chat_format = 0;
    out_result->thinking_forced_open = false;
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

        inputs.messages = common_chat_msgs_parse_oaicompat<std::string>(params->messages);
        if (params->tools && std::strlen(params->tools) > 0) {
            inputs.tools = common_chat_tools_parse_oaicompat<std::string>(params->tools);
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
            if (!kwargs.is_object()) {
                throw std::invalid_argument("chat_template_kwargs must be a JSON object");
            }
            for (const auto & item : kwargs.items()) {
                inputs.chat_template_kwargs[item.key()] = item.value().dump();
            }
        }

        auto params_out = common_chat_templates_apply(tmpls.get(), inputs);
        out_result->prompt = llama_rs_dup_string(params_out.prompt);
        if (!params_out.grammar.empty()) {
            out_result->grammar = llama_rs_dup_string(params_out.grammar);
        }
        if (!params_out.parser.empty()) {
            out_result->parser = llama_rs_dup_string(params_out.parser);
        }
        out_result->chat_format = static_cast<int>(params_out.format);
        out_result->thinking_forced_open = params_out.thinking_forced_open;
        out_result->grammar_lazy = params_out.grammar_lazy;

        const auto status_triggers = dup_trigger_array(
            params_out.grammar_triggers,
            &out_result->grammar_triggers,
            &out_result->grammar_triggers_count);
        if (status_triggers != LLAMA_RS_STATUS_OK) {
            llama_rs_chat_template_result_free(out_result);
            return status_triggers;
        }
        const auto status_tokens = dup_string_array(
            params_out.preserved_tokens,
            &out_result->preserved_tokens,
            &out_result->preserved_tokens_count);
        if (status_tokens != LLAMA_RS_STATUS_OK) {
            llama_rs_chat_template_result_free(out_result);
            return status_tokens;
        }
        const auto status_stops = dup_string_array(
            params_out.additional_stops,
            &out_result->additional_stops,
            &out_result->additional_stops_count);
        if (status_stops != LLAMA_RS_STATUS_OK) {
            llama_rs_chat_template_result_free(out_result);
            return status_stops;
        }
        if (!out_result->prompt) {
            llama_rs_chat_template_result_free(out_result);
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
        return LLAMA_RS_STATUS_OK;
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
    bool thinking_forced_open,
    char ** out_json) {
    if (!input || !out_json) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *out_json = nullptr;

    try {
        common_chat_syntax syntax;
        syntax.format = static_cast<common_chat_format>(chat_format);
        syntax.parse_tool_calls = parse_tool_calls;
        syntax.thinking_forced_open = thinking_forced_open;
        if (parser_data && std::strlen(parser_data) > 0) {
            syntax.parser.load(parser_data);
        }

        auto msg = common_chat_parse(input, is_partial, syntax);
        std::vector<std::string> ids_cache;
        msg.set_tool_call_ids(ids_cache, []() { return random_string(); });
        auto json_msg = msg.to_json_oaicompat<json>().dump();
        *out_json = llama_rs_dup_string(json_msg);
        return *out_json ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_ALLOCATION_FAILED;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" void llama_rs_chat_msg_free_oaicompat(struct llama_rs_chat_msg_oaicompat * msg) {
    if (!msg) {
        return;
    }
    if (msg->role) {
        std::free(msg->role);
    }
    if (msg->content) {
        std::free(msg->content);
    }
    if (msg->content_parts) {
        for (size_t i = 0; i < msg->content_parts_count; ++i) {
            std::free(msg->content_parts[i].type);
            std::free(msg->content_parts[i].text);
        }
        std::free(msg->content_parts);
    }
    if (msg->reasoning_content) {
        std::free(msg->reasoning_content);
    }
    if (msg->tool_name) {
        std::free(msg->tool_name);
    }
    if (msg->tool_call_id) {
        std::free(msg->tool_call_id);
    }
    if (msg->tool_calls) {
        for (size_t i = 0; i < msg->tool_calls_count; ++i) {
            std::free(msg->tool_calls[i].name);
            std::free(msg->tool_calls[i].arguments);
            std::free(msg->tool_calls[i].id);
        }
        std::free(msg->tool_calls);
    }
    msg->role = nullptr;
    msg->content = nullptr;
    msg->content_parts = nullptr;
    msg->content_parts_count = 0;
    msg->reasoning_content = nullptr;
    msg->tool_name = nullptr;
    msg->tool_call_id = nullptr;
    msg->tool_calls = nullptr;
    msg->tool_calls_count = 0;
}

extern "C" void llama_rs_chat_msgs_free_oaicompat(struct llama_rs_chat_msg_oaicompat * msgs, size_t count) {
    if (!msgs) {
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        llama_rs_chat_msg_free_oaicompat(&msgs[i]);
    }
    std::free(msgs);
}

extern "C" struct llama_rs_chat_parse_state_oaicompat * llama_rs_chat_parse_state_init_oaicompat(
    int chat_format,
    bool parse_tool_calls,
    const char * parser_data,
    bool thinking_forced_open) {
    try {
        common_chat_syntax syntax;
        syntax.format = static_cast<common_chat_format>(chat_format);
        syntax.parse_tool_calls = parse_tool_calls;
        syntax.thinking_forced_open = thinking_forced_open;
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
            new_msg.set_tool_call_ids(state->generated_tool_call_ids, []() {
                return random_string();
            });
            state->chat_msg = new_msg;
            diffs = common_chat_msg_diff::compute_diffs(msg_prv_copy, state->chat_msg);
        }

        const auto status = fill_chat_msg(state->chat_msg, out_msg);
        if (status != LLAMA_RS_STATUS_OK) {
            llama_rs_chat_msg_free_oaicompat(out_msg);
            return status;
        }

        if (!diffs.empty()) {
            auto * diff_arr = static_cast<struct llama_rs_chat_msg_diff_oaicompat *>(
                std::malloc(sizeof(struct llama_rs_chat_msg_diff_oaicompat) * diffs.size()));
            if (!diff_arr) {
                llama_rs_chat_msg_free_oaicompat(out_msg);
                return LLAMA_RS_STATUS_ALLOCATION_FAILED;
            }
            for (size_t i = 0; i < diffs.size(); ++i) {
                diff_arr[i].reasoning_content_delta = nullptr;
                diff_arr[i].content_delta = nullptr;
                diff_arr[i].tool_call_delta.name = nullptr;
                diff_arr[i].tool_call_delta.arguments = nullptr;
                diff_arr[i].tool_call_delta.id = nullptr;
                diff_arr[i].tool_call_index = diffs[i].tool_call_index;

                if (!diffs[i].reasoning_content_delta.empty()) {
                    diff_arr[i].reasoning_content_delta =
                        llama_rs_dup_string(diffs[i].reasoning_content_delta);
                    if (!diff_arr[i].reasoning_content_delta) {
                        llama_rs_chat_msg_diff_free_oaicompat(diff_arr, i + 1);
                        llama_rs_chat_msg_free_oaicompat(out_msg);
                        return LLAMA_RS_STATUS_ALLOCATION_FAILED;
                    }
                }
                if (!diffs[i].content_delta.empty()) {
                    diff_arr[i].content_delta =
                        llama_rs_dup_string(diffs[i].content_delta);
                    if (!diff_arr[i].content_delta) {
                        llama_rs_chat_msg_diff_free_oaicompat(diff_arr, i + 1);
                        llama_rs_chat_msg_free_oaicompat(out_msg);
                        return LLAMA_RS_STATUS_ALLOCATION_FAILED;
                    }
                }
                if (diffs[i].tool_call_index != std::string::npos) {
                    if (!diffs[i].tool_call_delta.name.empty()) {
                        diff_arr[i].tool_call_delta.name =
                            llama_rs_dup_string(diffs[i].tool_call_delta.name);
                        if (!diff_arr[i].tool_call_delta.name) {
                            llama_rs_chat_msg_diff_free_oaicompat(diff_arr, i + 1);
                            llama_rs_chat_msg_free_oaicompat(out_msg);
                            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
                        }
                    }
                    if (!diffs[i].tool_call_delta.arguments.empty()) {
                        diff_arr[i].tool_call_delta.arguments =
                            llama_rs_dup_string(diffs[i].tool_call_delta.arguments);
                        if (!diff_arr[i].tool_call_delta.arguments) {
                            llama_rs_chat_msg_diff_free_oaicompat(diff_arr, i + 1);
                            llama_rs_chat_msg_free_oaicompat(out_msg);
                            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
                        }
                    }
                    if (!diffs[i].tool_call_delta.id.empty()) {
                        diff_arr[i].tool_call_delta.id =
                            llama_rs_dup_string(diffs[i].tool_call_delta.id);
                        if (!diff_arr[i].tool_call_delta.id) {
                            llama_rs_chat_msg_diff_free_oaicompat(diff_arr, i + 1);
                            llama_rs_chat_msg_free_oaicompat(out_msg);
                            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
                        }
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

extern "C" llama_rs_status llama_rs_chat_tools_parse_oaicompat(
    const char * tools_json,
    struct llama_rs_chat_tool_oaicompat ** out_tools,
    size_t * out_count) {
    if (!tools_json || !out_tools || !out_count) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *out_tools = nullptr;
    *out_count = 0;

    try {
        auto tools = common_chat_tools_parse_oaicompat<std::string>(tools_json);
        if (tools.empty()) {
            return LLAMA_RS_STATUS_OK;
        }

        auto * items = static_cast<struct llama_rs_chat_tool_oaicompat *>(
            std::malloc(sizeof(struct llama_rs_chat_tool_oaicompat) * tools.size()));
        if (!items) {
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
        for (size_t i = 0; i < tools.size(); ++i) {
            items[i].name = llama_rs_dup_string(tools[i].name);
            items[i].description = llama_rs_dup_string(tools[i].description);
            items[i].parameters = llama_rs_dup_string(tools[i].parameters);
            if ((!items[i].name && !tools[i].name.empty())
                || (!items[i].description && !tools[i].description.empty())
                || (!items[i].parameters && !tools[i].parameters.empty())) {
                for (size_t j = 0; j <= i; ++j) {
                    std::free(items[j].name);
                    std::free(items[j].description);
                    std::free(items[j].parameters);
                }
                std::free(items);
                return LLAMA_RS_STATUS_ALLOCATION_FAILED;
            }
        }
        *out_tools = items;
        *out_count = tools.size();
        return LLAMA_RS_STATUS_OK;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_chat_tools_to_oaicompat_json(
    const struct llama_rs_chat_tool_oaicompat * tools,
    size_t tools_count,
    char ** out_json) {
    if (!out_json) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *out_json = nullptr;
    if (tools_count > 0 && !tools) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    try {
        std::vector<common_chat_tool> parsed;
        parsed.reserve(tools_count);
        for (size_t i = 0; i < tools_count; ++i) {
            if (!tools[i].name || !tools[i].parameters) {
                return LLAMA_RS_STATUS_INVALID_ARGUMENT;
            }
            common_chat_tool tool;
            tool.name = tools[i].name;
            tool.description = tools[i].description ? tools[i].description : "";
            tool.parameters = tools[i].parameters;
            parsed.push_back(std::move(tool));
        }
        auto json_tools = common_chat_tools_to_json_oaicompat<json>(parsed).dump();
        *out_json = llama_rs_dup_string(json_tools);
        return *out_json ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_ALLOCATION_FAILED;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" void llama_rs_chat_tools_free_oaicompat(struct llama_rs_chat_tool_oaicompat * tools, size_t count) {
    if (!tools) {
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        std::free(tools[i].name);
        std::free(tools[i].description);
        std::free(tools[i].parameters);
    }
    std::free(tools);
}

extern "C" llama_rs_status llama_rs_chat_msgs_parse_oaicompat(
    const char * messages_json,
    struct llama_rs_chat_msg_oaicompat ** out_msgs,
    size_t * out_count) {
    if (!messages_json || !out_msgs || !out_count) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *out_msgs = nullptr;
    *out_count = 0;

    try {
        auto msgs = common_chat_msgs_parse_oaicompat<std::string>(messages_json);
        if (msgs.empty()) {
            return LLAMA_RS_STATUS_OK;
        }

        auto * items = static_cast<struct llama_rs_chat_msg_oaicompat *>(
            std::malloc(sizeof(struct llama_rs_chat_msg_oaicompat) * msgs.size()));
        if (!items) {
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
        for (size_t i = 0; i < msgs.size(); ++i) {
            const auto status = fill_chat_msg(msgs[i], &items[i]);
            if (status != LLAMA_RS_STATUS_OK) {
                for (size_t j = 0; j <= i; ++j) {
                    llama_rs_chat_msg_free_oaicompat(&items[j]);
                }
                std::free(items);
                return status;
            }
        }
        *out_msgs = items;
        *out_count = msgs.size();
        return LLAMA_RS_STATUS_OK;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_chat_msgs_to_oaicompat_json(
    const struct llama_rs_chat_msg_oaicompat * messages,
    size_t messages_count,
    bool concat_typed_text,
    char ** out_json) {
    if (!out_json) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *out_json = nullptr;
    if (messages_count > 0 && !messages) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    try {
        std::vector<common_chat_msg> parsed;
        parsed.reserve(messages_count);
        for (size_t i = 0; i < messages_count; ++i) {
            common_chat_msg msg;
            const auto status = to_common_chat_msg(messages[i], msg);
            if (status != LLAMA_RS_STATUS_OK) {
                return status;
            }
            parsed.push_back(std::move(msg));
        }
        auto json_msgs = common_chat_msgs_to_json_oaicompat<json>(parsed, concat_typed_text).dump();
        *out_json = llama_rs_dup_string(json_msgs);
        return *out_json ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_ALLOCATION_FAILED;
    } catch (const std::exception &) {
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
        common_chat_msg_diff msg_diff;
        msg_diff.reasoning_content_delta =
            diff->reasoning_content_delta ? diff->reasoning_content_delta : "";
        msg_diff.content_delta =
            diff->content_delta ? diff->content_delta : "";
        msg_diff.tool_call_index = diff->tool_call_index;
        if (diff->tool_call_index != std::string::npos) {
            msg_diff.tool_call_delta.name =
                diff->tool_call_delta.name ? diff->tool_call_delta.name : "";
            msg_diff.tool_call_delta.arguments =
                diff->tool_call_delta.arguments ? diff->tool_call_delta.arguments : "";
            msg_diff.tool_call_delta.id =
                diff->tool_call_delta.id ? diff->tool_call_delta.id : "";
        }
        auto json_delta = common_chat_msg_diff_to_json_oaicompat<json>(msg_diff).dump();
        *out_json = llama_rs_dup_string(json_delta);
        return *out_json ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_ALLOCATION_FAILED;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_chat_tool_choice_parse_oaicompat(
    const char * tool_choice,
    enum llama_rs_chat_tool_choice_oaicompat * out_choice) {
    if (!out_choice) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    if (!tool_choice || tool_choice[0] == '\0') {
        *out_choice = LLAMA_RS_CHAT_TOOL_CHOICE_OAICOMPAT_AUTO;
        return LLAMA_RS_STATUS_OK;
    }
    try {
        const auto parsed = common_chat_tool_choice_parse_oaicompat(tool_choice);
        *out_choice = static_cast<llama_rs_chat_tool_choice_oaicompat>(parsed);
        return LLAMA_RS_STATUS_OK;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
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
