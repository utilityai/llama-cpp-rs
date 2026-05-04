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
    common_chat_parser_params syntax;
    common_chat_msg chat_msg;
    std::string generated_text;
    std::vector<std::string> generated_tool_call_ids;

    explicit llama_rs_chat_parse_state_oaicompat(common_chat_parser_params syntax_in)
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

static json template_result_to_json(const common_chat_params & params) {
    json result = json::object();
    result["prompt"] = params.prompt;
    result["chat_format"] = static_cast<int>(params.format);
    result["supports_thinking"] = params.supports_thinking;
    result["grammar_lazy"] = params.grammar_lazy;
    if (!params.grammar.empty()) {
        result["grammar"] = params.grammar;
    }
    if (!params.parser.empty()) {
        result["parser"] = params.parser;
    }
    if (!params.generation_prompt.empty()) {
        result["generation_prompt"] = params.generation_prompt;
    }
    json triggers = json::array();
    for (const auto & trigger : params.grammar_triggers) {
        json trigger_json = json::object();
        trigger_json["type"] = static_cast<int>(trigger.type);
        trigger_json["value"] = trigger.value;
        trigger_json["token"] = trigger.token;
        triggers.push_back(std::move(trigger_json));
    }
    result["grammar_triggers"] = std::move(triggers);
    result["preserved_tokens"] = params.preserved_tokens;
    result["additional_stops"] = params.additional_stops;
    return result;
}

static json diff_to_json(const common_chat_msg_diff & diff) {
    json delta = json::object();
    if (!diff.reasoning_content_delta.empty()) {
        delta["reasoning_content"] = diff.reasoning_content_delta;
    }
    if (!diff.content_delta.empty()) {
        delta["content"] = diff.content_delta;
    }
    if (diff.tool_call_index != std::string::npos) {
        json tool_call;
        tool_call["index"] = diff.tool_call_index;
        if (!diff.tool_call_delta.id.empty()) {
            tool_call["id"] = diff.tool_call_delta.id;
            tool_call["type"] = "function";
        }
        const bool has_name = !diff.tool_call_delta.name.empty();
        const bool has_arguments = !diff.tool_call_delta.arguments.empty();
        if (has_name || has_arguments) {
            json function = json::object();
            if (has_name) {
                function["name"] = diff.tool_call_delta.name;
            }
            if (has_arguments) {
                function["arguments"] = diff.tool_call_delta.arguments;
            }
            tool_call["function"] = function;
        }
        delta["tool_calls"] = json::array({ tool_call });
    }
    return delta;
}

extern "C" llama_rs_status llama_rs_apply_chat_template_with_tools_oaicompat(
    const struct llama_model * model,
    const char * chat_template,
    const struct llama_chat_message * messages,
    size_t message_count,
    const char * tools_json,
    const char * json_schema,
    bool add_generation_prompt,
    char ** out_json) {
    if (!chat_template || !out_json) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *out_json = nullptr;

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
            inputs.tools = common_chat_tools_parse_oaicompat(json::parse(tools_json));
        }
        if (json_schema && std::strlen(json_schema) > 0) {
            inputs.json_schema = json_schema;
        }

        auto params = common_chat_templates_apply(tmpls.get(), inputs);
        *out_json = llama_rs_dup_string(template_result_to_json(params).dump());
        return *out_json ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_ALLOCATION_FAILED;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_apply_chat_template_oaicompat(
    const struct llama_model * model,
    const char * chat_template,
    const struct llama_rs_chat_template_oaicompat_params * params,
    char ** out_json) {
    if (!chat_template || !params || !out_json) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *out_json = nullptr;

    if (!params->messages) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

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
            if (!kwargs.is_object()) {
                throw std::invalid_argument("chat_template_kwargs must be a JSON object");
            }
            for (const auto & item : kwargs.items()) {
                inputs.chat_template_kwargs[item.key()] = item.value().dump();
            }
        }

        auto params_out = common_chat_templates_apply(tmpls.get(), inputs);
        *out_json = llama_rs_dup_string(template_result_to_json(params_out).dump());
        return *out_json ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_ALLOCATION_FAILED;
    } catch (const std::exception &) {
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
        if (parser_data && std::strlen(parser_data) > 0) {
            syntax.parser.load(parser_data);
        }
        if (generation_prompt && std::strlen(generation_prompt) > 0) {
            syntax.generation_prompt = generation_prompt;
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
        if (parser_data && std::strlen(parser_data) > 0) {
            syntax.parser.load(parser_data);
        }
        if (generation_prompt && std::strlen(generation_prompt) > 0) {
            syntax.generation_prompt = generation_prompt;
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
    char ** out_diffs_json) {
    if (!state || !out_diffs_json) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *out_diffs_json = nullptr;

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

        json diffs_json = json::array();
        for (const auto & diff : diffs) {
            diffs_json.push_back(diff_to_json(diff));
        }
        *out_diffs_json = llama_rs_dup_string(diffs_json.dump());
        return *out_diffs_json ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_ALLOCATION_FAILED;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" void llama_rs_chat_parse_state_free_oaicompat(struct llama_rs_chat_parse_state_oaicompat * state) {
    delete state;
}
