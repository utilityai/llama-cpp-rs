#include "wrapper_chat_parse.h"

#include "llama.cpp/common/chat-auto-parser.h"
#include "llama.cpp/common/chat.h"
#include "llama.cpp/include/llama.h"

#include <exception>
#include <nlohmann/json.hpp>
#include <string>

struct llama_rs_parsed_chat {
    common_chat_msg message;
};

namespace {

std::string token_text_or_empty(const llama_vocab * vocab, llama_token token) {
    if (token == LLAMA_TOKEN_NULL) {
        return {};
    }

    const char * text = llama_vocab_get_text(vocab, token);
    if (!text) {
        return {};
    }

    return std::string(text);
}

}  // namespace

extern "C" llama_rs_status llama_rs_parse_chat_message(
    const struct llama_model * model,
    const char * tools_json,
    const char * input,
    int is_partial,
    llama_rs_parsed_chat_handle * out_handle,
    char ** out_error) {
    if (out_handle) {
        *out_handle = nullptr;
    }
    if (out_error) {
        *out_error = nullptr;
    }

    if (!model || !input || !out_handle || !out_error) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    try {
        const char * tmpl_src = llama_model_chat_template(model, nullptr);
        if (!tmpl_src) {
            return LLAMA_RS_STATUS_INVALID_ARGUMENT;
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        if (!vocab) {
            return LLAMA_RS_STATUS_INVALID_ARGUMENT;
        }

        std::string bos_token = token_text_or_empty(vocab, llama_vocab_bos(vocab));
        std::string eos_token = token_text_or_empty(vocab, llama_vocab_eos(vocab));

        common_chat_template tmpl(tmpl_src, bos_token, eos_token);

        autoparser::generation_params inputs;
        inputs.add_generation_prompt = true;
        inputs.enable_thinking = true;
        inputs.messages = nlohmann::ordered_json::array({
            { { "role", "user" }, { "content", "ping" } }
        });

        if (tools_json && tools_json[0] != '\0') {
            inputs.tools = nlohmann::ordered_json::parse(tools_json);
        } else {
            inputs.tools = nlohmann::ordered_json::array();
        }

        common_chat_params chat_params =
            autoparser::peg_generator::generate_parser(tmpl, inputs);

        common_chat_parser_params parser_params(chat_params);
        parser_params.parser.load(chat_params.parser);

        common_chat_msg parsed = common_chat_parse(input, is_partial != 0, parser_params);

        auto * handle = new llama_rs_parsed_chat{};
        handle->message = std::move(parsed);

        *out_handle = handle;

        return LLAMA_RS_STATUS_OK;
    } catch (const std::exception & ex) {
        *out_error = llama_rs_dup_string(std::string(ex.what()));

        return LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        *out_error = llama_rs_dup_string(std::string("unknown c++ exception"));

        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" void llama_rs_parsed_chat_free(llama_rs_parsed_chat_handle handle) {
    delete handle;
}

extern "C" size_t llama_rs_parsed_chat_tool_call_count(llama_rs_parsed_chat_handle handle) {
    if (!handle) {
        return 0;
    }
    return handle->message.tool_calls.size();
}

extern "C" char * llama_rs_parsed_chat_tool_call_id(
    llama_rs_parsed_chat_handle handle, size_t index) {
    if (!handle || index >= handle->message.tool_calls.size()) {
        return nullptr;
    }
    return llama_rs_dup_string(handle->message.tool_calls[index].id);
}

extern "C" char * llama_rs_parsed_chat_tool_call_name(
    llama_rs_parsed_chat_handle handle, size_t index) {
    if (!handle || index >= handle->message.tool_calls.size()) {
        return nullptr;
    }
    return llama_rs_dup_string(handle->message.tool_calls[index].name);
}

extern "C" char * llama_rs_parsed_chat_tool_call_arguments(
    llama_rs_parsed_chat_handle handle, size_t index) {
    if (!handle || index >= handle->message.tool_calls.size()) {
        return nullptr;
    }
    return llama_rs_dup_string(handle->message.tool_calls[index].arguments);
}

extern "C" char * llama_rs_parsed_chat_content(llama_rs_parsed_chat_handle handle) {
    if (!handle) {
        return nullptr;
    }
    return llama_rs_dup_string(handle->message.content);
}

extern "C" char * llama_rs_parsed_chat_reasoning_content(llama_rs_parsed_chat_handle handle) {
    if (!handle) {
        return nullptr;
    }
    return llama_rs_dup_string(handle->message.reasoning_content);
}
