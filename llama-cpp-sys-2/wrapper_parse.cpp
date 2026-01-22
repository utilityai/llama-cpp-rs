#include "wrapper_common.h"

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

struct llama_rs_chat_parse_state {
    common_chat_syntax syntax;
    common_chat_msg chat_msg;
    std::string generated_text;
    std::vector<std::string> generated_tool_call_ids;

    explicit llama_rs_chat_parse_state(common_chat_syntax syntax_in)
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

static nlohmann::ordered_json chat_msg_to_oaicompat_json(const common_chat_msg & msg) {
    using json = nlohmann::ordered_json;

    json message{
        {"role", "assistant"},
    };
    if (!msg.reasoning_content.empty()) {
        message["reasoning_content"] = msg.reasoning_content;
    }
    if (msg.content.empty() && !msg.tool_calls.empty()) {
        message["content"] = json();
    } else {
        message["content"] = msg.content;
    }
    if (!msg.tool_calls.empty()) {
        auto tool_calls = json::array();
        for (const auto & tc : msg.tool_calls) {
            json entry{
                {"type", "function"},
                {"function", {
                    {"name", tc.name},
                    {"arguments", tc.arguments},
                }},
                {"id", tc.id},
            };
            tool_calls.push_back(entry);
        }
        message["tool_calls"] = tool_calls;
    }
    return message;
}

static void init_chat_msg(struct llama_rs_chat_msg * out_msg) {
    if (!out_msg) {
        return;
    }
    out_msg->role = nullptr;
    out_msg->content = nullptr;
    out_msg->reasoning_content = nullptr;
    out_msg->tool_name = nullptr;
    out_msg->tool_call_id = nullptr;
    out_msg->tool_calls = nullptr;
    out_msg->tool_calls_count = 0;
}

static int fill_chat_msg(
    const common_chat_msg & msg,
    struct llama_rs_chat_msg * out_msg) {
    if (!out_msg) {
        return -1;
    }
    init_chat_msg(out_msg);

    if (!msg.role.empty()) {
        out_msg->role = llama_rs_dup_string(msg.role);
        if (!out_msg->role) {
            return -2;
        }
    }
    if (!msg.content.empty()) {
        out_msg->content = llama_rs_dup_string(msg.content);
        if (!out_msg->content) {
            return -2;
        }
    }
    if (!msg.reasoning_content.empty()) {
        out_msg->reasoning_content = llama_rs_dup_string(msg.reasoning_content);
        if (!out_msg->reasoning_content) {
            return -2;
        }
    }
    if (!msg.tool_name.empty()) {
        out_msg->tool_name = llama_rs_dup_string(msg.tool_name);
        if (!out_msg->tool_name) {
            return -2;
        }
    }
    if (!msg.tool_call_id.empty()) {
        out_msg->tool_call_id = llama_rs_dup_string(msg.tool_call_id);
        if (!out_msg->tool_call_id) {
            return -2;
        }
    }

    if (!msg.tool_calls.empty()) {
        auto * calls = static_cast<struct llama_rs_tool_call *>(
            std::malloc(sizeof(struct llama_rs_tool_call) * msg.tool_calls.size()));
        if (!calls) {
            return -2;
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
                return -2;
            }
        }
        out_msg->tool_calls = calls;
        out_msg->tool_calls_count = msg.tool_calls.size();
    }

    return 0;
}

extern "C" int llama_rs_chat_parse_to_oaicompat(
    const char * input,
    bool is_partial,
    int chat_format,
    bool parse_tool_calls,
    const char * parser_data,
    bool thinking_forced_open,
    char ** out_json) {
    if (!input || !out_json) {
        return -1;
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
        auto json = chat_msg_to_oaicompat_json(msg).dump();
        *out_json = llama_rs_dup_string(json);
        return *out_json ? 0 : -2;
    } catch (const std::exception &) {
        return -3;
    }
}

extern "C" int llama_rs_chat_parse(
    const char * input,
    bool is_partial,
    int chat_format,
    bool parse_tool_calls,
    const char * parser_data,
    bool thinking_forced_open,
    struct llama_rs_chat_msg * out_msg) {
    if (!input || !out_msg) {
        return -1;
    }

    init_chat_msg(out_msg);

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

        if (fill_chat_msg(msg, out_msg) != 0) {
            llama_rs_chat_msg_free(out_msg);
            return -2;
        }

        return 0;
    } catch (const std::exception &) {
        llama_rs_chat_msg_free(out_msg);
        return -3;
    }
}

extern "C" void llama_rs_chat_msg_free(struct llama_rs_chat_msg * msg) {
    if (!msg) {
        return;
    }
    if (msg->role) {
        std::free(msg->role);
    }
    if (msg->content) {
        std::free(msg->content);
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
    msg->reasoning_content = nullptr;
    msg->tool_name = nullptr;
    msg->tool_call_id = nullptr;
    msg->tool_calls = nullptr;
    msg->tool_calls_count = 0;
}

extern "C" struct llama_rs_chat_parse_state * llama_rs_chat_parse_state_init(
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
        return new llama_rs_chat_parse_state(std::move(syntax));
    } catch (const std::exception &) {
        return nullptr;
    }
}

extern "C" int llama_rs_chat_parse_state_update(
    struct llama_rs_chat_parse_state * state,
    const char * text_added,
    bool is_partial,
    struct llama_rs_chat_msg * out_msg,
    struct llama_rs_chat_msg_diff ** out_diffs,
    size_t * out_diffs_count) {
    if (!state || !out_msg || !out_diffs || !out_diffs_count) {
        return -1;
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

        if (fill_chat_msg(state->chat_msg, out_msg) != 0) {
            llama_rs_chat_msg_free(out_msg);
            return -2;
        }

        if (!diffs.empty()) {
            auto * diff_arr = static_cast<struct llama_rs_chat_msg_diff *>(
                std::malloc(sizeof(struct llama_rs_chat_msg_diff) * diffs.size()));
            if (!diff_arr) {
                llama_rs_chat_msg_free(out_msg);
                return -2;
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
                        llama_rs_chat_msg_diff_free(diff_arr, i + 1);
                        llama_rs_chat_msg_free(out_msg);
                        return -2;
                    }
                }
                if (!diffs[i].content_delta.empty()) {
                    diff_arr[i].content_delta =
                        llama_rs_dup_string(diffs[i].content_delta);
                    if (!diff_arr[i].content_delta) {
                        llama_rs_chat_msg_diff_free(diff_arr, i + 1);
                        llama_rs_chat_msg_free(out_msg);
                        return -2;
                    }
                }
                if (diffs[i].tool_call_index != std::string::npos) {
                    if (!diffs[i].tool_call_delta.name.empty()) {
                        diff_arr[i].tool_call_delta.name =
                            llama_rs_dup_string(diffs[i].tool_call_delta.name);
                        if (!diff_arr[i].tool_call_delta.name) {
                            llama_rs_chat_msg_diff_free(diff_arr, i + 1);
                            llama_rs_chat_msg_free(out_msg);
                            return -2;
                        }
                    }
                    if (!diffs[i].tool_call_delta.arguments.empty()) {
                        diff_arr[i].tool_call_delta.arguments =
                            llama_rs_dup_string(diffs[i].tool_call_delta.arguments);
                        if (!diff_arr[i].tool_call_delta.arguments) {
                            llama_rs_chat_msg_diff_free(diff_arr, i + 1);
                            llama_rs_chat_msg_free(out_msg);
                            return -2;
                        }
                    }
                    if (!diffs[i].tool_call_delta.id.empty()) {
                        diff_arr[i].tool_call_delta.id =
                            llama_rs_dup_string(diffs[i].tool_call_delta.id);
                        if (!diff_arr[i].tool_call_delta.id) {
                            llama_rs_chat_msg_diff_free(diff_arr, i + 1);
                            llama_rs_chat_msg_free(out_msg);
                            return -2;
                        }
                    }
                }
            }
            *out_diffs = diff_arr;
            *out_diffs_count = diffs.size();
        }

        return 0;
    } catch (const std::exception &) {
        llama_rs_chat_msg_free(out_msg);
        return -3;
    }
}

extern "C" void llama_rs_chat_msg_diff_free(
    struct llama_rs_chat_msg_diff * diffs,
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

extern "C" void llama_rs_chat_parse_state_free(struct llama_rs_chat_parse_state * state) {
    delete state;
}
