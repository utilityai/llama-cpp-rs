#include "wrapper_tool_calls.h"
#include "wrapper_token_text.h"

#include "llama.cpp/common/chat-auto-parser.h"
#include "llama.cpp/common/chat-auto-parser-helpers.h"
#include "llama.cpp/common/chat.h"
#include "llama.cpp/include/llama.h"

#include <exception>
#include <nlohmann/json.hpp>
#include <string>

using wrapper_helpers::token_text_or_empty;

namespace {

// Render the chat template with a deterministic tool-call assistant turn and
// diff it against the no-tool-call variant. Returns the raw section between
// the model's tool-call open/close markers — i.e. the `<...>{...}</...>`
// fragment the model is expected to emit, with any reasoning prelude removed.
//
// We deliberately reproduce the autoparser's diff-based approach (so the
// detected markers come from the model's actual template behavior, not from a
// hardcoded list), but use plain-ASCII synthetic names where the upstream
// autoparser uses sentinel strings that some Jinja templates choke on.
std::string detect_tool_call_haystack(
    const common_chat_template & tmpl,
    const autoparser::analyze_reasoning & reasoning) {
    nlohmann::ordered_json user_msg = {
        { "role",    "user"                },
        { "content", "Please use the tool" }
    };
    nlohmann::ordered_json assistant_no_tools = {
        { "role",    "assistant"      },
        { "content", "Sure, calling." }
    };
    nlohmann::ordered_json first_tool_call = {
        { "id",       "call_001"  },
        { "type",     "function"  },
        { "function", {
            { "name",      "tool_first" },
            { "arguments", {
                { "arg_first", "XXXX" },
                { "arg_second", "YYYY" },
            }}
        }}
    };
    nlohmann::ordered_json assistant_with_tools = {
        { "role",       "assistant"                                                  },
        { "content",    ""                                                           },
        { "tool_calls", nlohmann::ordered_json::array({ first_tool_call })           }
    };
    nlohmann::ordered_json tool_definition = {
        { "type",     "function"  },
        { "function", {
            { "name",        "tool_first"           },
            { "description", "First test tool"      },
            { "parameters", {
                { "type", "object" },
                { "properties", {
                    { "arg_first", { { "type", "string" }, { "description", "first arg" } } },
                    { "arg_second", { { "type", "string" }, { "description", "second arg" } } },
                }},
                { "required", nlohmann::ordered_json::array({ "arg_first", "arg_second" }) },
            }}
        }}
    };

    template_params params_no_tools;
    params_no_tools.messages              = nlohmann::ordered_json::array({ user_msg, assistant_no_tools });
    params_no_tools.tools                 = nlohmann::ordered_json::array({ tool_definition });
    params_no_tools.add_generation_prompt = false;
    params_no_tools.enable_thinking       = true;

    template_params params_with_tools = params_no_tools;
    params_with_tools.messages =
        nlohmann::ordered_json::array({ user_msg, assistant_with_tools });

    std::string output_no_tools = autoparser::apply_template(tmpl, params_no_tools);
    std::string output_with_tools = autoparser::apply_template(tmpl, params_with_tools);

    if (output_no_tools.empty() || output_with_tools.empty()) {
        return {};
    }

    diff_split diff = calculate_diff_split(output_no_tools, output_with_tools);
    std::string haystack = diff.right;

    // Strip reasoning markers so the surrounding tool-call markers can be
    // located reliably — the autoparser does the same for the JSON-native
    // path.
    auto remove_first = [&haystack](const std::string & needle) {
        if (needle.empty()) {
            return;
        }
        auto pos = haystack.find(needle);
        if (pos != std::string::npos) {
            haystack = haystack.substr(0, pos) + haystack.substr(pos + needle.length());
        }
    };

    remove_first(reasoning.start);
    remove_first(reasoning.end);

    return haystack;
}

}  // namespace

extern "C" llama_rs_status llama_rs_compute_tool_call_haystack(
    const struct llama_model * model,
    char ** out_haystack,
    char ** out_error) {
    if (out_haystack) {
        *out_haystack = nullptr;
    }
    if (out_error) {
        *out_error = nullptr;
    }

    if (!model || !out_haystack || !out_error) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    try {
        const char * tmpl_src = llama_model_chat_template(model, nullptr);
        if (!tmpl_src) {
            return LLAMA_RS_STATUS_OK;
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        if (!vocab) {
            return LLAMA_RS_STATUS_OK;
        }

        std::string bos_token = token_text_or_empty(vocab, llama_vocab_bos(vocab));
        std::string eos_token = token_text_or_empty(vocab, llama_vocab_eos(vocab));

        common_chat_template tmpl(tmpl_src, bos_token, eos_token);
        auto jinja_caps = tmpl.original_caps();
        autoparser::analyze_reasoning reasoning(tmpl, jinja_caps.supports_tool_calls);

        std::string haystack = detect_tool_call_haystack(tmpl, reasoning);
        if (haystack.empty()) {
            return LLAMA_RS_STATUS_OK;
        }

        char * haystack_dup = llama_rs_dup_string(haystack);
        if (!haystack_dup) {
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }

        *out_haystack = haystack_dup;

        return LLAMA_RS_STATUS_OK;
    } catch (const std::exception & ex) {
        *out_error = llama_rs_dup_string(std::string(ex.what()));

        return LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        *out_error = llama_rs_dup_string(std::string("unknown c++ exception"));

        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_diagnose_tool_call_synthetic_renders(
    const struct llama_model * model,
    char ** out_no_tools,
    char ** out_with_tools,
    char ** out_error) {
    if (out_no_tools) {
        *out_no_tools = nullptr;
    }
    if (out_with_tools) {
        *out_with_tools = nullptr;
    }
    if (out_error) {
        *out_error = nullptr;
    }

    if (!model || !out_no_tools || !out_with_tools || !out_error) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    try {
        const char * tmpl_src = llama_model_chat_template(model, nullptr);
        if (!tmpl_src) {
            return LLAMA_RS_STATUS_OK;
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        if (!vocab) {
            return LLAMA_RS_STATUS_OK;
        }

        std::string bos_token = token_text_or_empty(vocab, llama_vocab_bos(vocab));
        std::string eos_token = token_text_or_empty(vocab, llama_vocab_eos(vocab));

        common_chat_template tmpl(tmpl_src, bos_token, eos_token);

        nlohmann::ordered_json user_msg = {
            { "role",    "user"                },
            { "content", "Please use the tool" }
        };
        nlohmann::ordered_json assistant_no_tools = {
            { "role",    "assistant"      },
            { "content", "Sure, calling." }
        };
        nlohmann::ordered_json first_tool_call = {
            { "id",       "call_001"  },
            { "type",     "function"  },
            { "function", {
                { "name",      "tool_first" },
                { "arguments", {
                    { "arg_first", "XXXX" },
                    { "arg_second", "YYYY" },
                }}
            }}
        };
        nlohmann::ordered_json assistant_with_tools = {
            { "role",       "assistant"                                                  },
            { "content",    ""                                                           },
            { "tool_calls", nlohmann::ordered_json::array({ first_tool_call })           }
        };
        nlohmann::ordered_json tool_definition = {
            { "type",     "function"  },
            { "function", {
                { "name",        "tool_first"           },
                { "description", "First test tool"      },
                { "parameters", {
                    { "type", "object" },
                    { "properties", {
                        { "arg_first", { { "type", "string" }, { "description", "first arg" } } },
                        { "arg_second", { { "type", "string" }, { "description", "second arg" } } },
                    }},
                    { "required", nlohmann::ordered_json::array({ "arg_first", "arg_second" }) },
                }}
            }}
        };

        template_params params_no_tools;
        params_no_tools.messages              = nlohmann::ordered_json::array({ user_msg, assistant_no_tools });
        params_no_tools.tools                 = nlohmann::ordered_json::array({ tool_definition });
        params_no_tools.add_generation_prompt = false;
        params_no_tools.enable_thinking       = true;

        template_params params_with_tools = params_no_tools;
        params_with_tools.messages =
            nlohmann::ordered_json::array({ user_msg, assistant_with_tools });

        std::string output_a = autoparser::apply_template(tmpl, params_no_tools);
        std::string output_b = autoparser::apply_template(tmpl, params_with_tools);

        char * a_dup = llama_rs_dup_string(output_a);
        char * b_dup = llama_rs_dup_string(output_b);

        if (!a_dup || !b_dup) {
            std::free(a_dup);
            std::free(b_dup);

            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }

        *out_no_tools = a_dup;
        *out_with_tools = b_dup;

        return LLAMA_RS_STATUS_OK;
    } catch (const std::exception & ex) {
        *out_error = llama_rs_dup_string(std::string(ex.what()));

        return LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        *out_error = llama_rs_dup_string(std::string("unknown c++ exception"));

        return LLAMA_RS_STATUS_EXCEPTION;
    }
}
