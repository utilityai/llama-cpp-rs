#include "wrapper_common.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "llama.cpp/common/chat.h"
#include "llama.cpp/common/json-schema-to-grammar.h"
#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#include <nlohmann/json.hpp>

static bool dup_string_array(
    const std::vector<std::string> & values,
    char *** out_items,
    size_t * out_count) {
    if (!out_items || !out_count) {
        return false;
    }
    *out_items = nullptr;
    *out_count = 0;
    if (values.empty()) {
        return true;
    }

    char ** items =
        static_cast<char **>(std::malloc(sizeof(char *) * values.size()));
    if (!items) {
        return false;
    }
    for (size_t i = 0; i < values.size(); ++i) {
        items[i] = llama_rs_dup_string(values[i]);
        if (!items[i]) {
            for (size_t j = 0; j < i; ++j) {
                std::free(items[j]);
            }
            std::free(items);
            return false;
        }
    }
    *out_items = items;
    *out_count = values.size();
    return true;
}

static bool dup_trigger_array(
    const std::vector<common_grammar_trigger> & triggers,
    struct llama_rs_grammar_trigger ** out_items,
    size_t * out_count) {
    if (!out_items || !out_count) {
        return false;
    }
    *out_items = nullptr;
    *out_count = 0;
    if (triggers.empty()) {
        return true;
    }

    auto * items = static_cast<struct llama_rs_grammar_trigger *>(
        std::malloc(sizeof(struct llama_rs_grammar_trigger) * triggers.size()));
    if (!items) {
        return false;
    }
    for (size_t i = 0; i < triggers.size(); ++i) {
        items[i].type = static_cast<int>(triggers[i].type);
        items[i].token = triggers[i].token;
        items[i].value = llama_rs_dup_string(triggers[i].value);
        if (!items[i].value && !triggers[i].value.empty()) {
            for (size_t j = 0; j < i; ++j) {
                std::free(items[j].value);
            }
            std::free(items);
            return false;
        }
    }
    *out_items = items;
    *out_count = triggers.size();
    return true;
}

extern "C" int llama_rs_json_schema_to_grammar(
    const char * schema_json,
    bool force_gbnf,
    char ** out_grammar) {
    if (!schema_json || !out_grammar) {
        return -1;
    }

    *out_grammar = nullptr;
    try {
        const auto schema = nlohmann::ordered_json::parse(schema_json);
        const auto grammar = json_schema_to_grammar(schema, force_gbnf);
        *out_grammar = llama_rs_dup_string(grammar);
        return *out_grammar ? 0 : -2;
    } catch (const std::exception &) {
        return -3;
    }
}

extern "C" int llama_rs_apply_chat_template_with_tools(
    const struct llama_model * model,
    const char * chat_template,
    const struct llama_chat_message * messages,
    size_t message_count,
    const char * tools_json,
    const char * json_schema,
    bool add_generation_prompt,
    struct llama_rs_chat_template_result * out_result) {
    if (!chat_template || !out_result) {
        return -1;
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
        if (!dup_trigger_array(
                params.grammar_triggers,
                &out_result->grammar_triggers,
                &out_result->grammar_triggers_count)) {
            llama_rs_chat_template_result_free(out_result);
            return -2;
        }
        if (!dup_string_array(
                params.preserved_tokens,
                &out_result->preserved_tokens,
                &out_result->preserved_tokens_count)) {
            llama_rs_chat_template_result_free(out_result);
            return -2;
        }
        if (!dup_string_array(
                params.additional_stops,
                &out_result->additional_stops,
                &out_result->additional_stops_count)) {
            llama_rs_chat_template_result_free(out_result);
            return -2;
        }
        if (!out_result->prompt) {
            llama_rs_chat_template_result_free(out_result);
            return -2;
        }
        return 0;
    } catch (const std::exception &) {
        llama_rs_chat_template_result_free(out_result);
        return -3;
    }
}

extern "C" void llama_rs_chat_template_result_free(struct llama_rs_chat_template_result * result) {
    if (!result) {
        return;
    }
    if (result->prompt) {
        std::free(result->prompt);
    }
    if (result->grammar) {
        std::free(result->grammar);
    }
    if (result->parser) {
        std::free(result->parser);
    }
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
    result->chat_format = 0;
    result->thinking_forced_open = false;
    result->grammar_lazy = false;
    result->grammar_triggers = nullptr;
    result->grammar_triggers_count = 0;
    result->preserved_tokens = nullptr;
    result->preserved_tokens_count = 0;
    result->additional_stops = nullptr;
    result->additional_stops_count = 0;
}

extern "C" void llama_rs_string_free(char * ptr) {
    if (ptr) {
        std::free(ptr);
    }
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

extern "C" int llama_rs_sampler_accept(struct llama_sampler * sampler, llama_token token) {
    if (!sampler) {
        return -1;
    }
    try {
        llama_sampler_accept(sampler, token);
        return 0;
    } catch (const std::exception &) {
        return -2;
    } catch (...) {
        return -3;
    }
}
