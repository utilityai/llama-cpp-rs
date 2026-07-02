#include "wrapper_gbnf.h"

#include "llama-grammar.h"
#include "unicode.h"

#include <cstdint>
#include <string>
#include <vector>

extern "C" {

llama_rs_gbnf_parse_status llama_rs_gbnf_parse(
    const char * grammar_str,
    const char * grammar_root,
    struct llama_grammar ** out_grammar) noexcept {
    llama_grammar_parser parser(nullptr);

    if (!parser.parse(grammar_str)) {
        return LLAMA_RS_GBNF_PARSE_SYNTAX_ERROR;
    }

    if (parser.rules.empty()) {
        return LLAMA_RS_GBNF_PARSE_EMPTY_RULE_SET;
    }

    const auto root_iterator = parser.symbol_ids.find(grammar_root);

    if (root_iterator == parser.symbol_ids.end()) {
        return LLAMA_RS_GBNF_PARSE_ROOT_SYMBOL_MISSING;
    }

    auto grammar_rules = parser.c_rules();
    llama_grammar * grammar = llama_grammar_init_impl(
        nullptr,
        grammar_rules.data(),
        grammar_rules.size(),
        root_iterator->second);

    if (grammar == nullptr) {
        return LLAMA_RS_GBNF_PARSE_LEFT_RECURSION;
    }

    *out_grammar = grammar;

    return LLAMA_RS_GBNF_PARSE_OK;
}

void llama_rs_gbnf_accept_str(
    struct llama_grammar * grammar,
    const char * piece,
    size_t piece_len) noexcept {
    const std::string input(piece, piece_len);
    const std::vector<uint32_t> code_points = unicode_cpts_from_utf8(input);
    auto & stacks = llama_grammar_get_stacks(grammar);

    for (const uint32_t code_point : code_points) {
        if (stacks.empty()) {
            break;
        }

        llama_grammar_accept(grammar, code_point);
    }
}

bool llama_rs_gbnf_is_accepting(struct llama_grammar * grammar) noexcept {
    auto & stacks = llama_grammar_get_stacks(grammar);

    for (const auto & stack : stacks) {
        if (stack.empty()) {
            return true;
        }
    }

    return false;
}

bool llama_rs_gbnf_is_rejected(struct llama_grammar * grammar) noexcept {
    return llama_grammar_get_stacks(grammar).empty();
}

struct llama_grammar * llama_rs_gbnf_clone(const struct llama_grammar * grammar) noexcept {
    return llama_grammar_clone_impl(*grammar);
}

void llama_rs_gbnf_free(struct llama_grammar * grammar) noexcept {
    llama_grammar_free_impl(grammar);
}

}
