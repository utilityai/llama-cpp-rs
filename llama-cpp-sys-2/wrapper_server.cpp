#include "wrapper_server.h"

// Include llama.cpp server infrastructure headers
#include "llama.cpp/common/common.h"
#include "llama.cpp/common/arg.h"
#include "llama.cpp/tools/server/server-context.h"
#include "llama.cpp/tools/server/server-task.h"
#include "llama.cpp/tools/server/server-queue.h"
#include "llama.cpp/tools/server/server-common.h"

#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <functional>

//
// Helper to duplicate C strings
//

static char * strdup_safe(const char * s) {
    if (!s) return nullptr;
    size_t len = strlen(s);
    char * copy = (char *)malloc(len + 1);
    if (copy) {
        memcpy(copy, s, len + 1);
    }
    return copy;
}

static char * strdup_safe(const std::string & s) {
    return strdup_safe(s.c_str());
}

//
// Wrapper structures
//

struct llama_server_context {
    server_context ctx;
    common_params params;
    std::string system_prompt;
};

struct llama_server_response_reader {
    std::unique_ptr<server_response_reader> reader;
    llama_server_context * parent;
};

struct llama_server_task_result {
    server_task_result_ptr result;
    llama_server_result_type type;
    
    // Cached data for C API access
    std::vector<llama_server_msg_diff> cached_diffs;
    std::vector<std::string> cached_strings;  // Keep strings alive
    std::string cached_content;
    std::string cached_error;
    std::string cached_json;
    result_timings timings;
};

//
// Server context implementation
//

extern "C" {

llama_server_context * llama_server_context_new(void) {
    return new llama_server_context();
}

void llama_server_context_free(llama_server_context * ctx) {
    if (ctx) {
        delete ctx;
    }
}

bool llama_server_context_load_model(
    llama_server_context * ctx,
    const char * model_path,
    int32_t n_ctx,
    int32_t n_batch,
    int32_t n_ubatch,
    int32_t n_threads,
    int32_t n_threads_batch,
    int32_t n_gpu_layers,
    bool use_mmap,
    bool use_mlock,
    int32_t flash_attn_type,
    const char * chat_template,
    const char * system_prompt
) {
    if (!ctx || !model_path) {
        return false;
    }

    // Set up common_params
    ctx->params.model.path = model_path;
    ctx->params.n_ctx = n_ctx > 0 ? n_ctx : 4096;
    ctx->params.n_batch = n_batch > 0 ? n_batch : 2048;
    ctx->params.n_ubatch = n_ubatch > 0 ? n_ubatch : 512;
    ctx->params.cpuparams.n_threads = n_threads > 0 ? n_threads : -1;
    ctx->params.cpuparams_batch.n_threads = n_threads_batch > 0 ? n_threads_batch : -1;
    ctx->params.n_gpu_layers = n_gpu_layers;
    ctx->params.use_mmap = use_mmap;
    ctx->params.use_mlock = use_mlock;
    ctx->params.flash_attn_type = static_cast<llama_flash_attn_type>(flash_attn_type);
    
    if (chat_template && strlen(chat_template) > 0) {
        ctx->params.chat_template = chat_template;
    }
    
    if (system_prompt && strlen(system_prompt) > 0) {
        ctx->system_prompt = system_prompt;
    }

    // Initialize common (logging, etc.)
    common_init();

    return ctx->ctx.load_model(ctx->params);
}

void llama_server_context_start_loop(llama_server_context * ctx) {
    if (ctx) {
        ctx->ctx.start_loop();
    }
}

void llama_server_context_terminate(llama_server_context * ctx) {
    if (ctx) {
        ctx->ctx.terminate();
    }
}

llama_context * llama_server_context_get_llama_context(llama_server_context * ctx) {
    if (!ctx) return nullptr;
    return ctx->ctx.get_llama_context();
}

llama_server_context_meta * llama_server_context_get_meta(llama_server_context * ctx) {
    if (!ctx) return nullptr;
    
    auto meta = ctx->ctx.get_meta();
    auto * result = new llama_server_context_meta();
    
    result->build_info = strdup_safe(meta.build_info);
    result->model_name = strdup_safe(meta.model_name);
    result->model_path = strdup_safe(meta.model_path);
    result->has_mtmd = meta.has_mtmd;
    result->has_inp_image = meta.has_inp_image;
    result->has_inp_audio = meta.has_inp_audio;
    result->slot_n_ctx = meta.slot_n_ctx;
    result->pooling_type = (int32_t)meta.pooling_type;
    result->chat_template = strdup_safe("");  // No longer directly available in meta
    result->bos_token_str = strdup_safe(meta.bos_token_str);
    result->eos_token_str = strdup_safe(meta.eos_token_str);
    result->model_vocab_n_tokens = meta.model_vocab_n_tokens;
    result->model_n_ctx_train = meta.model_n_ctx_train;
    result->model_n_embd_inp = meta.model_n_embd_inp;
    result->model_n_params = meta.model_n_params;
    result->model_size = meta.model_size;
    
    return result;
}

void llama_server_context_meta_free(llama_server_context_meta * meta) {
    if (meta) {
        free((void *)meta->build_info);
        free((void *)meta->model_name);
        free((void *)meta->model_path);
        free((void *)meta->chat_template);
        free((void *)meta->bos_token_str);
        free((void *)meta->eos_token_str);
        delete meta;
    }
}

//
// Response reader implementation
//

llama_server_response_reader * llama_server_response_reader_new(llama_server_context * ctx) {
    if (!ctx) return nullptr;
    
    auto * reader = new llama_server_response_reader();
    reader->parent = ctx;
    // We need to store the reader as a unique_ptr since server_response_reader is non-copyable
    auto rd = ctx->ctx.get_response_reader();
    reader->reader = std::make_unique<server_response_reader>(std::move(rd));
    return reader;
}

void llama_server_response_reader_free(llama_server_response_reader * reader) {
    if (reader) {
        delete reader;
    }
}

int32_t llama_server_response_reader_get_new_id(llama_server_response_reader * reader) {
    if (!reader || !reader->reader) return -1;
    return reader->reader->get_new_id();
}

bool llama_server_response_reader_post_completion(
    llama_server_response_reader * reader,
    int32_t task_id,
    const llama_server_task_params * params,
    const char * messages_json,
    const uint8_t ** file_buffers,
    const size_t * file_sizes,
    size_t file_count
) {
    if (!reader || !reader->reader || !messages_json) {
        return false;
    }

    try {
        server_task task(SERVER_TASK_TYPE_COMPLETION);
        task.id = task_id;
        task.index = 0;
        
        // Set task parameters
        if (params) {
            task.params.stream = params->stream;
            task.params.cache_prompt = params->cache_prompt;
            task.params.return_tokens = params->return_tokens;
            task.params.return_progress = params->return_progress;
            task.params.timings_per_token = params->timings_per_token;
            task.params.post_sampling_probs = params->post_sampling_probs;
            task.params.n_keep = params->n_keep;
            task.params.n_discard = params->n_discard;
            task.params.n_predict = params->n_predict;
            task.params.n_indent = params->n_indent;
            
            // Sampling params
            task.params.sampling.seed = params->seed;
            task.params.sampling.temp = params->temperature;
            task.params.sampling.top_p = params->top_p;
            task.params.sampling.min_p = params->min_p;
            task.params.sampling.typ_p = params->typical_p;
            task.params.sampling.penalty_repeat = params->repeat_penalty;
            task.params.sampling.penalty_present = params->presence_penalty;
            task.params.sampling.penalty_freq = params->frequency_penalty;
            task.params.sampling.top_k = params->top_k;
            task.params.sampling.penalty_last_n = params->repeat_last_n;
            task.params.sampling.mirostat = params->mirostat;
            task.params.sampling.mirostat_tau = params->mirostat_tau;
            task.params.sampling.mirostat_eta = params->mirostat_eta;
            
            // Antiprompt
            if (params->antiprompt && params->antiprompt_count > 0) {
                for (size_t i = 0; i < params->antiprompt_count; i++) {
                    if (params->antiprompt[i]) {
                        task.params.antiprompt.push_back(params->antiprompt[i]);
                    }
                }
            }
        } else {
            // Defaults for streaming CLI
            task.params.stream = true;
            task.params.timings_per_token = true;
        }
        
        // Format chat using the same mechanism as cli.cpp
        task.cli = true;
        
        // Get chat template metadata
        auto meta = reader->parent->ctx.get_meta();
        auto & chat_params = meta.chat_params;
        
        // Parse messages JSON string into a JSON object first, then parse to chat messages
        // This matches how cli.cpp works where messages is a json::array()
        json messages = json::parse(messages_json);
        
        // Prepare chat template inputs
        common_chat_templates_inputs inputs;
        inputs.messages              = common_chat_msgs_parse_oaicompat(messages);
        inputs.tools                 = {};
        inputs.tool_choice           = COMMON_CHAT_TOOL_CHOICE_NONE;
        inputs.json_schema           = "";
        inputs.grammar               = "";
        inputs.use_jinja             = chat_params.use_jinja;
        inputs.parallel_tool_calls   = false;
        inputs.add_generation_prompt = true;
        inputs.enable_thinking       = chat_params.enable_thinking;
        
        // Apply chat template to format the messages
        auto formatted = common_chat_templates_apply(chat_params.tmpls.get(), inputs);
        task.cli_prompt = formatted.prompt;
        
        // Set chat parser params
        task.params.chat_parser_params = common_chat_parser_params(formatted);
        task.params.chat_parser_params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        if (!formatted.parser.empty()) {
            task.params.chat_parser_params.parser.load(formatted.parser);
        }
        
        // Add files if provided
        if (file_buffers && file_sizes && file_count > 0) {
            for (size_t i = 0; i < file_count; i++) {
                if (file_buffers[i] && file_sizes[i] > 0) {
                    raw_buffer buf;
                    buf.assign(file_buffers[i], file_buffers[i] + file_sizes[i]);
                    task.cli_files.push_back(std::move(buf));
                }
            }
        }
        
        std::vector<server_task> tasks;
        tasks.push_back(std::move(task));
        reader->reader->post_tasks(std::move(tasks));
        
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

bool llama_server_response_reader_has_next(llama_server_response_reader * reader) {
    if (!reader || !reader->reader) return false;
    return reader->reader->has_next();
}

llama_server_task_result * llama_server_response_reader_next(
    llama_server_response_reader * reader,
    llama_server_should_stop_fn should_stop,
    void * user_data
) {
    if (!reader || !reader->reader) return nullptr;

    // Create a std::function wrapper for the callback
    std::function<bool()> stop_fn;
    if (should_stop) {
        stop_fn = [should_stop, user_data]() {
            return should_stop(user_data);
        };
    } else {
        stop_fn = []() { return false; };
    }

    server_task_result_ptr result = reader->reader->next(stop_fn);
    if (!result) {
        return nullptr;
    }

    auto * wrapper = new llama_server_task_result();
    
    // Determine type and cache data
    if (result->is_error()) {
        wrapper->type = LLAMA_SERVER_RESULT_TYPE_ERROR;
        json err_json = result->to_json();
        if (err_json.contains("message")) {
            wrapper->cached_error = err_json["message"].get<std::string>();
        } else {
            wrapper->cached_error = err_json.dump();
        }
    } else {
        // Try to cast to partial or final
        auto * partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get());
        auto * final_res = dynamic_cast<server_task_result_cmpl_final *>(result.get());
        
        if (partial) {
            wrapper->type = LLAMA_SERVER_RESULT_TYPE_PARTIAL;
            wrapper->timings = partial->timings;
            
            // Cache message diffs
            for (const auto & diff : partial->oaicompat_msg_diffs) {
                llama_server_msg_diff c_diff = {};
                
                if (!diff.content_delta.empty()) {
                    wrapper->cached_strings.push_back(diff.content_delta);
                    c_diff.content_delta = wrapper->cached_strings.back().c_str();
                }
                if (!diff.reasoning_content_delta.empty()) {
                    wrapper->cached_strings.push_back(diff.reasoning_content_delta);
                    c_diff.reasoning_content_delta = wrapper->cached_strings.back().c_str();
                }
                // Tool call fields would be added here if needed
                
                wrapper->cached_diffs.push_back(c_diff);
            }
        } else if (final_res) {
            wrapper->type = LLAMA_SERVER_RESULT_TYPE_FINAL;
            wrapper->timings = final_res->timings;
            wrapper->cached_content = final_res->content;
        } else {
            wrapper->type = LLAMA_SERVER_RESULT_TYPE_UNKNOWN;
        }
    }
    
    wrapper->result = std::move(result);
    return wrapper;
}

void llama_server_response_reader_stop(llama_server_response_reader * reader) {
    if (reader && reader->reader) {
        reader->reader->stop();
    }
}

//
// Task result implementation
//

void llama_server_task_result_free(llama_server_task_result * result) {
    if (result) {
        delete result;
    }
}

llama_server_result_type llama_server_task_result_get_type(llama_server_task_result * result) {
    if (!result) return LLAMA_SERVER_RESULT_TYPE_UNKNOWN;
    return result->type;
}

bool llama_server_task_result_is_error(llama_server_task_result * result) {
    if (!result || !result->result) return false;
    return result->result->is_error();
}

bool llama_server_task_result_is_stop(llama_server_task_result * result) {
    if (!result || !result->result) return true;
    return result->result->is_stop();
}

char * llama_server_task_result_get_error(llama_server_task_result * result) {
    if (!result || result->cached_error.empty()) return nullptr;
    return strdup_safe(result->cached_error);
}

llama_server_result_timings llama_server_task_result_get_timings(llama_server_task_result * result) {
    llama_server_result_timings t = {};
    if (result) {
        t.cache_n = result->timings.cache_n;
        t.prompt_n = result->timings.prompt_n;
        t.prompt_ms = result->timings.prompt_ms;
        t.prompt_per_token_ms = result->timings.prompt_per_token_ms;
        t.prompt_per_second = result->timings.prompt_per_second;
        t.predicted_n = result->timings.predicted_n;
        t.predicted_ms = result->timings.predicted_ms;
        t.predicted_per_token_ms = result->timings.predicted_per_token_ms;
        t.predicted_per_second = result->timings.predicted_per_second;
        t.draft_n = result->timings.draft_n;
        t.draft_n_accepted = result->timings.draft_n_accepted;
    }
    return t;
}

size_t llama_server_task_result_get_diff_count(llama_server_task_result * result) {
    if (!result) return 0;
    return result->cached_diffs.size();
}

bool llama_server_task_result_get_diff(
    llama_server_task_result * result,
    size_t index,
    llama_server_msg_diff * out_diff
) {
    if (!result || !out_diff || index >= result->cached_diffs.size()) {
        return false;
    }
    *out_diff = result->cached_diffs[index];
    return true;
}

char * llama_server_task_result_get_content(llama_server_task_result * result) {
    if (!result || result->cached_content.empty()) return nullptr;
    return strdup_safe(result->cached_content);
}

char * llama_server_task_result_to_json(llama_server_task_result * result) {
    if (!result || !result->result) return nullptr;
    if (result->cached_json.empty()) {
        result->cached_json = result->result->to_json().dump();
    }
    return strdup_safe(result->cached_json);
}

//
// Utility functions
//

void llama_server_string_free(char * str) {
    free(str);
}

llama_server_task_params llama_server_task_params_default(void) {
    llama_server_task_params params = {};
    
    params.stream = true;
    params.cache_prompt = true;
    params.return_tokens = false;
    params.return_progress = false;
    params.timings_per_token = true;
    params.post_sampling_probs = false;
    
    params.n_keep = 0;
    params.n_discard = 0;
    params.n_predict = -1;
    params.n_indent = 0;
    params.seed = LLAMA_DEFAULT_SEED;
    
    params.temperature = 0.8f;
    params.top_p = 0.95f;
    params.min_p = 0.05f;
    params.typical_p = 1.0f;
    params.repeat_penalty = 1.1f;
    params.presence_penalty = 0.0f;
    params.frequency_penalty = 0.0f;
    
    params.top_k = 40;
    params.repeat_last_n = 64;
    params.mirostat = 0;
    params.mirostat_tau = 5.0f;
    params.mirostat_eta = 0.1f;
    
    params.antiprompt = nullptr;
    params.antiprompt_count = 0;
    
    return params;
}

} // extern "C"
