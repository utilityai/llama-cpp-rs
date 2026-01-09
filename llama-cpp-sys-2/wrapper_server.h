#ifndef WRAPPER_SERVER_H
#define WRAPPER_SERVER_H

#include "llama.cpp/include/llama.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//
// Opaque types
//

typedef struct llama_server_context llama_server_context;
typedef struct llama_server_response_reader llama_server_response_reader;
typedef struct llama_server_task llama_server_task;
typedef struct llama_server_task_result llama_server_task_result;

//
// Server context metadata (read-only info about loaded model)
//

typedef struct llama_server_context_meta {
    const char * build_info;
    const char * model_name;
    const char * model_path;
    bool has_mtmd;
    bool has_inp_image;
    bool has_inp_audio;
    int32_t slot_n_ctx;
    int32_t pooling_type;
    const char * chat_template;
    const char * bos_token_str;
    const char * eos_token_str;
    int32_t model_vocab_n_tokens;
    int32_t model_n_ctx_train;
    int32_t model_n_embd_inp;
    uint64_t model_n_params;
    uint64_t model_size;
} llama_server_context_meta;

//
// Task parameters for completion requests
//

typedef struct llama_server_task_params {
    bool stream;
    bool cache_prompt;
    bool return_tokens;
    bool return_progress;
    bool timings_per_token;
    bool post_sampling_probs;

    int32_t n_keep;
    int32_t n_discard;
    int32_t n_predict;
    int32_t n_indent;
    int32_t seed;

    float temperature;
    float top_p;
    float min_p;
    float typical_p;
    float repeat_penalty;
    float presence_penalty;
    float frequency_penalty;

    int32_t top_k;
    int32_t repeat_last_n;
    int32_t mirostat;
    float mirostat_tau;
    float mirostat_eta;

    // Antiprompt strings (stop sequences)
    const char ** antiprompt;
    size_t antiprompt_count;
} llama_server_task_params;

//
// Result timings
//

typedef struct llama_server_result_timings {
    int32_t cache_n;
    int32_t prompt_n;
    double prompt_ms;
    double prompt_per_token_ms;
    double prompt_per_second;
    int32_t predicted_n;
    double predicted_ms;
    double predicted_per_token_ms;
    double predicted_per_second;
    int32_t draft_n;
    int32_t draft_n_accepted;
} llama_server_result_timings;

//
// Result types
//

typedef enum llama_server_result_type {
    LLAMA_SERVER_RESULT_TYPE_UNKNOWN = 0,
    LLAMA_SERVER_RESULT_TYPE_ERROR = 1,
    LLAMA_SERVER_RESULT_TYPE_PARTIAL = 2,
    LLAMA_SERVER_RESULT_TYPE_FINAL = 3,
} llama_server_result_type;

//
// Message diff for streaming (content deltas)
//

typedef struct llama_server_msg_diff {
    const char * content_delta;
    const char * reasoning_content_delta;
    const char * tool_call_id;
    const char * tool_call_name;
    const char * tool_call_arguments;
} llama_server_msg_diff;

//
// Server context functions
//

// Create a new server context
llama_server_context * llama_server_context_new(void);

// Free a server context
void llama_server_context_free(llama_server_context * ctx);

// Load model with common_params JSON string
// Returns true on success
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
    int32_t flash_attn_type,  // llama_flash_attn_type enum: -1=auto, 0=disabled, 1=enabled
    const char * chat_template,  // can be NULL
    const char * system_prompt   // can be NULL
);

// Start the server loop (blocking - run in a separate thread)
void llama_server_context_start_loop(llama_server_context * ctx);

// Terminate the server loop
void llama_server_context_terminate(llama_server_context * ctx);

// Get the underlying llama_context (can be NULL if sleeping)
struct llama_context * llama_server_context_get_llama_context(llama_server_context * ctx);

// Get server metadata (caller must free with llama_server_context_meta_free)
llama_server_context_meta * llama_server_context_get_meta(llama_server_context * ctx);

// Free metadata
void llama_server_context_meta_free(llama_server_context_meta * meta);

//
// Response reader functions
//

// Get a new response reader from the server context
llama_server_response_reader * llama_server_response_reader_new(llama_server_context * ctx);

// Free a response reader
void llama_server_response_reader_free(llama_server_response_reader * reader);

// Get a new task ID
int32_t llama_server_response_reader_get_new_id(llama_server_response_reader * reader);

// Post a completion task with chat messages (JSON array string)
// Returns true on success
bool llama_server_response_reader_post_completion(
    llama_server_response_reader * reader,
    int32_t task_id,
    const llama_server_task_params * params,
    const char * messages_json,  // JSON array of {role, content} objects
    const uint8_t ** file_buffers,  // array of file buffer pointers (can be NULL)
    const size_t * file_sizes,      // array of file sizes
    size_t file_count               // number of files
);

// Check if there are more results to read
bool llama_server_response_reader_has_next(llama_server_response_reader * reader);

// Get next result (blocks until available or should_stop returns true)
// Returns NULL if stopped or no more results
// Caller must free result with llama_server_task_result_free
typedef bool (*llama_server_should_stop_fn)(void * user_data);
llama_server_task_result * llama_server_response_reader_next(
    llama_server_response_reader * reader,
    llama_server_should_stop_fn should_stop,
    void * user_data
);

// Stop/cancel the reader
void llama_server_response_reader_stop(llama_server_response_reader * reader);

//
// Task result functions
//

// Free a task result
void llama_server_task_result_free(llama_server_task_result * result);

// Get result type
llama_server_result_type llama_server_task_result_get_type(llama_server_task_result * result);

// Check if result is an error
bool llama_server_task_result_is_error(llama_server_task_result * result);

// Check if result is a stop (final result)
bool llama_server_task_result_is_stop(llama_server_task_result * result);

// Get error message (only valid if is_error returns true)
// Returns NULL if not an error. Caller must free with llama_server_string_free
char * llama_server_task_result_get_error(llama_server_task_result * result);

// Get timings from result
llama_server_result_timings llama_server_task_result_get_timings(llama_server_task_result * result);

// Get number of message diffs (for partial results)
size_t llama_server_task_result_get_diff_count(llama_server_task_result * result);

// Get message diff at index
// Returns false if index out of bounds
// Strings in diff are valid until result is freed
bool llama_server_task_result_get_diff(
    llama_server_task_result * result,
    size_t index,
    llama_server_msg_diff * out_diff
);

// Get full content (for final results)
// Caller must free with llama_server_string_free
char * llama_server_task_result_get_content(llama_server_task_result * result);

// Get result as JSON string
// Caller must free with llama_server_string_free
char * llama_server_task_result_to_json(llama_server_task_result * result);

//
// Utility functions
//

// Free a string returned by other functions
void llama_server_string_free(char * str);

// Get default task params
llama_server_task_params llama_server_task_params_default(void);

#ifdef __cplusplus
}
#endif

#endif // WRAPPER_SERVER_H
