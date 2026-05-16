use std::string::FromUtf8Error;

#[derive(Debug, thiserror::Error)]
pub enum MarkerDetectionError {
    #[error("ffi returned non-utf8 marker bytes: {0}")]
    MarkerUtf8Error(#[from] FromUtf8Error),
    #[error("llama_rs_detect_reasoning_markers called with null model")]
    DetectReasoningMarkersNullModelArg,
    #[error("llama_rs_detect_reasoning_markers called with null out_open")]
    DetectReasoningMarkersNullOutOpenArg,
    #[error("llama_rs_detect_reasoning_markers called with null out_close")]
    DetectReasoningMarkersNullOutCloseArg,
    #[error("llama_rs_detect_reasoning_markers called with null out_error")]
    DetectReasoningMarkersNullOutErrorArg,
    #[error(
        "llama_rs_detect_reasoning_markers wrapper failed to duplicate the C++ exception string"
    )]
    DetectReasoningMarkersErrorStringAllocationFailed,
    #[error("llama_rs_detect_reasoning_markers threw a C++ exception: {message}")]
    DetectReasoningMarkersVendoredThrewCxxException { message: String },
    #[error("llama_rs_compute_tool_call_haystack called with null model")]
    ComputeToolCallHaystackNullModelArg,
    #[error("llama_rs_compute_tool_call_haystack called with null out_haystack")]
    ComputeToolCallHaystackNullOutHaystackArg,
    #[error("llama_rs_compute_tool_call_haystack called with null out_error")]
    ComputeToolCallHaystackNullOutErrorArg,
    #[error(
        "llama_rs_compute_tool_call_haystack wrapper failed to duplicate the C++ exception string"
    )]
    ComputeToolCallHaystackErrorStringAllocationFailed,
    #[error("llama_rs_compute_tool_call_haystack threw a C++ exception: {message}")]
    ComputeToolCallHaystackVendoredThrewCxxException { message: String },
    #[error("llama_rs_diagnose_tool_call_synthetic_renders called with null model")]
    DiagnoseToolCallSyntheticRendersNullModelArg,
    #[error("llama_rs_diagnose_tool_call_synthetic_renders called with null out_no_tools")]
    DiagnoseToolCallSyntheticRendersNullOutNoToolsArg,
    #[error("llama_rs_diagnose_tool_call_synthetic_renders called with null out_with_tools")]
    DiagnoseToolCallSyntheticRendersNullOutWithToolsArg,
    #[error("llama_rs_diagnose_tool_call_synthetic_renders called with null out_error")]
    DiagnoseToolCallSyntheticRendersNullOutErrorArg,
    #[error(
        "llama_rs_diagnose_tool_call_synthetic_renders wrapper failed to duplicate the C++ exception string"
    )]
    DiagnoseToolCallSyntheticRendersErrorStringAllocationFailed,
    #[error("llama_rs_diagnose_tool_call_synthetic_renders threw a C++ exception: {message}")]
    DiagnoseToolCallSyntheticRendersVendoredThrewCxxException { message: String },
}
