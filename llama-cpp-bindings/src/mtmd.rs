//! Safe wrapper around multimodal (MTMD) functionality in llama.cpp.
//!
//! This module provides Rust bindings for llama.cpp's multimodal support,
//! allowing processing of text, image, and audio inputs through a unified interface.
//!
//! # Warning
//! This API is experimental and subject to breaking changes.

pub mod image_chunk_batch_size_mismatch;
pub mod mtmd_bitmap;
pub mod mtmd_context;
pub mod mtmd_context_params;
pub mod mtmd_default_marker;
pub mod mtmd_error;
pub mod mtmd_input_chunk;
pub mod mtmd_input_chunk_type;
pub mod mtmd_input_chunks;
pub mod mtmd_input_text;

pub use image_chunk_batch_size_mismatch::ImageChunkBatchSizeMismatch;
pub use mtmd_bitmap::MtmdBitmap;
pub use mtmd_context::MtmdContext;
pub use mtmd_context_params::MtmdContextParams;
pub use mtmd_default_marker::mtmd_default_marker;
pub use mtmd_error::{
    MtmdBitmapError, MtmdEncodeError, MtmdEvalError, MtmdInitError, MtmdInputChunkError,
    MtmdInputChunksError, MtmdTokenizeError,
};
pub use mtmd_input_chunk::MtmdInputChunk;
pub use mtmd_input_chunk_type::{MtmdInputChunkType, MtmdInputChunkTypeError};
pub use mtmd_input_chunks::MtmdInputChunks;
pub use mtmd_input_text::MtmdInputText;
