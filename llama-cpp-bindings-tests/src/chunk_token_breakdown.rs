use anyhow::Context;
use anyhow::Result;
use llama_cpp_bindings::mtmd::mtmd_input_chunk_type::MtmdInputChunkType;
use llama_cpp_bindings::mtmd::mtmd_input_chunks::MtmdInputChunks;

pub struct ChunkTokenBreakdown {
    pub text: u64,
    pub image: u64,
    pub audio: u64,
}

impl ChunkTokenBreakdown {
    /// # Errors
    ///
    /// Forwards chunk access and chunk-type classification errors.
    pub fn from_chunks(chunks: &MtmdInputChunks) -> Result<Self> {
        let mut breakdown = Self {
            text: 0,
            image: 0,
            audio: 0,
        };
        for index in 0..chunks.len() {
            let chunk = chunks
                .get(index)
                .with_context(|| format!("chunk index {index} is missing"))?;
            let n_tokens = u64::try_from(chunk.n_tokens())?;
            match chunk.chunk_type()? {
                MtmdInputChunkType::Text => breakdown.text += n_tokens,
                MtmdInputChunkType::Image => breakdown.image += n_tokens,
                MtmdInputChunkType::Audio => breakdown.audio += n_tokens,
            }
        }

        Ok(breakdown)
    }
}
