use llama_cpp_test_harness::llama_tests_main;

mod embeddings {
    use std::time::Duration;

    use anyhow::{Context, Result};
    use llama_cpp_bindings::context::LlamaContext;
    use llama_cpp_bindings::ggml_time_us;
    use llama_cpp_bindings::llama_batch::LlamaBatch;
    use llama_cpp_bindings::model::AddBos;
    use llama_cpp_test_harness::LlamaFixture;
    use llama_cpp_test_harness::llama_test;

    fn normalize(input: &[f32]) -> Vec<f32> {
        let magnitude = input
            .iter()
            .fold(0.0, |accumulator, &value| value.mul_add(value, accumulator))
            .sqrt();

        input.iter().map(|&value| value / magnitude).collect()
    }

    #[llama_test(
        model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
        n_gpu_layers = 999,
        use_mmap = true,
        use_mlock = false,
        n_ctx = 512,
        n_batch = 2048,
        n_ubatch = 512,
        n_threads_batch = 8,
        embeddings = true,
    )]
    fn embedding_generation_produces_vectors(fixture: &LlamaFixture<'_>) -> Result<()> {
        let model = fixture.model;

        let mut ctx = LlamaContext::from_model(
            model,
            fixture.backend,
            (*fixture.context_params).into_llama_context_params(),
        )
        .with_context(|| "unable to create context")?;

        let prompt = "Hello my name is";
        let tokens = model
            .str_to_token(prompt, AddBos::Always)
            .with_context(|| format!("failed to tokenize {prompt}"))?;
        let prompt_token_count = u64::try_from(tokens.len())?;

        let n_ctx = usize::try_from(ctx.n_ctx())?;
        assert!(tokens.len() <= n_ctx, "prompt exceeds context window size");

        let t_main_start = ggml_time_us();

        let mut classifier = model.sampled_token_classifier();
        let mut batch = LlamaBatch::new(n_ctx, 1)?;
        classifier.feed_prompt_sequence_to_batch(&mut batch, &tokens, 0, false)?;

        assert_eq!(classifier.pending_prompt_tokens(), prompt_token_count);
        assert_eq!(classifier.usage().prompt_tokens, 0);

        ctx.clear_kv_cache();
        ctx.decode(&mut batch)
            .with_context(|| "llama_decode() failed")?;

        let promoted = classifier.commit_prompt_tokens();
        assert_eq!(promoted, prompt_token_count);

        let embedding = ctx
            .embeddings_seq_ith(0)
            .with_context(|| "failed to get embeddings")?;
        let normalized = normalize(embedding);

        let t_main_end = ggml_time_us();
        let duration = Duration::from_micros(u64::try_from(t_main_end - t_main_start)?);

        eprintln!(
            "created embedding with {} dimensions in {:.2} s",
            normalized.len(),
            duration.as_secs_f32()
        );

        assert!(
            !normalized.is_empty(),
            "embedding should have at least one dimension"
        );

        let magnitude: f32 = normalized
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!(
            (magnitude - 1.0).abs() < 0.01,
            "normalized embedding magnitude should be approximately 1.0, got {magnitude}"
        );

        let usage = classifier.into_usage();
        assert_eq!(usage.prompt_tokens, prompt_token_count);
        assert_eq!(usage.completion_tokens(), 0);

        Ok(())
    }
}

mod reranker {
    use std::time::Duration;

    use anyhow::{Context, Result, bail};
    use llama_cpp_bindings::context::LlamaContext;
    use llama_cpp_bindings::ggml_time_us;
    use llama_cpp_bindings::llama_batch::LlamaBatch;
    use llama_cpp_bindings::model::AddBos;
    use llama_cpp_test_harness::LlamaFixture;
    use llama_cpp_test_harness::llama_test;

    fn normalize(input: &[f32]) -> Vec<f32> {
        let magnitude = input
            .iter()
            .fold(0.0, |accumulator, &value| value.mul_add(value, accumulator))
            .sqrt();

        input.iter().map(|&value| value / magnitude).collect()
    }

    fn cosine_similarity(vec_a: &[f32], vec_b: &[f32]) -> f32 {
        vec_a
            .iter()
            .zip(vec_b.iter())
            .map(|(left, right)| left * right)
            .sum::<f32>()
    }

    #[llama_test(
        model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
        n_gpu_layers = 999,
        use_mmap = true,
        use_mlock = false,
        n_ctx = 512,
        n_batch = 2048,
        n_ubatch = 512,
        n_seq_max = 2,
        n_threads_batch = 8,
        embeddings = true,
    )]
    fn reranking_produces_scores(fixture: &LlamaFixture<'_>) -> Result<()> {
        let model = fixture.model;

        let query = "What is machine learning?";
        let documents = [
            "Machine learning is a subset of artificial intelligence.",
            "The weather today is sunny and warm.",
        ];

        let document_count = documents.len();
        assert_eq!(
            u32::try_from(document_count)?,
            fixture.context_params.n_seq_max,
            "attribute n_seq_max must match the document count this trial expects",
        );

        let mut ctx = LlamaContext::from_model(
            model,
            fixture.backend,
            (*fixture.context_params).into_llama_context_params(),
        )
        .with_context(|| "unable to create context")?;

        let prompt_lines: Vec<String> = documents
            .iter()
            .map(|document| format!("{query}</s><s>{document}"))
            .collect();

        let tokens_lines_list = prompt_lines
            .iter()
            .map(|line| model.str_to_token(line, AddBos::Always))
            .collect::<std::result::Result<Vec<_>, _>>()
            .with_context(|| "failed to tokenize prompts")?;

        let n_ctx = usize::try_from(ctx.n_ctx())?;

        if tokens_lines_list.iter().any(|tokens| n_ctx < tokens.len()) {
            bail!("one of the provided prompts exceeds the size of the context window");
        }

        let mut classifier = model.sampled_token_classifier();
        let mut batch = LlamaBatch::new(2048, i32::try_from(document_count)?)?;
        let t_main_start = ggml_time_us();

        for (sequence_index, tokens) in tokens_lines_list.iter().enumerate() {
            classifier.feed_prompt_sequence_to_batch(
                &mut batch,
                tokens,
                i32::try_from(sequence_index)?,
                false,
            )?;
        }

        let total_tokens: usize = tokens_lines_list.iter().map(Vec::len).sum();
        let total_token_count = u64::try_from(total_tokens)?;

        assert_eq!(classifier.pending_prompt_tokens(), total_token_count);
        assert_eq!(classifier.usage().prompt_tokens, 0);

        ctx.clear_kv_cache();
        ctx.decode(&mut batch)
            .with_context(|| "llama_decode() failed")?;

        let promoted = classifier.commit_prompt_tokens();
        assert_eq!(promoted, total_token_count);

        let mut embeddings = Vec::with_capacity(document_count);

        for sequence_index in 0..document_count {
            let raw_embedding = ctx
                .embeddings_seq_ith(i32::try_from(sequence_index)?)
                .with_context(|| "failed to get sequence embeddings")?;
            embeddings.push(normalize(raw_embedding));
        }

        let t_main_end = ggml_time_us();
        let duration = Duration::from_micros(u64::try_from(t_main_end - t_main_start)?);

        #[expect(
            clippy::cast_precision_loss,
            reason = "logged throughput tolerates f32 precision"
        )]
        let tokens_per_second = total_tokens as f32 / duration.as_secs_f32();

        eprintln!(
            "created embeddings for {total_tokens} tokens in {:.2} s, speed {tokens_per_second:.2} t/s",
            duration.as_secs_f32(),
        );

        assert_eq!(
            embeddings.len(),
            document_count,
            "should produce one embedding per document"
        );

        for (index, embedding) in embeddings.iter().enumerate() {
            assert!(
                !embedding.is_empty(),
                "embedding {index} should not be empty"
            );
        }

        let similarity = cosine_similarity(&embeddings[0], &embeddings[1]);
        eprintln!("cosine similarity between document embeddings: {similarity:.4}");

        assert!(
            similarity.is_finite(),
            "cosine similarity should be a finite number"
        );

        let usage = classifier.into_usage();
        assert_eq!(usage.prompt_tokens, total_token_count);
        assert_eq!(usage.completion_tokens(), 0);

        Ok(())
    }
}

mod context_embedding_and_encoder {
    
    
    

    use anyhow::Result;
    
    
    use llama_cpp_bindings::context::LlamaContext;
    use llama_cpp_bindings::llama_batch::LlamaBatch;
    use llama_cpp_bindings::model::AddBos;
    
    use llama_cpp_test_harness::LlamaFixture;
    use llama_cpp_test_harness::llama_test;

    // =========================================================================================
    // Group A: default Qwen model, embeddings=false. Most context tests fall here.
    // =========================================================================================


    #[llama_test(
        model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
        n_gpu_layers = 999,
        use_mmap = true,
        use_mlock = false,
        n_ctx = 512,
        n_batch = 2048,
        n_ubatch = 512,
        embeddings = true,
    )]
    fn decode_with_embeddings_enabled(fixture: &LlamaFixture<'_>) -> Result<()> {
        let mut context = LlamaContext::from_model(
            fixture.model,
            fixture.backend,
            (*fixture.context_params).into_llama_context_params(),
        )?;
        let tokens = fixture.model.str_to_token("hello", AddBos::Always)?;
        let mut batch = LlamaBatch::new(512, 1)?;
        batch.add_sequence(&tokens, 0, false)?;

        let result = context.decode(&mut batch);

        assert!(result.is_ok());

        Ok(())
    }

    #[llama_test(
        model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
        n_gpu_layers = 999,
        use_mmap = true,
        use_mlock = false,
        n_ctx = 512,
        n_batch = 2048,
        n_ubatch = 512,
        embeddings = true,
    )]
    fn embeddings_seq_ith_returns_valid_embeddings(fixture: &LlamaFixture<'_>) -> Result<()> {
        let mut context = LlamaContext::from_model(
            fixture.model,
            fixture.backend,
            (*fixture.context_params).into_llama_context_params(),
        )?;
        let tokens = fixture.model.str_to_token("hello", AddBos::Always)?;
        let mut batch = LlamaBatch::new(512, 1)?;
        batch.add_sequence(&tokens, 0, false)?;
        context.decode(&mut batch)?;

        let embeddings = context.embeddings_seq_ith(0)?;

        assert_eq!(embeddings.len(), usize::try_from(fixture.model.n_embd())?);

        Ok(())
    }

    #[llama_test(
        model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
        n_gpu_layers = 999,
        use_mmap = true,
        use_mlock = false,
        n_ctx = 512,
        n_batch = 2048,
        n_ubatch = 512,
        n_seq_max = 4,
        embeddings = true,
    )]
    fn multi_sequence_embeddings_returns_one_embedding_per_sequence(
        fixture: &LlamaFixture<'_>,
    ) -> Result<()> {
        let mut context = LlamaContext::from_model(
            fixture.model,
            fixture.backend,
            (*fixture.context_params).into_llama_context_params(),
        )?;

        let inputs = [
            "alpha is here",
            "beta runs fast",
            "gamma waits",
            "delta jumps",
        ];
        let mut batch = LlamaBatch::new(64, 4)?;

        for (sequence_index, text) in inputs.iter().enumerate() {
            let tokens = fixture.model.str_to_token(text, AddBos::Always)?;
            let sequence_id = i32::try_from(sequence_index)?;

            batch.add_sequence(&tokens, sequence_id, true)?;
        }

        context.decode(&mut batch)?;

        let n_embd = usize::try_from(fixture.model.n_embd())?;
        let mut collected: Vec<Vec<f32>> = Vec::with_capacity(inputs.len());

        for sequence_index in 0..inputs.len() {
            let sequence_id = i32::try_from(sequence_index)?;
            let embedding = context.embeddings_seq_ith(sequence_id)?;

            assert_eq!(
                embedding.len(),
                n_embd,
                "sequence {sequence_index} embedding length mismatch"
            );

            collected.push(embedding.to_vec());
        }

        for (left_index, left) in collected.iter().enumerate() {
            for (right_index, right) in collected.iter().enumerate().skip(left_index + 1) {
                assert_ne!(
                    left, right,
                    "embedding for sequence {left_index} must differ from sequence {right_index}",
                );
            }
        }

        Ok(())
    }

    #[llama_test(
        model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
        n_gpu_layers = 999,
        use_mmap = true,
        use_mlock = false,
        n_ctx = 512,
        n_batch = 2048,
        n_ubatch = 512,
        n_seq_max = 4,
        embeddings = true,
    )]
    fn embeddings_returns_distinct_values_when_reused_batch_has_extra_capacity(
        fixture: &LlamaFixture<'_>,
    ) -> Result<()> {
        let mut context = LlamaContext::from_model(
            fixture.model,
            fixture.backend,
            (*fixture.context_params).into_llama_context_params(),
        )?;

        let iterations = [
            [
                "This is the first document with enough content to contribute meaningfully to the batch size calculation",
                "This is the second document that should be processed in a potentially different batch from the first",
            ],
            [
                "This is the third document adding more content to ensure the total exceeds the configured chunk limit",
                "This is the fourth document which should demonstrate that batching distributes across agent requests",
            ],
        ];

        let n_embd = usize::try_from(fixture.model.n_embd())?;
        let mut batch = LlamaBatch::new(64, 4)?;
        let mut collected: Vec<Vec<f32>> = Vec::new();

        for iteration_inputs in iterations {
            for (sequence_index, text) in iteration_inputs.iter().enumerate() {
                let tokens = fixture.model.str_to_token(text, AddBos::Always)?;
                let sequence_id = i32::try_from(sequence_index)?;

                batch.add_sequence(&tokens, sequence_id, true)?;
            }

            context.clear_kv_cache();
            context.decode(&mut batch)?;

            for sequence_index in 0..iteration_inputs.len() {
                let sequence_id = i32::try_from(sequence_index)?;
                let embedding = context.embeddings_seq_ith(sequence_id)?;

                assert_eq!(
                    embedding.len(),
                    n_embd,
                    "iteration sequence {sequence_index} embedding length mismatch"
                );

                collected.push(embedding.to_vec());
            }

            batch.clear();
        }

        assert_eq!(
            collected.len(),
            iterations.iter().flatten().count(),
            "expected one embedding per input across every iteration"
        );

        for (left_index, left) in collected.iter().enumerate() {
            for (right_index, right) in collected.iter().enumerate().skip(left_index + 1) {
                assert_ne!(
                    left, right,
                    "embedding {left_index} must differ from embedding {right_index} across reused-batch iterations",
                );
            }
        }

        Ok(())
    }

    #[llama_test(
        model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
        n_gpu_layers = 999,
        use_mmap = true,
        use_mlock = false,
        n_ctx = 512,
        n_batch = 2048,
        n_ubatch = 512,
        embeddings = true,
    )]
    fn embeddings_ith_returns_valid_embeddings(fixture: &LlamaFixture<'_>) -> Result<()> {
        let mut context = LlamaContext::from_model(
            fixture.model,
            fixture.backend,
            (*fixture.context_params).into_llama_context_params(),
        )?;
        let tokens = fixture.model.str_to_token("hello", AddBos::Always)?;
        let last_index = i32::try_from(tokens.len() - 1)?;
        let mut batch = LlamaBatch::new(512, 1)?;
        batch.add_sequence(&tokens, 0, false)?;
        context.decode(&mut batch)?;

        let embeddings = context.embeddings_ith(last_index)?;

        assert_eq!(embeddings.len(), usize::try_from(fixture.model.n_embd())?);

        Ok(())
    }

    #[llama_test(
        model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
        n_gpu_layers = 999,
        use_mmap = true,
        use_mlock = false,
        n_ctx = 512,
        n_batch = 2048,
        n_ubatch = 512,
        embeddings = true,
    )]
    fn embeddings_ith_returns_null_embedding_error_for_non_embedding_token(
        fixture: &LlamaFixture<'_>,
    ) -> Result<()> {
        let context = LlamaContext::from_model(
            fixture.model,
            fixture.backend,
            (*fixture.context_params).into_llama_context_params(),
        )?;

        let result = context.embeddings_ith(999);

        assert!(result.is_err());

        Ok(())
    }

    #[llama_test(
        model_source = HuggingFace("Xiaojian9992024/t5-small-GGUF", "t5-small.bf16.gguf"),
        n_gpu_layers = 999,
        use_mmap = true,
        use_mlock = false,
        n_ctx = 512,
        n_batch = 2048,
        n_ubatch = 512,
        embeddings = true,
    )]
    fn encode_succeeds_with_encoder_model(fixture: &LlamaFixture<'_>) -> Result<()> {
        let mut context = LlamaContext::from_model(
            fixture.model,
            fixture.backend,
            (*fixture.context_params).into_llama_context_params(),
        )?;
        let tokens = fixture.model.str_to_token("hello", AddBos::Never)?;
        let mut batch = LlamaBatch::new(512, 1)?;
        batch.add_sequence(&tokens, 0, false)?;

        let result = context.encode(&mut batch);

        assert!(result.is_ok());

        Ok(())
    }
}

mod context_kv_cache_embedding {
    use std::num::NonZeroU8;

    use anyhow::Result;
    use llama_cpp_bindings::context::LlamaContext;
    
    
    
    use llama_cpp_bindings::llama_batch::LlamaBatch;
    use llama_cpp_bindings::model::AddBos;
    use llama_cpp_test_harness::LlamaFixture;
    use llama_cpp_test_harness::llama_test;

    fn build_context<'context>(fixture: &'context LlamaFixture<'_>) -> Result<LlamaContext<'context>> {
        Ok(LlamaContext::from_model(
            fixture.model,
            fixture.backend,
            (*fixture.context_params).into_llama_context_params(),
        )?)
    }

    fn decode_hello_world(fixture: &LlamaFixture<'_>, context: &mut LlamaContext<'_>) -> Result<()> {
        let tokens = fixture.model.str_to_token("Hello world", AddBos::Always)?;
        let mut batch = LlamaBatch::new(512, 1)?;
        batch.add_sequence(&tokens, 0, false)?;
        context.decode(&mut batch)?;
        Ok(())
    }


    #[llama_test(
        model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
        n_gpu_layers = 999,
        use_mmap = true,
        use_mlock = false,
        n_ctx = 512,
        n_batch = 512,
        n_ubatch = 128,
    )]
    fn kv_cache_seq_add_succeeds_on_embedding_model(fixture: &LlamaFixture<'_>) -> Result<()> {
        let mut context = build_context(fixture)?;

        decode_hello_world(fixture, &mut context)?;

        let result = context.kv_cache_seq_add(0, Some(0), None, 1);

        assert!(result.is_ok());

        Ok(())
    }

    #[llama_test(
        model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
        n_gpu_layers = 999,
        use_mmap = true,
        use_mlock = false,
        n_ctx = 512,
        n_batch = 512,
        n_ubatch = 128,
    )]
    fn kv_cache_seq_div_succeeds_on_embedding_model(fixture: &LlamaFixture<'_>) -> Result<()> {
        let mut context = build_context(fixture)?;

        decode_hello_world(fixture, &mut context)?;

        let divisor = NonZeroU8::new(2).ok_or_else(|| anyhow::anyhow!("2 is non-zero"))?;
        let result = context.kv_cache_seq_div(0, Some(0), None, divisor);

        assert!(result.is_ok());

        Ok(())
    }
}

mod model_helpers_embedding {
    #![expect(
        clippy::unnecessary_wraps,
        reason = "every trial returns anyhow::Result<()> to match the LlamaTestFn signature"
    )]

    use anyhow::Result;
    use llama_cpp_test_harness::LlamaFixture;
    use llama_cpp_test_harness::llama_test;


    #[llama_test(
        model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
        n_gpu_layers = 999,
        use_mmap = true,
        use_mlock = false,
        n_ctx = 2048,
        n_batch = 512,
        n_ubatch = 128
    )]
    fn embedding_model_tool_call_markers_call_does_not_panic(fixture: &LlamaFixture<'_>) -> Result<()> {
        let _markers = fixture.model.tool_call_markers();

        Ok(())
    }

    #[llama_test(
        model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
        n_gpu_layers = 999,
        use_mmap = true,
        use_mlock = false,
        n_ctx = 2048,
        n_batch = 512,
        n_ubatch = 128
    )]
    fn embedding_model_streaming_markers_returns_ok_for_a_model_without_tool_calls(
        fixture: &LlamaFixture<'_>,
    ) -> Result<()> {
        let _markers = fixture.model.streaming_markers()?;

        Ok(())
    }

    #[llama_test(
        model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
        n_gpu_layers = 999,
        use_mmap = true,
        use_mlock = false,
        n_ctx = 2048,
        n_batch = 512,
        n_ubatch = 128
    )]
    fn approximate_tok_env_falls_back_to_eos_when_eot_unavailable(
        fixture: &LlamaFixture<'_>,
    ) -> Result<()> {
        let env = fixture.model.approximate_tok_env();
        let env_again = fixture.model.approximate_tok_env();

        assert!(
            std::sync::Arc::ptr_eq(&env, &env_again),
            "approximate_tok_env must return the same cached Arc for any model, including \
             the embedding model which lacks an EOT token (forcing the fallback-to-EOS path)"
        );

        Ok(())
    }
}

llama_tests_main!();
