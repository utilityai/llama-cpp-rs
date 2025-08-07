#!/usr/bin/env python3
"""Quick test of one model to verify setup"""

import subprocess
import json
import time
from sentence_transformers import SentenceTransformer
import numpy as np

# Test text
test_text = "Artificial intelligence is transforming the world."

# Test Python
print("Testing Python sentence-transformers...")
start = time.time()
model = SentenceTransformer('thenlper/gte-base')
py_embedding = model.encode([test_text], normalize_embeddings=True)[0]
py_time = (time.time() - start) * 1000
print(f"✅ Python: {len(py_embedding)} dims in {py_time:.0f}ms")

# Test Rust
print("\nTesting Rust llama-cpp-rs...")
gguf_path = "/Users/sarav/Downloads/Embedding/gte-base.Q8_0.gguf"
binary_path = "/Users/sarav/Downloads/side/rzn/llama-cpp-rs/target/release/embeddings"

# Check if binary exists
import os
if not os.path.exists(binary_path):
    print(f"❌ Binary not found at {binary_path}")
    print("   Build with: cargo build --release --bin embeddings")
    exit(1)

# Run Rust
cmd = [binary_path, "--json", "-n", "local", gguf_path]
start = time.time()
result = subprocess.run(cmd, input=test_text, capture_output=True, text=True)
rust_time = (time.time() - start) * 1000

if result.returncode == 0:
    rust_embedding = json.loads(result.stdout)[0]
    print(f"✅ Rust: {len(rust_embedding)} dims in {rust_time:.0f}ms")
    
    # Debug
    print(f"\nDebug info:")
    print(f"  Python norm: {np.linalg.norm(py_embedding):.4f}")
    print(f"  Rust norm: {np.linalg.norm(rust_embedding):.4f}")
    print(f"  Python first 5: {py_embedding[:5]}")
    print(f"  Rust first 5: {rust_embedding[:5]}")
    
    # Compare
    py_norm = np.linalg.norm(py_embedding)
    rust_norm = np.linalg.norm(rust_embedding)
    if py_norm > 0 and rust_norm > 0:
        similarity = np.dot(py_embedding, rust_embedding) / (py_norm * rust_norm)
        print(f"\n📊 Cosine similarity: {similarity:.4f}")
    else:
        print(f"\n⚠️ One or both embeddings have zero norm")
    
    print(f"🚀 Speedup: {py_time/rust_time:.1f}x")
else:
    print(f"❌ Rust error: {result.stderr}")