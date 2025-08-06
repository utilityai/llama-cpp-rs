#!/usr/bin/env python3
"""Test single model to verify report generation"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import benchmark functions
from comprehensive_benchmark import *

def test_single_model():
    """Test just GTE-Base to verify report generation"""
    print("=" * 80)
    print("🔬 Testing Single Model - GTE-Base")
    print("=" * 80)
    
    # Use just 5 texts for quick test
    test_texts = [
        "Artificial intelligence is transforming the world.",
        "Machine learning algorithms learn from data.",
        "Deep learning uses neural networks.",
        "Natural language processing understands text.",
        "Computer vision analyzes images."
    ]
    
    results = {}
    model_name = "GTE-Base"
    
    if model_name in MODELS:
        print(f"\n📊 Testing {model_name}")
        print("-" * 40)
        
        # Test Python
        py_emb, py_time = test_python_batch(model_name, test_texts)
        
        # Test Rust
        rust_emb, rust_time = test_rust_batch(model_name, test_texts)
        
        # Calculate similarity
        rust_sim = None
        if py_emb and rust_emb:
            similarities = []
            for i in range(min(len(py_emb), len(rust_emb))):
                if py_emb[i] is not None and rust_emb[i] is not None:
                    sim = calculate_similarity(py_emb[i], rust_emb[i])
                    if sim is not None:
                        similarities.append(sim)
            rust_sim = np.mean(similarities) if similarities else None
        
        results[model_name] = {
            'python_time': py_time,
            'rust_time': rust_time,
            'rust_similarity': rust_sim,
            'python_embeddings': py_emb,
            'rust_embeddings': rust_emb
        }
        
        # Print results
        print(f"\n📈 Results:")
        print(f"  Python time: {py_time:.1f}ms per text" if py_time else "  Python: N/A")
        print(f"  Rust time: {rust_time:.1f}ms per text" if rust_time else "  Rust: N/A")
        print(f"  Similarity: {rust_sim:.4f}" if rust_sim else "  Similarity: N/A")
        
        if py_time and rust_time:
            print(f"  Speedup: {py_time/rust_time:.1f}x")
    
    # Generate report
    print("\n" + "=" * 80)
    print("📄 Generating Report...")
    print("=" * 80)
    
    # Simple test articles for report
    test_articles = [{'id': i+1, 'title': f'Article {i+1}', 'text': text, 'tokens': 10} 
                     for i, text in enumerate(test_texts)]
    
    generate_markdown_report(results, test_articles)
    
    # Check if report was created
    import glob
    reports = glob.glob("benchmark_report_*.md")
    if reports:
        latest_report = sorted(reports)[-1]
        print(f"\n✅ Report generated: {latest_report}")
        
        # Show first 50 lines of report
        print("\n📖 Report Preview:")
        print("-" * 40)
        with open(latest_report, 'r') as f:
            lines = f.readlines()[:50]
            for line in lines:
                print(line.rstrip())
    else:
        print("\n❌ No report generated")

if __name__ == "__main__":
    test_single_model()