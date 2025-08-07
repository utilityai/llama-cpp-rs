#!/bin/bash
# Embedding Benchmark Runner

set -e

echo "=========================================="
echo "Embedding Benchmark Suite"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "comprehensive_benchmark.py" ]; then
    echo "❌ Run this script from the benchmarks directory"
    echo "   cd examples/embeddings/benchmarks"
    exit 1
fi

# Check if config.json exists
if [ ! -f "config.json" ]; then
    echo "❌ config.json not found!"
    echo "   Please create config.json with your model paths"
    echo "   See README.md for configuration instructions"
    exit 1
fi

# Check if Rust binary exists
if [ ! -f "../../../target/release/embeddings" ]; then
    echo "❌ Rust binary not found. Building..."
    cd ../../..
    echo "Building embeddings binary with hardware acceleration..."
    
    # Detect platform and build with appropriate features
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Detected macOS - building with Metal support"
        cargo build --release --bin embeddings --features metal
    elif command -v nvidia-smi >/dev/null 2>&1; then
        echo "Detected NVIDIA GPU - building with CUDA support"
        cargo build --release --bin embeddings --features cuda
    else
        echo "Building with CPU support"
        cargo build --release --bin embeddings
    fi
    
    cd examples/embeddings/benchmarks
    echo "✅ Build complete"
fi

# Check Python dependencies
echo "Checking Python dependencies..."
python -c "import sentence_transformers, numpy" 2>/dev/null || {
    echo "❌ Missing Python dependencies. Install with:"
    echo "   pip install sentence-transformers numpy"
    exit 1
}

python -c "import embed_anything" 2>/dev/null || {
    echo "⚠️  EmbedAnything not found. ONNX tests will be skipped."
    echo "   Install with: pip install embed-anything"
}

# Menu
echo "📋 Config loaded: $(jq -r '.models | keys | join(", ")' config.json 2>/dev/null || echo "Unable to read model list")"

echo ""
echo "🚀 Ready to benchmark! Select test:"
echo "1. Comprehensive benchmark (recommended) - All models, timing + accuracy + CSV"
echo "2. Exit"
echo ""
read -p "Enter choice (1-2): " choice

case $choice in
    1)
        echo ""
        echo "🔬 Running comprehensive benchmark..."
        echo "This tests all 5 models with 50 texts and exports CSV comparisons"
        echo "Estimated time: 5-15 minutes depending on hardware"
        echo ""
        python comprehensive_benchmark.py
        ;;
    2)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac

echo ""
echo "✅ Benchmark complete!"
echo "📄 Check the embeddings_comparison_*.csv files for detailed analysis"