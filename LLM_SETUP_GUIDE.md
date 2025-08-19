# LLM Setup Guide for Store Device Mapping

This guide shows you how to set up different LLM providers for text extraction from store device images. Since OpenAI may not be available in Hong Kong, here are several alternatives that work well.

## 🌟 Recommended Options for Hong Kong

### 1. **Google Gemini** (Recommended - Free & Available in Hong Kong)
- ✅ **Free tier available**
- ✅ **Works in Hong Kong** 
- ✅ **Excellent vision capabilities**

**Setup:**
```bash
# Get API key from https://aistudio.google.com/app/apikey
export GOOGLE_API_KEY='your-google-api-key-here'

# Run with Google Gemini
uv run python main.py --start --use-llm --llm-provider google
```

### 2. **Anthropic Claude** (Excellent Quality)
- ✅ **High quality results**
- ✅ **Available in Hong Kong**
- 💰 **Paid service** (but reasonable pricing)

**Setup:**
```bash
# Get API key from https://console.anthropic.com/
export ANTHROPIC_API_KEY='your-anthropic-api-key-here'

# Run with Claude
uv run python main.py --start --use-llm --llm-provider anthropic
```

### 3. **Local Ollama** (Free & Private - Recommended for Privacy)
- ✅ **Completely free**
- ✅ **Runs locally** (no internet required after setup)
- ✅ **Privacy-focused** (data never leaves your computer)

**Setup:**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download LLaVA vision model (may take a while)
ollama pull llava

# Run Ollama in background
ollama serve

# Run with local Ollama
uv run python main.py --start --use-llm --llm-provider ollama
```

## 📋 All Available Commands

### Traditional OCR (Current Method)
```bash
uv run python main.py --start
```

### LLM with Auto-Detection
```bash
uv run python main.py --start --use-llm
# Automatically uses the first available provider
```

### Specific Provider Selection
```bash
# Google Gemini
uv run python main.py --start --use-llm --llm-provider google

# Anthropic Claude
uv run python main.py --start --use-llm --llm-provider anthropic

# Local Ollama
uv run python main.py --start --use-llm --llm-provider ollama

# OpenAI (if available)
uv run python main.py --start --use-llm --llm-provider openai
```

## 🧪 Testing Your Setup

Test any provider before running the full extraction:

```bash
# Test your LLM setup on sample images
uv run python test_llm_extraction.py
```

## 💡 Recommendations

1. **For Best Quality**: Use **Anthropic Claude** or **Google Gemini**
2. **For Privacy**: Use **Local Ollama** 
3. **For Budget**: Use **Google Gemini** (free tier)
4. **For Speed**: Use **Google Gemini** (fastest API)

## 🔧 Troubleshooting

**Provider not detected?**
- Check your API key environment variable is set
- For Ollama: Make sure `ollama serve` is running

**Getting errors?**
- Run `uv run python test_llm_extraction.py` to test your setup
- Check the logs for specific error messages

**Want to switch providers?**
- Just use a different `--llm-provider` flag
- You can mix and match for different runs

## 📊 Expected Performance

- **Accuracy**: LLM methods typically achieve 90-95% accuracy vs 60-70% for OCR
- **Speed**: 
  - Google Gemini: ~2-3 seconds per image
  - Anthropic Claude: ~3-4 seconds per image  
  - Local Ollama: ~5-10 seconds per image (depends on your hardware)
  - OCR: ~1-2 seconds per image

The improved accuracy usually makes the slightly slower speed worthwhile!