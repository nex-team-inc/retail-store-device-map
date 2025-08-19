#!/usr/bin/env python3
"""
Test script for LLM-based text extraction.
"""

import os
from pathlib import Path
from extract_store_mappings import StoreImageProcessor

def test_llm_extraction():
    """Test LLM extraction on a few sample images."""
    
    # Check if OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable is not set!")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Initialize processor with LLM enabled
    processor = StoreImageProcessor(use_llm=True)
    
    # Find some sample images
    datasource_dir = Path("datasource")
    image_files = list(datasource_dir.glob("*.jpg"))[:3]  # Test on first 3 images
    
    if not image_files:
        print("❌ No images found in datasource/ directory")
        return
    
    print("🤖 Testing LLM-based text extraction...")
    print("=" * 60)
    
    for image_path in image_files:
        print(f"\n📸 Processing: {image_path.name}")
        
        # Extract store number
        store_number = processor.extract_store_number(image_path.name)
        print(f"Store Number: {store_number}")
        
        # Extract text using LLM
        extracted_text, confidence = processor.extract_text_with_llm(image_path)
        print(f"Extracted Text: '{extracted_text}'")
        print(f"Confidence: {confidence}%")
        
        # Find device ID
        device_id, device_score = processor.find_device_id(extracted_text)
        print(f"Device ID: {device_id or 'None found'}")
        print(f"Device Score: {device_score}")
        print("-" * 40)

if __name__ == "__main__":
    test_llm_extraction()