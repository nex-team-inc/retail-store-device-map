#!/usr/bin/env python3
"""
Test script to verify OCR functionality on a few sample images.
"""

from extract_store_mappings import StoreImageProcessor
from pathlib import Path
import pandas as pd

def test_sample_images():
    """Test the OCR on a few sample images."""
    processor = StoreImageProcessor()
    
    # Test with specific sample images mentioned by user
    sample_files = [
        "Elem 294928 Tmp 68716 Q 2. Ch 8 St 307 image_3183377842_202506300516_36.jpg",  # Should have 081980G
        "Elem 294928 Tmp 68716 Q 2. Ch 8 St 601 image_3187242491_202507021152_28.jpg",  # User said this one is not well captured
        "Elem 294928 Tmp 68716 Q 2. Ch 8 St 422 image_3184237901_202506301417_25.jpg",  # Another sample
    ]
    
    results = []
    
    for filename in sample_files:
        image_path = Path("datasource") / filename
        if image_path.exists():
            print(f"\nProcessing: {filename}")
            result = processor.process_single_image(image_path)
            results.append(result)
            
            print(f"Store Number: {result['store_number']}")
            print(f"Extracted Text: '{result['extracted_text']}'")
            print(f"Device ID: {result['device_id']}")
            print(f"OCR Confidence: {result['ocr_confidence']}")
            print(f"Quality Flag: {result['quality_flag']}")
            print(f"Needs Review: {result['needs_review']}")
        else:
            print(f"File not found: {filename}")
    
    if results:
        # Save sample results
        df = pd.DataFrame(results)
        df.to_csv("sample_results.csv", index=False)
        print(f"\nSample results saved to: sample_results.csv")
        
        print(f"\nSample Results Table:")
        print(df.to_string(index=False))

if __name__ == "__main__":
    test_sample_images()