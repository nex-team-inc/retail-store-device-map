#!/usr/bin/env python3
"""
Debug script to examine specific image preprocessing and save debug images.
"""

from extract_store_mappings import StoreImageProcessor
from pathlib import Path
import cv2

def debug_image_processing():
    """Debug image processing for the 307 store image."""
    processor = StoreImageProcessor()
    
    # The image that should contain "081980G"
    image_path = Path("datasource") / "Elem 294928 Tmp 68716 Q 2. Ch 8 St 307 image_3183377842_202506300516_36.jpg"
    
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return
        
    print(f"Debugging: {image_path.name}")
    
    # Get all preprocessed versions
    processed_images = processor.preprocess_image(image_path)
    
    # Save each version to see what they look like
    debug_dir = Path("debug_images")
    debug_dir.mkdir(exist_ok=True)
    
    version_names = [
        "01_original_gray",
        "02_enhanced_contrast", 
        "03_gaussian_threshold",
        "04_adaptive_threshold",
        "05_bright_areas",
        "06_dark_text",
        "07_morphological"
    ]
    
    for i, img in enumerate(processed_images):
        if img is not None:
            filename = f"{version_names[i]}_store307.png"
            cv2.imwrite(str(debug_dir / filename), img)
            print(f"Saved: {filename}")
            
    # Try OCR on each version separately to see which works best
    import pytesseract
    
    configs = [
        '--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        '--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        '--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        '--psm 6',
        '--psm 8',
    ]
    
    print("\nOCR Results for each preprocessing version:")
    for i, img in enumerate(processed_images):
        if img is not None:
            print(f"\n{version_names[i]}:")
            for config in configs:
                try:
                    text = pytesseract.image_to_string(img, config=config).strip()
                    if text:
                        print(f"  {config}: '{text}'")
                except:
                    continue

if __name__ == "__main__":
    debug_image_processing()