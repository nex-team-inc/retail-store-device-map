#!/usr/bin/env python3
"""
Script to check the current progress of image processing.
"""

import json
import pandas as pd
from pathlib import Path

def check_progress():
    """Check and display current processing progress."""
    checkpoint_file = Path("processing_checkpoint.json")
    
    if not checkpoint_file.exists():
        print("No checkpoint file found. Processing hasn't started yet.")
        return
        
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            
        results = checkpoint_data.get('results', [])
        processed_files = checkpoint_data.get('processed_files', [])
        timestamp = checkpoint_data.get('timestamp', 'Unknown')
        
        total_processed = len(results)
        successful_extractions = len([r for r in results if r['device_id'] is not None])
        flagged_for_review = len([r for r in results if r['needs_review']])
        
        print("="*50)
        print("PROCESSING PROGRESS REPORT")
        print("="*50)
        print(f"Last updated: {timestamp}")
        print(f"Total images processed: {total_processed}")
        print(f"Successful device ID extractions: {successful_extractions}")
        print(f"Success rate: {(successful_extractions/total_processed)*100:.1f}%" if total_processed > 0 else "0%")
        print(f"Images flagged for review: {flagged_for_review}")
        print(f"Review rate: {(flagged_for_review/total_processed)*100:.1f}%" if total_processed > 0 else "0%")
        
        if results:
            # Create DataFrame for analysis
            df = pd.DataFrame(results)
            
            print("\n" + "="*50)
            print("SAMPLE SUCCESSFUL EXTRACTIONS")
            print("="*50)
            successful = df[df['device_id'].notna()]
            if not successful.empty:
                print(successful[['filename', 'store_number', 'device_id', 'ocr_confidence']].head(10).to_string(index=False))
            else:
                print("No successful extractions yet.")
                
            print("\n" + "="*50)
            print("SAMPLE FLAGGED IMAGES")
            print("="*50)
            flagged = df[df['needs_review'] == True]
            if not flagged.empty:
                print(flagged[['filename', 'store_number', 'quality_flag']].head(10).to_string(index=False))
            else:
                print("No images flagged for review.")
                
            print("\n" + "="*50)
            print("QUALITY FLAGS SUMMARY")
            print("="*50)
            flag_counts = df['quality_flag'].value_counts()
            for flag, count in flag_counts.items():
                percentage = (count / total_processed) * 100
                print(f"{flag}: {count} ({percentage:.1f}%)")
        
        # Check total images in datasource
        datasource_dir = Path("datasource")
        if datasource_dir.exists():
            total_images = len(list(datasource_dir.glob("*.jpg")) + list(datasource_dir.glob("*.JPG")))
            remaining = total_images - total_processed
            print(f"\n" + "="*50)
            print("OVERALL PROGRESS")
            print("="*50)
            print(f"Total images in datasource: {total_images}")
            print(f"Images processed: {total_processed}")
            print(f"Images remaining: {remaining}")
            print(f"Overall completion: {(total_processed/total_images)*100:.1f}%" if total_images > 0 else "0%")
            
    except Exception as e:
        print(f"Error reading checkpoint file: {e}")

def main():
    check_progress()

if __name__ == "__main__":
    main()