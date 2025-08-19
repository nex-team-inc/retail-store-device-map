#!/usr/bin/env python3
"""
Main control script for store device mapping extraction.

This script provides options to:
1. Start/resume processing
2. Check progress
3. Export current results
4. Clean up and restart
"""

import argparse
import sys
from pathlib import Path
from extract_store_mappings import StoreImageProcessor
from check_progress import check_progress
import json
import pandas as pd

def start_processing(use_llm=False, llm_provider='auto'):
    """Start or resume the image processing."""
    print("Starting/resuming image processing...")
    print("Press Ctrl+C at any time to pause and save progress.")
    print("="*60)
    
    try:
        processor = StoreImageProcessor(use_llm=use_llm, llm_provider=llm_provider)
        if use_llm:
            print(f"🤖 Using LLM ({llm_provider}) for text extraction")
        else:
            print("🔍 Using traditional OCR (Tesseract) for text extraction")
        print("-" * 60)
        
        results = processor.process_all_images()
        
        if results:
            output_file = processor.save_results_to_csv(results)
            print(f"\n{'='*60}")
            
            # Check if processing was interrupted or completed naturally
            if processor.should_stop:
                print("PROCESSING PAUSED!")
                print(f"Partial results saved to: {output_file}")
                print(f"Images processed so far: {len(results)}")
                print("Checkpoint preserved for resuming.")
            else:
                print("PROCESSING COMPLETED!")
                print(f"Results saved to: {output_file}")
                print(f"Total images processed: {len(results)}")
                
                # Only clean up checkpoint file if processing completed naturally
                if processor.checkpoint_file.exists():
                    processor.checkpoint_file.unlink()
                    print("Checkpoint file cleaned up.")
        
    except KeyboardInterrupt:
        print("\nProcessing paused by user.")
        print("Run 'python main.py --check' to see progress.")
        print("Run 'python main.py --start' to resume.")
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1
        
    return 0

def export_current_results():
    """Export current results to CSV even if processing isn't complete."""
    checkpoint_file = Path("processing_checkpoint.json")
    
    if not checkpoint_file.exists():
        print("No checkpoint file found. No results to export.")
        return 1
        
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            
        results = checkpoint_data.get('results', [])
        
        if not results:
            print("No results found in checkpoint.")
            return 1
            
        # Create partial results file
        df = pd.DataFrame(results)
        df = df.sort_values('store_number', na_position='last')
        
        output_file = "store_device_mapping_partial.csv"
        df.to_csv(output_file, index=False)
        
        print(f"Partial results exported to: {output_file}")
        print(f"Images processed so far: {len(results)}")
        
        # Show summary
        successful = len([r for r in results if r['device_id'] is not None])
        print(f"Successful extractions: {successful} ({(successful/len(results))*100:.1f}%)")
        
    except Exception as e:
        print(f"Error exporting results: {e}")
        return 1
        
    return 0

def reset_processing():
    """Clean up checkpoint and start fresh."""
    checkpoint_file = Path("processing_checkpoint.json")
    
    if checkpoint_file.exists():
        response = input("This will delete all progress and start fresh. Are you sure? (y/N): ")
        if response.lower() == 'y':
            checkpoint_file.unlink()
            print("Checkpoint file deleted. Processing will start fresh on next run.")
        else:
            print("Reset cancelled.")
    else:
        print("No checkpoint file found. Nothing to reset.")

def main():
    parser = argparse.ArgumentParser(description="Store Device Mapping Extraction Tool")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--start', action='store_true', 
                      help='Start or resume image processing')
    group.add_argument('--check', action='store_true',
                      help='Check current processing progress')
    group.add_argument('--export', action='store_true',
                      help='Export current results to CSV (even if incomplete)')
    group.add_argument('--reset', action='store_true',
                      help='Reset progress and start fresh')
    
    parser.add_argument('--use-llm', action='store_true',
                       help='Use LLM instead of OCR for text extraction')
    parser.add_argument('--llm-provider', choices=['auto', 'openai', 'anthropic', 'google', 'ollama'],
                       default='auto', help='Choose LLM provider (default: auto)')
    
    args = parser.parse_args()
    
    if args.start:
        return start_processing(use_llm=args.use_llm, llm_provider=args.llm_provider)
    elif args.check:
        check_progress()
        return 0
    elif args.export:
        return export_current_results()
    elif args.reset:
        reset_processing()
        return 0

if __name__ == "__main__":
    sys.exit(main())