#!/usr/bin/env python3
"""
Web-based visualizer for store device mapping results.
Allows easy review of OCR results with original images.
"""

import os
import re
import base64
from pathlib import Path
from io import BytesIO
from flask import Flask, render_template, request, jsonify
import pandas as pd
from PIL import Image
import json

app = Flask(__name__)

class ResultsVisualizer:
    def __init__(self, csv_file="store_device_mapping.csv", images_dir="datasource"):
        self.csv_file = Path(csv_file)
        self.images_dir = Path(images_dir)
        self.df = None
        self.available_files = self.get_available_csv_files()
        self.load_data()
        
    def load_data(self):
        """Load the CSV results."""
        if self.csv_file.exists():
            self.df = pd.read_csv(self.csv_file)
            # Sort by store number for easier navigation
            self.df = self.df.sort_values('store_number', na_position='last')
        else:
            raise FileNotFoundError(f"Results file not found: {self.csv_file}")
    
    def get_available_csv_files(self):
        """Get list of available CSV files for results."""
        files = []
        potential_files = [
            "store_device_mapping.csv",
            "store_device_mapping_partial.csv", 
            "store_device_mapping_corrected.csv",
            "store_device_mapping_improved.csv"
        ]
        
        for file in potential_files:
            file_path = Path(file)
            if file_path.exists():
                # Get basic stats about the file
                try:
                    df = pd.read_csv(file_path)
                    total_records = len(df)
                    successful = len(df[df['device_id'].notna()])
                    success_rate = (successful / total_records * 100) if total_records > 0 else 0
                    
                    files.append({
                        'filename': file,
                        'display_name': file.replace('store_device_mapping', 'Results').replace('.csv', ''),
                        'total_records': total_records,
                        'successful': successful,
                        'success_rate': f"{success_rate:.1f}%"
                    })
                except Exception:
                    files.append({
                        'filename': file,
                        'display_name': file.replace('store_device_mapping', 'Results').replace('.csv', ''),
                        'total_records': 0,
                        'successful': 0,
                        'success_rate': "0%"
                    })
        
        return files
    
    def switch_file(self, new_file):
        """Switch to a different CSV file."""
        new_path = Path(new_file)
        if new_path.exists():
            self.csv_file = new_path
            self.load_data()
            return True
        return False
            
    def get_image_base64(self, filename):
        """Convert image to base64 for web display."""
        image_path = self.images_dir / filename
        if not image_path.exists():
            return None
            
        try:
            with Image.open(image_path) as img:
                # Resize image for web display (max width 800px)
                img.thumbnail((800, 600), Image.Resampling.LANCZOS)
                
                # Convert to base64
                buffer = BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                img_str = base64.b64encode(buffer.getvalue()).decode()
                return f"data:image/jpeg;base64,{img_str}"
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
            return None
            
    def get_summary_stats(self):
        """Get summary statistics."""
        total = len(self.df)
        successful = len(self.df[self.df['device_id'].notna()])
        flagged = len(self.df[self.df['needs_review'] == True])
        
        return {
            'total_images': total,
            'successful_extractions': successful,
            'success_rate': f"{(successful/total)*100:.1f}%" if total > 0 else "0%",
            'flagged_for_review': flagged,
            'flagged_rate': f"{(flagged/total)*100:.1f}%" if total > 0 else "0%"
        }
        
    def filter_results(self, filter_type="all", search_term="", page=1, per_page=20):
        """Filter and paginate results."""
        df_filtered = self.df.copy()
        
        # Apply filters
        if filter_type == "successful":
            df_filtered = df_filtered[df_filtered['device_id'].notna()]
        elif filter_type == "flagged":
            df_filtered = df_filtered[df_filtered['needs_review'] == True]
        elif filter_type == "failed":
            df_filtered = df_filtered[df_filtered['device_id'].isna()]
            
        # Apply search
        if search_term:
            search_term = search_term.lower()
            mask = (
                df_filtered['filename'].str.lower().str.contains(search_term, na=False) |
                df_filtered['store_number'].astype(str).str.contains(search_term, na=False) |
                df_filtered['device_id'].astype(str).str.lower().str.contains(search_term, na=False) |
                df_filtered['extracted_text'].astype(str).str.lower().str.contains(search_term, na=False)
            )
            df_filtered = df_filtered[mask]
            
        # Pagination
        total_items = len(df_filtered)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        df_page = df_filtered.iloc[start_idx:end_idx]
        
        # Convert to list of dicts with images
        results = []
        for _, row in df_page.iterrows():
            result = row.to_dict()
            result['image_data'] = self.get_image_base64(row['filename'])
            # Convert NaN values to None for JSON serialization
            for key, value in result.items():
                if pd.isna(value):
                    result[key] = None
            results.append(result)
            
        return {
            'results': results,
            'total_items': total_items,
            'current_page': page,
            'total_pages': (total_items + per_page - 1) // per_page,
            'per_page': per_page
        }

# Initialize visualizer
visualizer = ResultsVisualizer()

@app.route('/')
def index():
    """Main page."""
    stats = visualizer.get_summary_stats()
    return render_template('index.html', stats=stats)

@app.route('/api/results')
def api_results():
    """API endpoint to get filtered results."""
    filter_type = request.args.get('filter', 'all')
    search_term = request.args.get('search', '')
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    
    results = visualizer.filter_results(filter_type, search_term, page, per_page)
    return jsonify(results)

@app.route('/api/stats')
def api_stats():
    """API endpoint to get summary statistics."""
    return jsonify(visualizer.get_summary_stats())

@app.route('/api/files')
def api_files():
    """API endpoint to get available CSV files."""
    return jsonify({
        'files': visualizer.available_files,
        'current_file': str(visualizer.csv_file.name)
    })

@app.route('/api/switch-file', methods=['POST'])
def api_switch_file():
    """API endpoint to switch to a different CSV file."""
    try:
        data = request.json
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'Filename is required'}), 400
            
        success = visualizer.switch_file(filename)
        if success:
            # Refresh available files list
            visualizer.available_files = visualizer.get_available_csv_files()
            return jsonify({'success': True, 'message': f'Switched to {filename}'})
        else:
            return jsonify({'error': 'File not found or could not be loaded'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/update', methods=['POST'])
def api_update():
    """API endpoint to update a record."""
    try:
        data = request.json
        filename = data.get('filename')
        new_device_id = data.get('device_id', '').strip()
        
        if not filename:
            return jsonify({'error': 'Filename is required'}), 400
            
        # Validate device ID format if provided
        if new_device_id and not re.match(r'^\d{6}[GN]$', new_device_id):
            return jsonify({'error': 'Device ID must be 6 digits followed by G or N'}), 400
            
        # Update the record in the DataFrame
        mask = visualizer.df['filename'] == filename
        if not mask.any():
            return jsonify({'error': 'Record not found'}), 404
            
        # Update the record
        if new_device_id:
            visualizer.df.loc[mask, 'device_id'] = new_device_id
            visualizer.df.loc[mask, 'device_score'] = 1.0  # Manual correction gets highest score
            visualizer.df.loc[mask, 'quality_flag'] = 'MANUALLY_CORRECTED'
            visualizer.df.loc[mask, 'needs_review'] = False
        else:
            visualizer.df.loc[mask, 'device_id'] = None
            visualizer.df.loc[mask, 'device_score'] = 0.0
            visualizer.df.loc[mask, 'quality_flag'] = 'MANUALLY_MARKED_FAILED'
            visualizer.df.loc[mask, 'needs_review'] = True
            
        # Save the updated DataFrame back to CSV
        visualizer.df.to_csv(visualizer.csv_file, index=False)
        
        # Also create a backup of corrected results
        corrected_file = str(visualizer.csv_file).replace('.csv', '_corrected.csv')
        visualizer.df.to_csv(corrected_file, index=False)
        
        return jsonify({'success': True, 'message': 'Record updated successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    print("Starting Store Device Mapping Visualizer...")
    print("Open your browser to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    app.run(debug=True, host='0.0.0.0', port=5000)