# Store Device Mapping Visualizer

A web-based tool to review OCR results with original images for easy verification and analysis.

## Features

### 📊 **Dashboard Overview**
- **Total Images**: Shows count of all processed images
- **Successful Extractions**: Count of images with valid device IDs
- **Success Rate**: Percentage of successful extractions
- **Flagged for Review**: Count of images that need manual review

### 🔍 **Smart Filtering**
- **All Results**: View all processed images
- **Successful Only**: Show only images with extracted device IDs
- **Flagged for Review**: Show images that need manual verification
- **Failed Extractions**: Show images where no device ID was found

### 🔎 **Search Functionality**
Search across:
- Store numbers
- Device IDs
- Filenames
- Extracted text content

### 📱 **Image Display**
- **Original Images**: View actual store device photos
- **Device IDs**: Clearly highlighted successful extractions
- **OCR Text**: Raw text extracted from each image
- **Confidence Scores**: OCR confidence ratings (High/Medium/Low)
- **Quality Flags**: Indicators for images needing review

### 📄 **Navigation**
- **Pagination**: Navigate through large result sets
- **Adjustable Page Size**: View 10, 20, 50, or 100 results per page
- **Responsive Design**: Works on desktop and mobile devices

## How to Use

### 1. **Start the Visualizer**
```bash
uv run python visualizer.py
```

### 2. **Open in Browser**
Navigate to: `http://localhost:5000`

### 3. **Review Results**
- Use filters to focus on specific result types
- Search for specific stores or device IDs
- Click through pages to review all results
- Verify successful extractions by comparing device IDs with images
- Identify patterns in failed extractions

### 4. **Quality Control**
- **Green Device IDs**: Successfully extracted 6-digit + G/N codes
- **Red "No device ID found"**: Images that need manual review
- **High Confidence (Green)**: OCR confidence ≥ 80%
- **Medium Confidence (Orange)**: OCR confidence 50-79%
- **Low Confidence (Red)**: OCR confidence < 50%

## Tips for Review

### ✅ **Successful Extractions**
- Verify the device ID matches what's visible in the image
- Look for any OCR errors (e.g., 0 vs O, 6 vs G)
- Check that the format is exactly 6 digits + G or N

### ⚠️ **Flagged Images**
- Images may be blurry or dark
- Display screens might be off or showing different content
- Text might be partially obscured
- OCR may have low confidence in the extraction

### 🔄 **Common Issues**
- **Reflections**: Screen glare can interfere with OCR
- **Viewing Angle**: Off-angle photos may reduce text clarity
- **Display State**: Some screens might show menus instead of device IDs
- **Image Quality**: Low resolution or motion blur affects OCR accuracy

## Controls

| Control | Description |
|---------|-------------|
| **Filter Dropdown** | Select result type to view |
| **Per Page Dropdown** | Choose number of results per page |
| **Search Box** | Enter search terms (real-time search with 500ms delay) |
| **Pagination Buttons** | Navigate between result pages |

## Keyboard Shortcuts

- **Enter** in search box: Apply search immediately
- **Arrow Keys**: Navigate pagination when buttons are focused

The visualizer automatically saves your filter and search preferences during your session for a seamless review experience.