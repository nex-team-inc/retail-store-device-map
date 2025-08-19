#!/usr/bin/env python3
"""
Script to extract store number to string mappings from retail store device images.

This script:
1. Processes all JPG images in the datasource/ directory
2. Extracts store numbers from filenames using regex
3. Uses OCR to extract text strings from images
4. Implements quality checking to flag uncertain extractions
5. Outputs results to a CSV file

Author: AI Assistant
"""

import os
import re
import csv
import json
import logging
import base64
from pathlib import Path
from typing import Tuple, Optional, List
import cv2
import numpy as np
import pytesseract
from PIL import Image
import pandas as pd
import signal
import sys
import requests
import time
from datetime import datetime, timedelta
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
try:
    import anthropic
except ImportError:
    anthropic = None
try:
    import google.generativeai as genai
except ImportError:
    genai = None


class StoreImageProcessor:
    """Process retail store device images to extract text mappings."""
    
    def __init__(self, datasource_dir: str = "datasource", checkpoint_file: str = "processing_checkpoint.json", 
                 use_llm: bool = True, llm_provider: str = "auto"):
        """Initialize the processor."""
        self.datasource_dir = Path(datasource_dir)
        self.checkpoint_file = Path(checkpoint_file)
        self.should_stop = False
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.setup_logging()
        self.setup_signal_handlers()
        
        # Initialize LLM clients if using LLM
        if self.use_llm:
            self.setup_llm_clients()
            self.last_request_time = {}  # Track last request time per provider
            self.retry_delays = {}  # Track retry delays from rate limit responses
        
        # Pattern to extract store number from filename
        # Format: "Elem ... St {store_number} image_..."
        self.store_pattern = re.compile(r'St\s+(\d+)\s+image_')
        
        # Pattern to identify device strings: exactly 6 digits followed by G or N
        self.device_id_patterns = [
            re.compile(r'\b\d{6}[GN]\b'),  # Exact pattern: 6 digits + G or N
            re.compile(r'\d{6}[GN]'),  # Same but without word boundaries (in case OCR misses spaces)
            re.compile(r'(\d\s*){6}[GN]'),  # With potential spaces between digits
            re.compile(r'[0O]{6}[GN]'),  # In case OCR confuses 0 with O
            re.compile(r'[\dO]{6}[GN]'),  # Mixed digits and O's
        ]
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('store_extraction.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info("\nReceived interrupt signal. Saving progress and stopping gracefully...")
            self.should_stop = True
            
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    def setup_llm_clients(self):
        """Setup LLM clients based on available API keys."""
        self.available_providers = []
        
        # Check OpenAI
        if OpenAI and os.getenv('OPENAI_API_KEY'):
            try:
                self.openai_client = OpenAI()
                self.available_providers.append('openai')
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI client: {e}")
        
        # Check Anthropic (Claude)
        if anthropic and os.getenv('ANTHROPIC_API_KEY'):
            try:
                self.anthropic_client = anthropic.Anthropic()
                self.available_providers.append('anthropic')
            except Exception as e:
                self.logger.warning(f"Failed to initialize Anthropic client: {e}")
        
        # Check Google Gemini
        if genai and os.getenv('GOOGLE_API_KEY'):
            try:
                genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                self.available_providers.append('google')
            except Exception as e:
                self.logger.warning(f"Failed to initialize Google Gemini client: {e}")
        
        # Check for local Ollama
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                vision_models = [m for m in models if 'llava' in m.get('name', '').lower()]
                if vision_models:
                    self.available_providers.append('ollama')
        except:
            pass
        
        if not self.available_providers:
            self.logger.error("No LLM providers available! Please set up at least one:")
            self.logger.error("- OpenAI: export OPENAI_API_KEY='your-key'")
            self.logger.error("- Anthropic: export ANTHROPIC_API_KEY='your-key'") 
            self.logger.error("- Google: export GOOGLE_API_KEY='your-key'")
            self.logger.error("- Ollama: Install and run 'ollama pull llava'")
            raise RuntimeError("No LLM providers configured")
        
        # Select provider
        if self.llm_provider == "auto":
            self.active_provider = self.available_providers[0]
        elif self.llm_provider in self.available_providers:
            self.active_provider = self.llm_provider
        else:
            self.logger.warning(f"Requested provider '{self.llm_provider}' not available. Using '{self.available_providers[0]}'")
            self.active_provider = self.available_providers[0]
        
        self.logger.info(f"Available LLM providers: {self.available_providers}")
        self.logger.info(f"Using provider: {self.active_provider}")
    
    def _wait_for_rate_limit(self, provider: str):
        """Wait if necessary to respect rate limits and retry delays."""
        # Check if we have a specific retry delay for this provider
        if provider in self.retry_delays:
            retry_until = self.retry_delays[provider]
            if datetime.now() < retry_until:
                wait_time = (retry_until - datetime.now()).total_seconds()
                self.logger.info(f"Rate limit: waiting {wait_time:.1f}s for {provider} (API-specified delay)")
                self._interruptible_sleep(wait_time)
                # Clear the retry delay after waiting
                del self.retry_delays[provider]
        
        # Apply general rate limiting between requests
        rate_limits = {
            'google': 2.0,    # Conservative: ~30 requests per minute (well under 50/day limit)
            'anthropic': 1.0, # Conservative rate for paid API
            'openai': 1.0,    # Conservative rate for paid API
            'ollama': 0.1     # Local, no rate limit needed
        }
        
        min_interval = rate_limits.get(provider, 1.0)
        
        if provider in self.last_request_time:
            time_since_last = (datetime.now() - self.last_request_time[provider]).total_seconds()
            if time_since_last < min_interval:
                wait_time = min_interval - time_since_last
                self.logger.debug(f"Rate limiting: waiting {wait_time:.1f}s for {provider}")
                self._interruptible_sleep(wait_time)
    
    def _interruptible_sleep(self, duration: float):
        """Sleep that can be interrupted by setting should_stop flag."""
        end_time = time.time() + duration
        while time.time() < end_time and not self.should_stop:
            # Sleep in small chunks so we can check should_stop frequently
            remaining = end_time - time.time()
            chunk_duration = min(0.1, remaining)  # Check every 0.1 seconds
            if chunk_duration > 0:
                time.sleep(chunk_duration)
    
    def _handle_rate_limit_error(self, provider: str, error_msg: str):
        """Extract retry delay from rate limit error and schedule retry."""
        try:
            # Try to extract retry_delay from Google's error format
            if "retry_delay" in error_msg and "seconds:" in error_msg:
                # Extract seconds value from error message
                import re
                match = re.search(r'seconds:\s*(\d+)', error_msg)
                if match:
                    delay_seconds = int(match.group(1))
                    # Add some buffer time
                    delay_seconds += 2
                    retry_time = datetime.now() + timedelta(seconds=delay_seconds)
                    self.retry_delays[provider] = retry_time
                    self.logger.warning(f"Rate limit for {provider}: will retry after {delay_seconds}s")
                    return delay_seconds
                    
            # Default delays for different providers
            default_delays = {
                'google': 10,     # Google often needs longer delays
                'anthropic': 60,  # Conservative for paid API
                'openai': 60,     # Conservative for paid API
                'ollama': 1       # Local, minimal delay
            }
            
            delay = default_delays.get(provider, 30)
            retry_time = datetime.now() + timedelta(seconds=delay)
            self.retry_delays[provider] = retry_time
            self.logger.warning(f"Rate limit for {provider}: will retry after {delay}s")
            return delay
            
        except Exception as e:
            self.logger.error(f"Error parsing rate limit response: {e}")
            # Fallback to a reasonable delay
            delay = 30
            retry_time = datetime.now() + timedelta(seconds=delay)
            self.retry_delays[provider] = retry_time
            return delay
        
    def save_checkpoint(self, processed_files: List[str], results: List[dict]):
        """Save current progress to checkpoint file."""
        checkpoint_data = {
            'processed_files': processed_files,
            'results': results,
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_processed': len(results)
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
            
        self.logger.info(f"Checkpoint saved: {len(results)} images processed")
        
    def load_checkpoint(self) -> Tuple[List[str], List[dict]]:
        """Load progress from checkpoint file."""
        if not self.checkpoint_file.exists():
            return [], []
            
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                
            processed_files = checkpoint_data.get('processed_files', [])
            results = checkpoint_data.get('results', [])
            
            self.logger.info(f"Resumed from checkpoint: {len(results)} images already processed")
            return processed_files, results
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            return [], []
        
    def extract_store_number(self, filename: str) -> Optional[int]:
        """Extract store number from filename."""
        match = self.store_pattern.search(filename)
        if match:
            return int(match.group(1))
        return None
        
    def preprocess_image(self, image_path: Path) -> List[np.ndarray]:
        """Preprocess image for better OCR results. Returns multiple preprocessed versions."""
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
                
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            processed_images = []
            
            # Version 1: Original grayscale
            processed_images.append(gray)
            
            # Version 2: Enhanced contrast
            enhanced = cv2.equalizeHist(gray)
            processed_images.append(enhanced)
            
            # Version 3: Gaussian blur + threshold
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images.append(thresh1)
            
            # Version 4: Adaptive threshold
            adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                   cv2.THRESH_BINARY, 11, 2)
            processed_images.append(adaptive_thresh)
            
            # Version 5: For digital displays - try to isolate bright areas
            # This is good for LED/LCD displays showing text
            _, bright_areas = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            processed_images.append(bright_areas)
            
            # Version 6: For dark text on light background
            _, dark_text = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
            processed_images.append(dark_text)
            
            # Version 7: Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
            processed_images.append(cleaned)
            
            return processed_images
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image {image_path}: {e}")
            return [None]
    
    def extract_text_with_llm(self, image_path: Path) -> Tuple[str, float]:
        """
        Extract text from image using LLM vision capabilities with rate limiting and retry.
        
        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        # Create the prompt for device ID extraction
        prompt = """Look at this retail store device image and extract ONLY the device identifier string.

I'm looking for a string that consists of exactly:
- 6 digits followed by either 'G' or 'N'
- Example formats: 123456G, 987654N, 021382G

Please respond with ONLY the device identifier if you can clearly see one that matches this exact pattern. If you cannot clearly see such a string, respond with "NO_DEVICE_ID_FOUND".

Do not include any other text, explanations, or formatting - just the 7-character device identifier or "NO_DEVICE_ID_FOUND"."""

        # Try with retry logic for rate limits
        max_retries = 3
        for attempt in range(max_retries):
            # Check if we should stop before each attempt
            if self.should_stop:
                self.logger.info("Processing stopped before LLM extraction")
                return "", 0.0
                
            try:
                # Wait for rate limit before making request
                self._wait_for_rate_limit(self.active_provider)
                
                # Check again after waiting (in case stopped during wait)
                if self.should_stop:
                    self.logger.info("Processing stopped during rate limit wait")
                    return "", 0.0
                
                if self.active_provider == 'openai':
                    extracted_text = self._extract_with_openai(image_path, prompt)
                elif self.active_provider == 'anthropic':
                    extracted_text = self._extract_with_anthropic(image_path, prompt)
                elif self.active_provider == 'google':
                    extracted_text = self._extract_with_google(image_path, prompt)
                elif self.active_provider == 'ollama':
                    extracted_text = self._extract_with_ollama(image_path, prompt)
                else:
                    raise ValueError(f"Unknown provider: {self.active_provider}")
                
                # Record successful request time
                self.last_request_time[self.active_provider] = datetime.now()
                
                # Calculate confidence based on the response
                if extracted_text == "NO_DEVICE_ID_FOUND":
                    return "", 0.0
                elif re.match(r'^\d{6}[GN]$', extracted_text):
                    # Perfect match to our pattern
                    return extracted_text, 95.0
                else:
                    # LLM returned something, but it doesn't match our pattern exactly
                    return extracted_text, 30.0
                    
            except Exception as e:
                error_msg = str(e).lower()
                
                # Log the full exception details for debugging
                self.logger.error(f"Full exception details for {self.active_provider}:")
                self.logger.error(f"Exception type: {type(e).__name__}")
                self.logger.error(f"Exception message: {str(e)}")
                
                # Check if this is a rate limit error
                if any(keyword in error_msg for keyword in ['429', 'rate limit', 'quota', 'too many requests']):
                    if attempt < max_retries - 1:  # Not the last attempt
                        delay = self._handle_rate_limit_error(self.active_provider, str(e))
                        self.logger.warning(f"Rate limit hit for {self.active_provider}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                        self._interruptible_sleep(delay)
                        # Check if we should stop after the sleep
                        if self.should_stop:
                            self.logger.info("Processing stopped during rate limit wait")
                            return "", 0.0
                        continue
                    else:
                        self.logger.error(f"Rate limit exceeded for {self.active_provider} after {max_retries} attempts")
                        self.logger.error(f"Final error: {str(e)}")
                        return "", 0.0
                else:
                    # Non-rate-limit error
                    self.logger.error(f"Non-rate-limit error extracting text with LLM from {image_path}")
                    self.logger.error(f"Error details: {str(e)}")
                    return "", 0.0
        
        # If we get here, all retries failed
        return "", 0.0
    
    def _extract_with_openai(self, image_path: Path, prompt: str) -> str:
        """Extract using OpenAI GPT-4 Vision."""
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }],
                max_tokens=50,
                temperature=0
            )
            
            result = response.choices[0].message.content.strip()
            self.logger.debug(f"OpenAI GPT-4 response: '{result}'")
            return result
            
        except Exception as e:
            self.logger.error(f"OpenAI GPT-4 API call failed for {image_path.name}")
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error(f"Full error message: {str(e)}")
            raise
    
    def _extract_with_anthropic(self, image_path: Path, prompt: str) -> str:
        """Extract using Anthropic Claude Vision."""
        try:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=50,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }]
            )
            
            result = response.content[0].text.strip()
            self.logger.debug(f"Anthropic Claude response: '{result}'")
            return result
            
        except Exception as e:
            self.logger.error(f"Anthropic Claude API call failed for {image_path.name}")
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error(f"Full error message: {str(e)}")
            raise
    
    def _extract_with_google(self, image_path: Path, prompt: str) -> str:
        """Extract using Google Gemini Vision."""
        try:
            from PIL import Image as PILImage
            
            image = PILImage.open(image_path)
            response = self.gemini_model.generate_content([prompt, image])
            
            # Log successful response details
            self.logger.debug(f"Google Gemini response received for {image_path.name}")
            if hasattr(response, 'text') and response.text:
                result = response.text.strip()
                self.logger.debug(f"Response text: '{result}'")
                return result
            else:
                self.logger.warning(f"Google Gemini returned empty response for {image_path.name}")
                self.logger.warning(f"Response object: {response}")
                return "NO_DEVICE_ID_FOUND"
                
        except Exception as e:
            # Log detailed error information
            self.logger.error(f"Google Gemini API call failed for {image_path.name}")
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error(f"Full error message: {str(e)}")
            
            # Try to extract additional details from the response if available
            if hasattr(e, 'response'):
                self.logger.error(f"HTTP response: {e.response}")
            if hasattr(e, 'details'):
                self.logger.error(f"Error details: {e.details}")
            
            # Re-raise to be handled by the retry logic
            raise
    
    def _extract_with_ollama(self, image_path: Path, prompt: str) -> str:
        """Extract using local Ollama with LLaVA."""
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        response = requests.post('http://localhost:11434/api/generate', 
            json={
                "model": "llava",
                "prompt": prompt,
                "images": [base64_image],
                "stream": False
            })
        
        if response.status_code == 200:
            return response.json().get('response', '').strip()
        else:
            raise Exception(f"Ollama request failed: {response.status_code}")
            
    def extract_text_from_image(self, image_path: Path) -> Tuple[str, float]:
        """
        Extract text from image using either LLM or OCR.
        
        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        # Use LLM extraction if enabled
        if self.use_llm:
            return self.extract_text_with_llm(image_path)
        
        # Fall back to OCR extraction
        try:
            # Preprocess image (get multiple versions)
            processed_images = self.preprocess_image(image_path)
            if not processed_images or processed_images[0] is None:
                return "", 0.0
                
            # Configure Tesseract for device IDs (6 digits + G or N)
            # Use different PSM (Page Segmentation Mode) values
            configs = [
                '--psm 8 -c tessedit_char_whitelist=0123456789GN',  # Single word, only digits and G/N
                '--psm 7 -c tessedit_char_whitelist=0123456789GN',  # Single text line, only digits and G/N
                '--psm 6 -c tessedit_char_whitelist=0123456789GN',  # Single block, only digits and G/N
                '--psm 13 -c tessedit_char_whitelist=0123456789GN', # Raw line, only digits and G/N
                '--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',  # Single word, all chars
                '--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',  # Single text line, all chars
                '--psm 6',  # Single block without whitelist
                '--psm 8',  # Single word without whitelist
            ]
            
            best_text = ""
            best_confidence = 0.0
            
            # Try OCR on each preprocessed version (but limit to first few for speed)
            # We'll try the most promising preprocessing methods first
            for img_idx, processed_img in enumerate(processed_images[:4]):  # Only try first 4 preprocessing methods
                if processed_img is None:
                    continue
                    
                # Try only the most effective configs first
                priority_configs = configs[:4]  # Focus on targeted configs
                
                for config in priority_configs:
                    try:
                        # Try simple extraction first (faster)
                        simple_text = pytesseract.image_to_string(processed_img, config=config).strip()
                        if simple_text and len(simple_text) > 2:
                            # Check if we found our target pattern immediately
                            device_id, score = self.find_device_id(simple_text)
                            if device_id and score >= 0.9:
                                # Found high-confidence match, return early
                                return simple_text, 95.0
                            
                            # Estimate confidence based on content
                            confidence = 50.0
                            if len(simple_text) >= 6:
                                confidence += 20.0
                            if re.search(r'\d{6}[GN]', simple_text):
                                confidence += 25.0  # Bonus for target pattern
                            elif re.search(r'\d{6,}', simple_text):
                                confidence += 15.0
                            if re.search(r'[GN]', simple_text):
                                confidence += 10.0
                                
                            if confidence > best_confidence:
                                best_text = simple_text
                                best_confidence = confidence
                                
                    except Exception as e:
                        continue
                        
                # If we haven't found a good match yet, try detailed extraction on promising images
                if best_confidence < 70.0:
                    for config in priority_configs[:2]:  # Only try top 2 configs for detailed extraction
                        try:
                            data = pytesseract.image_to_data(processed_img, config=config, output_type=pytesseract.Output.DICT)
                            
                            text_parts = []
                            confidences = []
                            
                            for i, conf in enumerate(data['conf']):
                                if int(conf) > 30:
                                    text = data['text'][i].strip()
                                    if text and len(text) > 1:
                                        text_parts.append(text)
                                        confidences.append(int(conf))
                            
                            if text_parts:
                                combined_text = ' '.join(text_parts)
                                avg_confidence = sum(confidences) / len(confidences)
                                
                                # Check for target pattern
                                device_id, score = self.find_device_id(combined_text)
                                if device_id and score >= 0.9:
                                    return combined_text, avg_confidence
                                
                                if avg_confidence > best_confidence:
                                    best_text = combined_text
                                    best_confidence = avg_confidence
                                    
                        except Exception as e:
                            continue
                    
            return best_text, best_confidence
            
        except Exception as e:
            self.logger.error(f"Error extracting text from {image_path}: {e}")
            return "", 0.0
            
    def find_device_id(self, text: str) -> Tuple[Optional[str], float]:
        """
        Find device ID in extracted text (6 digits + G or N).
        
        Returns:
            Tuple of (device_id, quality_score)
        """
        if not text:
            return None, 0.0
            
        # Clean up text and try different cleaning approaches
        text_variants = [
            text,
            re.sub(r'\s+', '', text),  # Remove all spaces
            re.sub(r'[^\w\s]', '', text),  # Remove punctuation
            re.sub(r'\s+', ' ', text.strip()),  # Normalize spaces
            text.replace('O', '0'),  # Replace O with 0
            text.replace('o', '0'),  # Replace lowercase o with 0
        ]
        
        best_match = None
        best_score = 0.0
        
        for text_variant in text_variants:
            for i, pattern in enumerate(self.device_id_patterns):
                matches = pattern.findall(text_variant)
                for match in matches:
                    # Clean up the match
                    if isinstance(match, tuple):  # For grouped patterns
                        match = ''.join(match)
                    
                    # Remove spaces from match
                    clean_match = re.sub(r'\s+', '', match)
                    
                    # Replace any remaining O's with 0's
                    clean_match = clean_match.replace('O', '0').replace('o', '0')
                    
                    # Validate the match format
                    if re.match(r'^\d{6}[GN]$', clean_match):
                        # Perfect match - score high
                        score = 1.0
                        
                        # Prefer exact pattern matches (first patterns)
                        if i == 0:
                            score = 1.0
                        elif i == 1:
                            score = 0.95
                        else:
                            score = 0.9
                            
                        if score > best_score:
                            best_match = clean_match
                            best_score = score
                    
                    elif len(clean_match) == 7 and clean_match[-1] in 'GN':
                        # Close match but needs validation
                        score = 0.8
                        if score > best_score:
                            best_match = clean_match
                            best_score = score
                            
        return best_match, best_score
        
    def assess_quality(self, ocr_confidence: float, device_id: Optional[str], 
                      device_score: float) -> Tuple[str, bool]:
        """
        Assess the quality of the extraction.
        
        Returns:
            Tuple of (quality_flag, should_flag)
        """
        flags = []
        should_flag = False
        
        # Check OCR confidence
        if ocr_confidence < 50:
            flags.append("LOW_OCR_CONFIDENCE")
            should_flag = True
            
        # Check if device ID was found
        if device_id is None:
            flags.append("NO_DEVICE_ID_FOUND")
            should_flag = True
        else:
            # Check device ID quality
            if device_score < 0.7:
                flags.append("UNCERTAIN_DEVICE_ID")
                should_flag = True
                
            # Check device ID format
            if len(device_id) < 6:
                flags.append("SHORT_DEVICE_ID")
                should_flag = True
                
        if not flags:
            flags.append("OK")
            
        return "|".join(flags), should_flag
        
    def process_single_image(self, image_path: Path) -> dict:
        """Process a single image and return results."""
        filename = image_path.name
        
        # Extract store number
        store_number = self.extract_store_number(filename)
        
        # Extract text from image
        extracted_text, ocr_confidence = self.extract_text_from_image(image_path)
        
        # Find device ID
        device_id, device_score = self.find_device_id(extracted_text)
        
        # Assess quality
        quality_flag, should_flag = self.assess_quality(ocr_confidence, device_id, device_score)
        
        result = {
            'filename': filename,
            'store_number': store_number,
            'extracted_text': extracted_text,
            'device_id': device_id,
            'ocr_confidence': round(ocr_confidence, 2),
            'device_score': round(device_score, 2),
            'quality_flag': quality_flag,
            'needs_review': should_flag
        }
        
        return result
        
    def process_all_images(self) -> List[dict]:
        """Process all images in the datasource directory with checkpointing."""
        if not self.datasource_dir.exists():
            raise FileNotFoundError(f"Datasource directory not found: {self.datasource_dir}")
            
        # Find all JPG files
        image_files = list(self.datasource_dir.glob("*.jpg")) + list(self.datasource_dir.glob("*.JPG"))
        
        if not image_files:
            raise ValueError(f"No JPG files found in {self.datasource_dir}")
            
        # Load existing progress if any
        processed_files, results = self.load_checkpoint()
        processed_set = set(processed_files)
        
        # Filter out already processed files
        remaining_files = [f for f in image_files if f.name not in processed_set]
        
        total_files = len(image_files)
        already_processed = len(processed_files)
        
        self.logger.info(f"Found {total_files} total images")
        if already_processed > 0:
            self.logger.info(f"Resuming: {already_processed} already processed, {len(remaining_files)} remaining")
        else:
            self.logger.info(f"Starting fresh: {len(remaining_files)} images to process")
        
        # Process remaining files
        for i, image_path in enumerate(remaining_files):
            if self.should_stop:
                self.logger.info("Processing stopped by user request")
                # Save final checkpoint before breaking
                if results:
                    self.save_checkpoint(processed_files, results)
                break
                
            current_index = already_processed + i + 1
            self.logger.info(f"Processing {current_index}/{total_files}: {image_path.name}")
            
            try:
                result = self.process_single_image(image_path)
                results.append(result)
                processed_files.append(image_path.name)
                
                # Save checkpoint every 10 images or if interrupted
                if (current_index % 10 == 0) or self.should_stop:
                    self.save_checkpoint(processed_files, results)
                    
                # Log progress every 25 images
                if current_index % 25 == 0:
                    success_count = len([r for r in results if r['device_id'] is not None])
                    success_rate = (success_count / len(results)) * 100 if results else 0
                    self.logger.info(f"Progress: {current_index}/{total_files} ({success_rate:.1f}% success rate)")
                    
            except Exception as e:
                self.logger.error(f"Error processing {image_path}: {e}")
                # Add error result
                error_result = {
                    'filename': image_path.name,
                    'store_number': self.extract_store_number(image_path.name),
                    'extracted_text': '',
                    'device_id': None,
                    'ocr_confidence': 0.0,
                    'device_score': 0.0,
                    'quality_flag': 'PROCESSING_ERROR',
                    'needs_review': True
                }
                results.append(error_result)
                processed_files.append(image_path.name)
                
        # Final checkpoint save
        if results:
            self.save_checkpoint(processed_files, results)
                
        return results
        
    def save_results_to_csv(self, results: List[dict], output_file: str = "store_device_mapping.csv"):
        """Save results to CSV file."""
        df = pd.DataFrame(results)
        
        # Sort by store number
        df = df.sort_values('store_number', na_position='last')
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
        # Create summary statistics
        total_images = len(results)
        successful_extractions = len([r for r in results if r['device_id'] is not None])
        flagged_images = len([r for r in results if r['needs_review']])
        
        summary = {
            'total_images': total_images,
            'successful_extractions': successful_extractions,
            'success_rate': f"{(successful_extractions/total_images)*100:.1f}%",
            'flagged_for_review': flagged_images,
            'flagged_rate': f"{(flagged_images/total_images)*100:.1f}%"
        }
        
        self.logger.info("Processing Summary:")
        for key, value in summary.items():
            self.logger.info(f"  {key}: {value}")
            
        # Save summary
        with open('processing_summary.txt', 'w') as f:
            f.write("Store Device Mapping - Processing Summary\n")
            f.write("=" * 45 + "\n\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
                
        return output_file


def main():
    """Main function to run the processing."""
    try:
        processor = StoreImageProcessor()
        
        # Process all images
        results = processor.process_all_images()
        
        # Save results
        output_file = processor.save_results_to_csv(results)
        
        print(f"\nProcessing complete!")
        print(f"Results saved to: {output_file}")
        print(f"Log file: store_extraction.log")
        print(f"Summary: processing_summary.txt")
        
        # Show sample results
        df = pd.read_csv(output_file)
        print(f"\nSample results:")
        print(df.head(10).to_string(index=False))
        
        print(f"\nImages needing review:")
        flagged_df = df[df['needs_review'] == True]
        if not flagged_df.empty:
            print(flagged_df[['filename', 'store_number', 'quality_flag']].head(10).to_string(index=False))
        else:
            print("No images flagged for review!")
            
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()