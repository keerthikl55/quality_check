from fastapi import FastAPI, UploadFile, Form ,File
from fastapi.responses import StreamingResponse, JSONResponse
import fitz  # PyMuPDF
import io
import re
import base64
from typing import List, Dict, Optional
import json
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from difflib import SequenceMatcher

app = FastAPI()

def detect_invoice_table_structure(page):
    """
    Detect invoice table structure using the first data row as the main key.
    Returns header information and all row coordinates.
    """
    text_dict = page.get_text("dict")
    
    # Collect all text blocks with positions
    text_blocks = []
    for block in text_dict["blocks"]:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["text"].strip():
                        text_blocks.append({
                            "text": span["text"].strip(),
                            "bbox": span["bbox"],
                            "x": span["bbox"][0],
                            "y": span["bbox"][1],
                            "font_size": span["size"],
                            "font_name": span["font"]
                        })
    
    # Sort by Y position (top to bottom), then X position (left to right)
    text_blocks.sort(key=lambda x: (x["y"], x["x"]))
    
    # Find table header keywords
    header_keywords = [
        "SR NO", "S NO", "DESCRIPTION", "PARTICULARS", "QTY", "QUANTITY", 
        "UNIT", "PRICE", "AMOUNT", "HS CODE", "PKG", "REF", "GROSS WEIGHT",
        "FULL DESCRIPTION", "GOODS"
    ]
    
    header_blocks = []
    header_y_position = None
    
    # Identify header row
    for block in text_blocks:
        text_upper = block["text"].upper()
        if any(keyword in text_upper for keyword in header_keywords):
            if header_y_position is None:
                header_y_position = block["y"]
            # Group blocks that are on the same horizontal line (within 10 pixels)
            if abs(block["y"] - header_y_position) <= 10:
                header_blocks.append(block)
    
    if not header_blocks:
        return {"error": "No table header found", "header_blocks": [], "table_rows": []}
    
    # Sort header blocks by X position
    header_blocks.sort(key=lambda x: x["x"])
    
    # Create column boundaries based on header positions
    column_boundaries = []
    for i, header in enumerate(header_blocks):
        start_x = header["x"]
        end_x = header_blocks[i + 1]["x"] if i + 1 < len(header_blocks) else page.rect.width
        column_boundaries.append({
            "column_name": header["text"],
            "start_x": start_x,
            "end_x": end_x,
            "header_bbox": header["bbox"]
        })
    
    # Find data rows (everything below header)
    data_start_y = header_y_position + 20  # Start looking for data 20 pixels below header
    data_blocks = [block for block in text_blocks if block["y"] > data_start_y]
    
    # Group data blocks into rows
    rows = []
    current_row = []
    current_y = None
    
    for block in data_blocks:
        # Skip if text looks like footer or non-data content
        text_upper = block["text"].upper()
        if any(skip_word in text_upper for skip_word in ["TOTAL", "CONTAINER", "PORT OF", "COUNTRY", "NET WEIGHT", 
                                                        "GROSS WEIGHT", "BANK ACCOUNT", "CURRENCY", "VAT", "AED", "USD",
                                                        "TERMS OF SALE", "DOT 20", "PFI", "SOFTWARE", "PAGE"]):
            continue
            
        # Group blocks by Y position (same row)
        if current_y is None or abs(block["y"] - current_y) <= 8:
            current_row.append(block)
            current_y = block["y"] if current_y is None else current_y
        else:
            # Process completed row
            if current_row and len(current_row) >= 2:  # At least 2 columns to be considered a valid row
                rows.append(process_invoice_row(current_row, column_boundaries))
            current_row = [block]
            current_y = block["y"]
    
    # Process last row
    if current_row and len(current_row) >= 2:
        rows.append(process_invoice_row(current_row, column_boundaries))
    
    # Filter valid rows
    valid_rows = [row for row in rows if row and row.get("is_valid_data_row")]
    
    return {
        "header_info": {
            "columns": [{"name": col["column_name"], "x_range": [col["start_x"], col["end_x"]]} for col in column_boundaries],
            "header_y_position": header_y_position,
            "total_columns": len(column_boundaries)
        },
        "table_rows": valid_rows,
        "total_data_rows": len(valid_rows)
    }

def process_invoice_row(row_blocks, column_boundaries):
    """
    Process a single invoice row and map text to appropriate columns.
    """
    row_blocks.sort(key=lambda x: x["x"])
    
    # Calculate full row bounding box
    min_x = min(block["bbox"][0] for block in row_blocks)
    min_y = min(block["bbox"][1] for block in row_blocks)
    max_x = max(block["bbox"][2] for block in row_blocks)
    max_y = max(block["bbox"][3] for block in row_blocks)
    
    full_row_bbox = [min_x, min_y, max_x, max_y]
    
    # Map text to columns
    row_data = {}
    for block in row_blocks:
        block_x = block["x"]
        # Find which column this text belongs to
        for col in column_boundaries:
            if col["start_x"] <= block_x < col["end_x"]:
                col_name = col["column_name"].lower().replace(" ", "_")
                if col_name not in row_data:
                    row_data[col_name] = []
                row_data[col_name].append(block["text"])
                break
    
    # Combine text for each column
    for col_name in row_data:
        row_data[col_name] = " ".join(row_data[col_name])
    
    # Check if this is a valid data row (has meaningful content)
    is_valid = False
    if row_data:
        # Check for typical invoice row indicators
        text_content = " ".join(row_data.values()).upper()
        
        # Check if row starts with a product/reference code (like 581511, 532421)
        first_text = row_blocks[0]["text"] if row_blocks else ""
        has_product_code = bool(re.match(r'^\d{6}', first_text))  # 6-digit product codes
        
        # Must have some substantial content and not be just numbers/codes
        if len(text_content) > 10 and any(indicator in text_content for indicator in 
                                        ["OIL", "MOTOR", "DIESEL", "GEAR", "HYDRAULIC", "TYRE", "EAGLE", "CASTLE", 
                                         "GOODYEAR", "EAG", "ASY", "XL", "FP"]):
            is_valid = True
        # Or if it has serial number + description pattern
        elif any(key for key in row_data.keys() if "sr" in key.lower() or "no" in key.lower() or "ref" in key.lower()):
            if len(text_content) > 15:
                is_valid = True
        # Or if it starts with a product code and has sufficient content
        elif has_product_code and len(text_content) > 20:
            is_valid = True
    
    return {
        "row_data": row_data,
        "full_row_bbox": full_row_bbox,
        "row_coordinates": {
            "x0": min_x,
            "y0": min_y, 
            "x1": max_x,
            "y1": max_y
        },
        "is_valid_data_row": is_valid,
        "text_blocks_count": len(row_blocks)
    }


def extract_text_with_multiple_methods(page):
    """
    Extract text using multiple methods to handle scanned documents better.
    """
    all_text_blocks = []
    
    # Method 1: Standard text extraction with words
    try:
        words = page.get_text("words")
        for word in words:
            if len(word) >= 5 and word[4].strip():  # word[4] is text
                all_text_blocks.append({
                    "text": word[4].strip(),
                    "bbox": [word[0], word[1], word[2], word[3]],
                    "method": "words"
                })
    except:
        pass
    
    # Method 2: Dictionary-based extraction
    try:
        text_dict = page.get_text("dict")
        for block in text_dict["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    line_text = ""
                    line_bbox = None
                    for span in line["spans"]:
                        if span["text"].strip():
                            line_text += span["text"] + " "
                            if line_bbox is None:
                                line_bbox = list(span["bbox"])
                            else:
                                # Expand bbox to include this span
                                line_bbox[0] = min(line_bbox[0], span["bbox"][0])
                                line_bbox[1] = min(line_bbox[1], span["bbox"][1])
                                line_bbox[2] = max(line_bbox[2], span["bbox"][2])
                                line_bbox[3] = max(line_bbox[3], span["bbox"][3])
                    
                    if line_text.strip() and line_bbox:
                        all_text_blocks.append({
                            "text": line_text.strip(),
                            "bbox": line_bbox,
                            "method": "dict"
                        })
    except:
        pass
    
    # Method 3: Block-based extraction
    try:
        blocks = page.get_text("blocks")
        for block in blocks:
            if len(block) >= 5 and block[4].strip():  # block[4] is text
                all_text_blocks.append({
                    "text": block[4].strip(),
                    "bbox": [block[0], block[1], block[2], block[3]],
                    "method": "blocks"
                })
    except:
        pass
    
    return all_text_blocks

def enhanced_search_multiline_text(page, query, max_line_gap=100, fuzzy_threshold=0.7):
    """
    Enhanced multiline text search with fuzzy matching for scanned documents.
    """
    # Get all text blocks using multiple methods
    all_text_blocks = extract_text_with_multiple_methods(page)
    
    if not all_text_blocks:
        return []
    
    # Sort by vertical position, then horizontal
    all_text_blocks.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
    
    # Normalize query
    normalized_query = normalize_text(query)
    query_words = normalized_query.split()
    
    if not query_words:
        return []
    
    matches = []
    
    # Method 1: Exact phrase search in individual blocks
    for block in all_text_blocks:
        block_text = normalize_text(block["text"])
        if normalized_query in block_text:
            matches.append(block["bbox"])
    
    # Method 2: Fuzzy matching for individual blocks
    for block in all_text_blocks:
        score = fuzzy_match_score(block["text"], query)
        if score >= fuzzy_threshold:
            matches.append(block["bbox"])
    
    # Method 3: Word-by-word fuzzy search across blocks
    for i in range(len(all_text_blocks)):
        current_blocks = [all_text_blocks[i]]
        current_text = normalize_text(all_text_blocks[i]["text"])
        current_words = current_text.split()
        
        # Try to match query words starting from this block
        matched_words = 0
        j = i + 1
        
        while matched_words < len(query_words) and j < len(all_text_blocks):
            # Check if next block is close enough (same line or nearby)
            y_diff = abs(all_text_blocks[j]["bbox"][1] - all_text_blocks[i]["bbox"][1])
            if y_diff > max_line_gap:
                break
            
            current_blocks.append(all_text_blocks[j])
            next_text = normalize_text(all_text_blocks[j]["text"])
            current_words.extend(next_text.split())
            j += 1
        
        # Check if current combination of blocks contains query
        combined_text = " ".join(current_words)
        if fuzzy_match_score(combined_text, normalized_query) >= fuzzy_threshold:
            # Calculate combined bounding box
            min_x = min(block["bbox"][0] for block in current_blocks)
            min_y = min(block["bbox"][1] for block in current_blocks)
            max_x = max(block["bbox"][2] for block in current_blocks)
            max_y = max(block["bbox"][3] for block in current_blocks)
            matches.append([min_x, min_y, max_x, max_y])
    
    # Method 4: Partial word matching for very unclear scans
    if not matches and len(query_words) == 1:
        single_word = query_words[0]
        for block in all_text_blocks:
            block_words = normalize_text(block["text"]).split()
            for word in block_words:
                if (len(word) >= 3 and len(single_word) >= 3 and 
                    fuzzy_match_score(word, single_word) >= 0.8):
                    matches.append(block["bbox"])
                    break
    
    # Remove duplicate matches (same area)
    unique_matches = []
    for match in matches:
        is_duplicate = False
        for existing in unique_matches:
            # Check if bounding boxes overlap significantly
            overlap_x = max(0, min(match[2], existing[2]) - max(match[0], existing[0]))
            overlap_y = max(0, min(match[3], existing[3]) - max(match[1], existing[1]))
            overlap_area = overlap_x * overlap_y
            
            match_area = (match[2] - match[0]) * (match[3] - match[1])
            existing_area = (existing[2] - existing[0]) * (existing[3] - existing[1])
            
            if overlap_area > 0.5 * min(match_area, existing_area):
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_matches.append(match)
    
    return unique_matches

def extract_table_data(page):

    words = page.get_text("words")  
    words.sort(key=lambda w: (w[1], w[0]))  
    
    rows = []
    current_row_y = None
    current_row = []

    for word in words:
        x0, y0, x1, y1, text = word[:5]
        if current_row_y is None or abs(y0 - current_row_y) <= 2:
            current_row.append(word)
            current_row_y = y0
        else:
            rows.append(current_row)
            current_row = [word]
            current_row_y = y0
    if current_row:
        rows.append(current_row)

    extracted_rows = []
    for row in rows:
        texts = [w[4] for w in row]
        bbox = [min(w[0] for w in row), min(w[1] for w in row),
                max(w[2] for w in row), max(w[3] for w in row)]
        
        extracted_rows.append({
            "text": " ".join(texts),
            "bbox": bbox,
            "words": row,
            "full_row_bbox": bbox  # Optional - same as bbox here, but can be adjusted
        })

    return extracted_rows


def process_table_row(row_blocks, particulars_column_x):
    """
    Process a single table row to extract relevant data.
    """
    row_blocks.sort(key=lambda x: x["x"])
    
    # Find particulars text (usually the longest text in the row)
    particulars_text = ""
    particulars_bbox = None
    s_number = ""
    qty = ""
    
    # Extract S# (serial number) - usually first column
    if row_blocks and re.match(r'^\d+$', row_blocks[0]["text"]):
        s_number = row_blocks[0]["text"]
    
    # Find particulars - look for product description patterns
    for block in row_blocks:
        text = block["text"]
        # Identify product descriptions (contains oil, engine, etc.)
        if (len(text) > 15 and 
            any(keyword in text.upper() for keyword in ["OIL", "ENGINE", "GEAR", "HYDRAULIC", "MOTOR"])):
            if len(text) > len(particulars_text):  # Take the longest matching text
                particulars_text = text
                particulars_bbox = block["bbox"]
    
    # Extract quantity (look for patterns like "399.00", "192.00")
    for block in row_blocks:
        text = block["text"]
        if re.match(r'^\d+\.?\d*$', text) and len(text) >= 3:
            qty = text
            break
    
    if particulars_text:
        return {
            "s_number": s_number,
            "particulars": particulars_text,
            "qty": qty,
            "bbox": particulars_bbox,
            "full_row_bbox": calculate_row_bbox(row_blocks)
        }
    
    return None

def calculate_row_bbox(row_blocks):
    """Calculate bounding box for entire row."""
    if not row_blocks:
        return None
    
    min_x = min(block["bbox"][0] for block in row_blocks)
    min_y = min(block["bbox"][1] for block in row_blocks)
    max_x = max(block["bbox"][2] for block in row_blocks)
    max_y = max(block["bbox"][3] for block in row_blocks)
    
    return [min_x, min_y, max_x, max_y]

def enhance_image_quality(pil_image):
    """
    Apply multiple enhancement techniques to improve image quality.
    """
    # Convert to RGB if not already
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # 1. Enhance sharpness
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(1.5)  # Increase sharpness by 50%
    
    # 2. Enhance contrast
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.2)  # Increase contrast by 20%
    
    # 3. Enhance brightness slightly
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(1.1)  # Increase brightness by 10%
    
    # 4. Apply unsharp mask filter for better text clarity
    pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    
    return pil_image

def create_quality_snippet(page, bbox, padding=20, quality_level="high"):
    """
    Create snippet image with different quality levels.
    
    Args:
        quality_level: "standard", "high", or "ultra"
    """
    # Quality settings
    quality_settings = {
        "standard": {"zoom": 2.0, "enhance": False, "max_width": 800, "max_height": 600},
        "high": {"zoom": 4.0, "enhance": True, "max_width": 1200, "max_height": 800},
        "ultra": {"zoom": 6.0, "enhance": True, "max_width": 1600, "max_height": 1200}
    }
    
    settings = quality_settings.get(quality_level, quality_settings["high"])
    
    # Add padding to the bounding box
    page_rect = page.rect
    padded_rect = fitz.Rect(
        max(bbox[0] - padding, 0),
        max(bbox[1] - padding, 0),
        min(bbox[2] + padding, page_rect.width),
        min(bbox[3] + padding, page_rect.height)
    )
    
    matrix = fitz.Matrix(settings["zoom"], settings["zoom"])
    
    pix = page.get_pixmap(
        clip=padded_rect, 
        matrix=matrix,
        alpha=False,  # No transparency for better file size
        colorspace=fitz.csRGB  # Ensure RGB colorspace
    )
    
    # Convert to PIL Image for enhancement
    img_bytes = pix.tobytes("png")
    pil_image = Image.open(io.BytesIO(img_bytes))
    
    # Apply quality enhancements
    if settings["enhance"]:
        pil_image = enhance_image_quality(pil_image)
    
    # Resize to reasonable dimensions while maintaining quality
    max_width = settings["max_width"]
    max_height = settings["max_height"]
    
    if pil_image.width > max_width or pil_image.height > max_height:
        # Calculate resize ratio maintaining aspect ratio
        width_ratio = max_width / pil_image.width
        height_ratio = max_height / pil_image.height
        resize_ratio = min(width_ratio, height_ratio)
        
        new_width = int(pil_image.width * resize_ratio)
        new_height = int(pil_image.height * resize_ratio)
        
        # Use LANCZOS resampling for high quality resize
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Convert back to bytes with high quality settings
    output_buffer = io.BytesIO()
    pil_image.save(
        output_buffer, 
        format='PNG', 
        optimize=True,
        compress_level=6  # Good compression without quality loss
    )
    
    img_bytes = output_buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    return {
        "image_data": img_base64,
        "image_url": f"data:image/png;base64,{img_base64}",
        "dimensions": {"width": pil_image.width, "height": pil_image.height},
        "file_size_kb": len(img_bytes) / 1024,
        "quality_level": quality_level
    }

def search_multiline_text(page, query, max_line_gap=50):
    """
    Search for text that might span multiple lines in a PDF page with exact word order matching.
    This version builds a word-level index to allow flexible line breaks.
    """
    # Flatten all spans into a list of words with coordinates and y position
    text_dict = page.get_text("dict")
    word_entries = []

    for block in text_dict["blocks"]:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    for word in span["text"].split():
                        word_entries.append({
                            "text": word,
                            "bbox": span["bbox"],
                            "y_center": (span["bbox"][1] + span["bbox"][3]) / 2
                        })

    if not word_entries:
        return []

    # Normalize query into words
    query_words = [w.upper() for w in query.strip().split()]
    if not query_words:
        return []

    # Normalize all page words
    page_words = [(i, w["text"].upper()) for i, w in enumerate(word_entries)]

    matches = []
    i = 0
    while i <= len(page_words) - len(query_words):
        match = True
        for j in range(len(query_words)):
            if page_words[i + j][1] != query_words[j]:
                match = False
                break
        if match:
            idx_range = range(i, i + len(query_words))
            matched_boxes = [word_entries[idx]["bbox"] for idx in idx_range]
            x0 = min(b[0] for b in matched_boxes)
            y0 = min(b[1] for b in matched_boxes)
            x1 = max(b[2] for b in matched_boxes)
            y1 = max(b[3] for b in matched_boxes)
            matches.append([x0, y0, x1, y1])
            i += len(query_words)
        else:
            i += 1

    return matches



    
@app.post("/get-snippet-image")
async def get_snippet_image(
    file: UploadFile,
    page: int = Form(...),
    x0: float = Form(...),
    y0: float = Form(...),
    x1: float = Form(...),
    y1: float = Form(...),
    padding: int = Form(20),
    quality_level: str = Form("high")
):
    """
    Get high-quality snippet image for specific coordinates.
    
    Args:
        quality_level: "standard", "high", or "ultra"
    """
    try:
        pdf_bytes = await file.read()
        stream = io.BytesIO(pdf_bytes)
        doc = fitz.open(stream=stream, filetype="pdf")
        
        if page < 1 or page > len(doc):
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid page number"}
            )
        
        selected_page = doc[page - 1]
        bbox = [x0, y0, x1, y1]
        
        # Create snippet based on quality level
        snippet = create_quality_snippet(selected_page, bbox, padding, quality_level)
        
        # Return as streaming image
        img_bytes = base64.b64decode(snippet["image_data"])
        return StreamingResponse(
            io.BytesIO(img_bytes), 
            media_type="image/png",
            headers={
                "Content-Disposition": f"inline; filename=snippet_p{page}_quality_{quality_level}.png",
                "X-Image-Quality": quality_level,
                "X-Image-Size": f"{snippet['dimensions']['width']}x{snippet['dimensions']['height']}",
                "X-File-Size-KB": str(snippet.get("file_size_kb", 0))
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# UTILITY ENDPOINTS
import re
import fitz
from difflib import SequenceMatcher

def normalize_text(text):
    """
    Enhanced text normalization for scanned documents with OCR errors.
    """
    if not text:
        return ""
    
    # Convert to uppercase for case-insensitive matching
    text = text.upper()
    
    # Common OCR error corrections
    ocr_corrections = {
        # Common character misreadings
        '0': 'O',  # Zero to O
        'O': '0',  # O to zero (bidirectional)
        '1': 'I',  # One to I
        'I': '1',  # I to one
        '5': 'S',  # Five to S
        'S': '5',  # S to five
        '6': 'G',  # Six to G
        'G': '6',  # G to six
        '8': 'B',  # Eight to B
        'B': '8',  # B to eight
        'Z': '2',  # Z to 2
        '2': 'Z',  # 2 to Z
        # Common punctuation issues
        '.': '',   # Remove dots that might be noise
        ',': '',   # Remove commas
        ';': '',   # Remove semicolons
        ':': '',   # Remove colons
        # Remove extra spaces
        '  ': ' ',
        '   ': ' ',
    }
    
    # Apply corrections
    corrected_text = text
    for wrong, correct in ocr_corrections.items():
        corrected_text = corrected_text.replace(wrong, correct)
    
    # Remove multiple spaces and strip
    corrected_text = re.sub(r'\s+', ' ', corrected_text).strip()
    
    return corrected_text


def fuzzy_match_score(text1, text2, threshold=0.6):
    """
    Calculate fuzzy matching score between two text strings.
    Returns score between 0 and 1, where 1 is perfect match.
    """
    if not text1 or not text2:
        return 0.0
    
    # Normalize both texts
    norm_text1 = normalize_text(text1)
    norm_text2 = normalize_text(text2)
    
    # Direct match first
    if norm_text1 == norm_text2:
        return 1.0
    
    # Fuzzy matching using SequenceMatcher
    similarity = SequenceMatcher(None, norm_text1, norm_text2).ratio()
    
    # Also check if one text contains the other (partial match)
    if norm_text1 in norm_text2 or norm_text2 in norm_text1:
        similarity = max(similarity, 0.8)
    
    return similarity

def extract_text_with_multiple_methods(page):
    """
    Extract text using multiple methods to handle scanned documents better.
    """
    all_text_blocks = []
    
    # Method 1: Standard text extraction with words
    try:
        words = page.get_text("words")
        for word in words:
            if len(word) >= 5 and word[4].strip():  # word[4] is text
                all_text_blocks.append({
                    "text": word[4].strip(),
                    "bbox": [word[0], word[1], word[2], word[3]],
                    "method": "words"
                })
    except:
        pass
    
    # Method 2: Dictionary-based extraction
    try:
        text_dict = page.get_text("dict")
        for block in text_dict["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    line_text = ""
                    line_bbox = None
                    for span in line["spans"]:
                        if span["text"].strip():
                            line_text += span["text"] + " "
                            if line_bbox is None:
                                line_bbox = list(span["bbox"])
                            else:
                                # Expand bbox to include this span
                                line_bbox[0] = min(line_bbox[0], span["bbox"][0])
                                line_bbox[1] = min(line_bbox[1], span["bbox"][1])
                                line_bbox[2] = max(line_bbox[2], span["bbox"][2])
                                line_bbox[3] = max(line_bbox[3], span["bbox"][3])
                    
                    if line_text.strip() and line_bbox:
                        all_text_blocks.append({
                            "text": line_text.strip(),
                            "bbox": line_bbox,
                            "method": "dict"
                        })
    except:
        pass
    
    # Method 3: Block-based extraction
    try:
        blocks = page.get_text("blocks")
        for block in blocks:
            if len(block) >= 5 and block[4].strip():  # block[4] is text
                all_text_blocks.append({
                    "text": block[4].strip(),
                    "bbox": [block[0], block[1], block[2], block[3]],
                    "method": "blocks"
                })
    except:
        pass
    
    return all_text_blocks

def enhanced_search_multiline_text(page, query, max_line_gap=100, fuzzy_threshold=0.7):
    """
    Enhanced multiline text search with fuzzy matching for scanned documents.
    """
    # Get all text blocks using multiple methods
    all_text_blocks = extract_text_with_multiple_methods(page)
    
    if not all_text_blocks:
        return []
    
    # Sort by vertical position, then horizontal
    all_text_blocks.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
    
    # Normalize query
    normalized_query = normalize_text(query)
    query_words = normalized_query.split()
    
    if not query_words:
        return []
    
    matches = []
    
    # Method 1: Exact phrase search in individual blocks
    for block in all_text_blocks:
        block_text = normalize_text(block["text"])
        if normalized_query in block_text:
            matches.append(block["bbox"])
    
    # Method 2: Fuzzy matching for individual blocks
    for block in all_text_blocks:
        score = fuzzy_match_score(block["text"], query)
        if score >= fuzzy_threshold:
            matches.append(block["bbox"])
    
    # Method 3: Word-by-word fuzzy search across blocks
    for i in range(len(all_text_blocks)):
        current_blocks = [all_text_blocks[i]]
        current_text = normalize_text(all_text_blocks[i]["text"])
        current_words = current_text.split()
        
        # Try to match query words starting from this block
        matched_words = 0
        j = i + 1
        
        while matched_words < len(query_words) and j < len(all_text_blocks):
            # Check if next block is close enough (same line or nearby)
            y_diff = abs(all_text_blocks[j]["bbox"][1] - all_text_blocks[i]["bbox"][1])
            if y_diff > max_line_gap:
                break
            
            current_blocks.append(all_text_blocks[j])
            next_text = normalize_text(all_text_blocks[j]["text"])
            current_words.extend(next_text.split())
            j += 1
        
        # Check if current combination of blocks contains query
        combined_text = " ".join(current_words)
        if fuzzy_match_score(combined_text, normalized_query) >= fuzzy_threshold:
            # Calculate combined bounding box
            min_x = min(block["bbox"][0] for block in current_blocks)
            min_y = min(block["bbox"][1] for block in current_blocks)
            max_x = max(block["bbox"][2] for block in current_blocks)
            max_y = max(block["bbox"][3] for block in current_blocks)
            matches.append([min_x, min_y, max_x, max_y])
    
    # Method 4: Partial word matching for very unclear scans
    if not matches and len(query_words) == 1:
        single_word = query_words[0]
        for block in all_text_blocks:
            block_words = normalize_text(block["text"]).split()
            for word in block_words:
                if (len(word) >= 3 and len(single_word) >= 3 and 
                    fuzzy_match_score(word, single_word) >= 0.8):
                    matches.append(block["bbox"])
                    break
    
    # Remove duplicate matches (same area)
    unique_matches = []
    for match in matches:
        is_duplicate = False
        for existing in unique_matches:
            # Check if bounding boxes overlap significantly
            overlap_x = max(0, min(match[2], existing[2]) - max(match[0], existing[0]))
            overlap_y = max(0, min(match[3], existing[3]) - max(match[1], existing[1]))
            overlap_area = overlap_x * overlap_y
            
            match_area = (match[2] - match[0]) * (match[3] - match[1])
            existing_area = (existing[2] - existing[0]) * (existing[3] - existing[1])
            
            if overlap_area > 0.5 * min(match_area, existing_area):
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_matches.append(match)
    
    return unique_matches
def calculate_exact_dpi(page) -> Dict:
    """
    Calculates exact DPI values based on rendered pixel dimensions and physical size.
    Returns dictionary with DPI information.
    """
    try:
        # Get page size in points (1 point = 1/72 inch)
        width_pt = page.rect.width
        height_pt = page.rect.height

        # Convert to inches
        width_in = width_pt / 72
        height_in = height_pt / 72

        # Render the page at default DPI (72)
        pix = page.get_pixmap()
        pixel_w = pix.width
        pixel_h = pix.height

        # Calculate DPI
        dpi_x = pixel_w / width_in
        dpi_y = pixel_h / height_in
        avg_dpi = (dpi_x + dpi_y) / 2

        return {
            "dpi_x": round(dpi_x, 1),
            "dpi_y": round(dpi_y, 1),
            "average_dpi": round(avg_dpi, 1),
            "page_size_inches": {
                "width": round(width_in, 2),
                "height": round(height_in, 2)
            },
            "pixel_dimensions": {
                "width": pixel_w,
                "height": pixel_h
            }
        }
    except Exception as e:
        return {
            "error": str(e),
            "average_dpi": None
        }

def is_scanned_page(page) -> bool:
    """
    Determines if a page is scanned/image-based based on text content.
    """
    text = page.get_text().strip()
    return len(text) < 50  # Very little text indicates scanned page

def estimate_images_page_quality(page) -> Dict:
    """
    Calculates exact DPI for both text-based and image-based pages.
    """
    dpi_info = calculate_exact_dpi(page)
    text = page.get_text().strip()
    page_type = "image-based" if len(text) < 50 else "text-based"

    return {
        "page_type": page_type,
        "dpi_estimate": dpi_info.get("average_dpi"),
        "dpi_details": dpi_info
    }

def estimate_page_quality(page) -> Dict:
    """
    Returns quality information for a single PDF page with accurate DPI estimation.
    Text-based pages get fixed 250 DPI, scanned pages get calculated DPI.
    """
    if is_scanned_page(page):
        dpi_info = calculate_exact_dpi(page)
        return {
            "page_type": "image-based",
            "dpi_estimate": dpi_info.get("average_dpi", 150),
            "dpi_details": dpi_info
        }
    else:
        return {
            "page_type": "text-based",
            "dpi_estimate": 250,  # Fixed value for text-based PDFs
            "dpi_details": {
                "note": "Text-based PDF assigned fixed 250 DPI",
                "average_dpi": 250
            }
        }



@app.post("/pdf-dpi-check")
async def check_pdf_dpi(file: UploadFile):
    """
    Check PDF DPI with accurate estimation for scanned pages and fixed 250 DPI for text-based pages.
    Returns per-page analysis and overall statistics.
    """
    try:
        contents = await file.read()
        input_stream = io.BytesIO(contents)
        doc = fitz.open(stream=input_stream, filetype="pdf")

        results = []
        dpi_values = []
        page_types = {"text-based": 0, "image-based": 0}

        for page_num, page in enumerate(doc):
            page_info = estimate_page_quality(page)
            page_info["page_number"] = page_num + 1
            results.append(page_info)
            
            # Collect statistics
            dpi_values.append(page_info["dpi_estimate"])
            page_types[page_info["page_type"]] += 1

        doc.close()

        # Calculate overall statistics
        valid_dpis = [d for d in dpi_values if d is not None]
        avg_dpi = round(sum(valid_dpis) / len(valid_dpis), 1) if valid_dpis else None

        return JSONResponse(content={
            "success": True,
            "page_count": len(results),
            "average_dpi": avg_dpi,
            "page_type_distribution": page_types,
            "pages": results
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.post("/diagnose-text-extraction")
async def diagnose_text_extraction(
    file: UploadFile,
    page: int = Form(1)
):
    """
    Diagnostic endpoint to see what text is being extracted from a page.
    Useful for debugging scanned documents.
    """
    try:
        pdf_bytes = await file.read()
        stream = io.BytesIO(pdf_bytes)
        doc = fitz.open(stream=stream, filetype="pdf")
        
        if page < 1 or page > len(doc):
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid page number. PDF has {len(doc)} pages."}
            )
        
        selected_page = doc[page - 1]
        
        # Extract using all methods
        all_text_blocks = extract_text_with_multiple_methods(selected_page)
        
        # Organize by extraction method
        results = {
            "page": page,
            "total_blocks_found": len(all_text_blocks),
            "extraction_methods": {}
        }
        
        for method in ["words", "dict", "blocks"]:
            method_blocks = [block for block in all_text_blocks if block["method"] == method]
            results["extraction_methods"][method] = {
                "count": len(method_blocks),
                "sample_texts": [block["text"][:100] + "..." if len(block["text"]) > 100 else block["text"] 
                               for block in method_blocks[:5]]  # First 5 samples
            }
        
        # Show raw text extraction
        try:
            raw_text = selected_page.get_text()
            results["raw_text_sample"] = raw_text[:500] + "..." if len(raw_text) > 500 else raw_text
        except:
            results["raw_text_sample"] = "Failed to extract raw text"
        
        doc.close()
        return results
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )



@app.get("/quality-settings")
async def get_quality_settings():
    """
    Get available quality settings and their descriptions.
    """
    return {
        "quality_levels": {
            "standard": {
                "zoom_factor": 2.0,
                "enhancements": False,
                "max_dimensions": "800x600",
                "description": "Basic quality, smaller file size",
                "recommended_for": "Quick previews, low bandwidth"
            },
            "high": {
                "zoom_factor": 4.0,
                "enhancements": True,
                "max_dimensions": "1200x800",
                "description": "High quality with image enhancements",
                "recommended_for": "Google Sheets, general use"
            },
            "ultra": {
                "zoom_factor": 6.0,
                "enhancements": True,
                "max_dimensions": "1600x1200",
                "description": "Ultra high quality for premium use cases",
                "recommended_for": "Print quality, detailed analysis"
            }
        },
        "enhancements_applied": [
            "Sharpness enhancement (+50%)",
            "Contrast enhancement (+20%)",
            "Brightness adjustment (+10%)",
            "Unsharp mask filter for text clarity",
            "LANCZOS resampling for resizing"
        ]
    }



@app.post("/get-row-by-query")
async def get_row_by_query(
    file: UploadFile,
    query: str = Form(...),
    page: Optional[int] = Form(None),
    padding: int = Form(5)
):
    """
    Find the entire row containing the specified query text.
    Returns the coordinates of the whole row where the text appears.
    
    Args:
        query: Text to search for in the PDF
        page: Optional specific page to search (searches all pages if not provided)
        padding: Additional pixels to add around the row coordinates
    """
    try:
        pdf_bytes = await file.read()
        stream = io.BytesIO(pdf_bytes)
        doc = fitz.open(stream=stream, filetype="pdf")
        
        results = []
        
        # Determine which pages to search
        if page:
            if page < 1 or page > len(doc):
                doc.close()
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Invalid page number. PDF has {len(doc)} pages."}
                )
            pages_to_search = [page - 1]
        else:
            pages_to_search = range(len(doc))
        
        for page_num in pages_to_search:
            current_page = doc[page_num]
            
            # First try exact search
            found_rects = current_page.search_for(query)
            
            # If no exact matches, try fuzzy search
            if not found_rects:
                found_rects = enhanced_search_multiline_text(current_page, query)
                if not found_rects:
                    continue
            
            # For each found rectangle, find the full row
            for rect in found_rects:
                if isinstance(rect, fitz.Rect):
                    bbox = [rect.x0, rect.y0, rect.x1, rect.y1]
                else:  # from enhanced search
                    bbox = rect
                
                # Get all words on the page
                words = current_page.get_text("words")
                
                # Find words in the same row (similar y-coordinates)
                row_words = []
                current_y = None
                
                for word in words:
                    word_bbox = word[:4]
                    word_y_center = (word_bbox[1] + word_bbox[3]) / 2
                    
                    # Check if word is in the same row (within a small y-range)
                    if current_y is None:
                        # First word in row - use the found word's y position
                        current_y = (bbox[1] + bbox[3]) / 2
                        y_threshold = max(10, (bbox[3] - bbox[1]) * 1.5)
                    
                    if abs(word_y_center - current_y) <= y_threshold:
                        row_words.append(word_bbox)
                
                if row_words:
                    # Calculate full row bounding box
                    min_x = min(w[0] for w in row_words)
                    min_y = min(w[1] for w in row_words)
                    max_x = max(w[2] for w in row_words)
                    max_y = max(w[3] for w in row_words)
                    
                    # Apply padding
                    page_rect = current_page.rect
                    padded_bbox = [
                        max(min_x - padding, 0),
                        max(min_y - padding, 0),
                        min(max_x + padding, page_rect.width),
                        min(max_y + padding, page_rect.height)
                    ]
                    
                    # Get the text of the full row
                    row_text = current_page.get_textbox(fitz.Rect(*padded_bbox))
                    
                    results.append({
                        "page": page_num + 1,
                        "query": query,
                        "found_at": {
                            "x0": bbox[0],
                            "y0": bbox[1],
                            "x1": bbox[2],
                            "y1": bbox[3]
                        },
                        "full_row": {
                            "x0": padded_bbox[0],
                            "y0": padded_bbox[1],
                            "x1": padded_bbox[2],
                            "y1": padded_bbox[3],
                            "text": row_text.strip()
                        }
                    })
        
        doc.close()
        
        if not results:
            return JSONResponse(
                status_code=404,
                content={"success": False, "message": f"Text '{query}' not found in document"}
            )
        
        return {
            "success": True,
            "total_matches": len(results),
            "matches": results,
            "usage_note": "Use the full_row coordinates with /get-snippet-image endpoint to generate row images"
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
    
@app.post("/image-dpi")
async def calculate_image_dpi(file: UploadFile = File(...)):
    """
    Calculate DPI of an image assuming it's a scanned A4 document.
    No DPI metadata required.
    """
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        width_px, height_px = img.size

        # Assumed A4 size in inches
        a4_width_in = 8.27
        a4_height_in = 11.69

        dpi_x = width_px / a4_width_in
        dpi_y = height_px / a4_height_in
        avg_dpi = round((dpi_x + dpi_y) / 2, 1)

        return {
            "dpi_x": round(dpi_x, 1),
            "dpi_y": round(dpi_y, 1),
            "average_dpi": avg_dpi,
            "pixel_dimensions": {"width": width_px, "height": height_px},
            "assumed_physical_size": "A4 (8.27 x 11.69 inches)",
            "note": "DPI calculated from image pixels assuming A4 size"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
