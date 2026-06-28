# test_ocr.py
# Quick manual test for the OCR Vision module. Safe to delete after testing.

from dotenv import load_dotenv
load_dotenv()  # must run before any module that reads OPENAI_API_KEY

import fitz  # we'll use PyMuPDF to create a simple test image

from extraction.ocr import extract_page_with_vision

# Create a tiny blank PDF page with some text, then render it as an image.
# This simulates what pdf_parser.py would hand to ocr.py for a real scanned page.
doc = fitz.open()  # creates a new, empty PDF in memory
page = doc.new_page()
page.insert_text((72, 72), "This is a test page for OCR verification.")

pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
image_bytes = pixmap.tobytes("png")

doc.close()

# Now send it through the real OCR function
result = extract_page_with_vision(image_bytes=image_bytes, page_number=1)

print("\n========== OCR RESULT ==========")
print(f"Method: {result.method}")
print(f"Char count: {result.char_count}")
print(f"Text: {result.text}")
print(f"Notes: {result.extraction_notes}")