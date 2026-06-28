# test_extraction.py
# Quick manual test for the PDF parser. Safe to delete after testing.

from extraction.pdf_parser import extract_text_from_pdf

# Replace the filename below with a real file from data/raw/contracts/
TEST_FILE = r"data/raw/contracts\ACCELERATEDTECHNOLOGIESHOLDINGCORP_04_24_2003-EX-10.13-JOINT VENTURE AGREEMENT.PDF"

result = extract_text_from_pdf(
    file_path=TEST_FILE,
    use_ocr_fallback=False   # keep this False for now — no OpenAI calls yet
)

print("\n========== SUMMARY ==========")
print(result.summary())

print("\n========== FIRST PAGE PREVIEW (first 300 characters) ==========")
if result.pages:
    print(result.pages[0].text[:300])
else:
    print("No pages were extracted.")

print("\n========== PER-PAGE METHOD BREAKDOWN ==========")
for page in result.pages:
    print(f"Page {page.page_number}: method={page.method}, char_count={page.char_count}")