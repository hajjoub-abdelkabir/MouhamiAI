import pytesseract
from pdf2image import convert_from_path
import re
import json
import os
from tqdm import tqdm

PDF_PATH = "moudawana.pdf"
OUTPUT_PATH = "moudawana_articles.json"

def clean_text(text):
    """
    Cleans the extracted text by removing newlines and extra spaces.
    """
    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_hierarchy(text, current_context):
    """
    Searches for new hierarchical headings (Book, Section, Chapter) within the text
    and updates the current context accordingly.
    """
    # Optimized Regex to better capture headings (e.g., Section Two, Chapter One...)
    # We search for the keyword + 3-5 subsequent words.
    
    # 1. Search for Book (الكتاب)
    book_match = re.search(r"(الكتاب\s+[\u0600-\u06FF]+\s?[\u0600-\u06FF]*)", text)
    if book_match:
        current_context['book'] = book_match.group(1).strip()
        # Reset Section and Chapter when a new Book starts
        current_context['section'] = None
        current_context['chapter'] = None

    # 2. Search for Section (القسم)
    section_match = re.search(r"(القسم\s+[\u0600-\u06FF]+\s?[\u0600-\u06FF]*)", text)
    if section_match:
        current_context['section'] = section_match.group(1).strip()
        # Reset Chapter when a new Section starts
        current_context['chapter'] = None

    # 3. Search for Chapter (الباب)
    chapter_match = re.search(r"(الباب\s+[\u0600-\u06FF]+\s?[\u0600-\u06FF]*)", text)
    if chapter_match:
        current_context['chapter'] = chapter_match.group(1).strip()
    
    return current_context

def process_pdf_to_json(pdf_path, output_path):
    print(f"🚀 Processing file and extracting hierarchy: {pdf_path}")
    
    # --- OCR Processing ---
    try:
        # Note: If images were converted previously, this step could be skipped to save time.
        # However, we proceed assuming a fresh run is needed.
        print("📸 Converting PDF pages to images...")
        images = convert_from_path(pdf_path)
    except Exception as e:
        print(f"❌ Error during PDF conversion: {e}")
        return

    full_text = ""
    print(f"📝 Extracting text via OCR...")
    for image in tqdm(images, desc="OCR Progress"):
        full_text += pytesseract.image_to_string(image, lang='ara') + " "

    print("🧹 Analyzing structural hierarchy...")
    full_text = clean_text(full_text)

    # Split text based on Article markers (المادة)
    pattern = r"(المادة \d+)"
    splits = re.split(pattern, full_text)

    articles = []
    
    # Initialize the context state
    context = {
        "book": None,
        "section": None,
        "chapter": None
    }

    # The first segment (splits[0]) usually contains the introduction.
    # We scan it for initial hierarchical headings.
    context = extract_hierarchy(splits[0], context)

    # Iterate through the split segments.
    # Step by 2 because re.split with a capture group returns [pre-match, match, post-match, ...]
    for i in range(1, len(splits) - 1, 2):
        article_title = splits[i].strip()       # e.g., "المادة X"
        article_content = splits[i+1].strip()   # Content of "المادة X"
        
        # --- Smart Context Linking ---
        # Before saving the current article, we must recognize that headings (Chapters/Sections)
        # for the *next* article are often found at the end of the *current* article's text, 
        # just before the next article title.
        # Therefore, we save the current article using the existing context first.
        
        current_article_data = {
            "id": article_title.replace("المادة", "").strip(),
            "title": article_title,
            "content": article_content,
            "source": "مدونة الأسرة",
            # Attach the hierarchical context
            "hierarchy": {
                "book": context['book'],
                "section": context['section'],
                "chapter": context['chapter']
            }
        }
        
        # Filter out abnormally short content (likely parsing artifacts)
        if len(article_content) > 10:
            articles.append(current_article_data)

        # --- Update Context for the Next Iteration ---
        # Now we search within this article's content (article_content) to see if a new heading
        # is introduced, setting the stage for the subsequent article.
        context = extract_hierarchy(article_content, context)

    # Save the structured data to a JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ Success! Extracted {len(articles)} articles with hierarchy.")
    
    # Print a sample to verify extraction
    if len(articles) > 74: 
        print("\n--- Sample: Article 74 ---")
        # Attempt to find Article 74 specifically
        art_74 = next((item for item in articles if item["id"] == "74"), None)
        if art_74:
            print(json.dumps(art_74, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(articles[50], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    if os.path.exists(PDF_PATH):
        process_pdf_to_json(PDF_PATH, OUTPUT_PATH)
    else:
        print(f"❌ File not found: {PDF_PATH}")