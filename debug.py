import fitz  # PyMuPDF

def inspect_pdf():
    """
    A simple debugging function to test PDF text extraction using PyMuPDF.
    Useful for checking encoding and formatting issues in the source document.
    """
    doc = fitz.open("moudawana.pdf")
    text = ""
    
    # Read only the first few pages to save time and resources
    for i in range(min(5, len(doc))): 
        text += doc[i].get_text()
    
    print("--- 🔍 Start of Extracted Text ---")
    print(text[:1000])  # Print the first 1000 characters for inspection
    print("\n--- 🏁 End of Preview ---")

if __name__ == "__main__":
    inspect_pdf()