import json
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

# Configuration Settings
JSON_PATH = "moudawana_articles.json"
DB_PATH = "./chroma_db"
MODEL_NAME = "intfloat/multilingual-e5-large"

def load_documents():
    """
    Loads parsed JSON articles and converts them into LangChain Document objects.
    """
    if not os.path.exists(JSON_PATH):
        print("❌ JSON file not found!")
        return []
    
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    print("🔄 Preparing documents...")
    
    for item in data:
        hierarchy = item.get('hierarchy', {})
        
        # Extra protection: Ensure values are strings and not None
        book = hierarchy.get('book') or ""
        section = hierarchy.get('section') or ""
        chapter = hierarchy.get('chapter') or ""
        
        # Combine hierarchy and content to enrich the vector representation
        context_str = f"{book} - {section} - {chapter}"
        full_text = f"{context_str} {item['title']}: {item['content']}"
        
        metadata = {
            "id": str(item['id']), # Convert ID to string to prevent ChromaDB metadata issues
            "title": item['title'],
            "book": book,
            "section": section,
            "chapter": chapter,
            "source": "Moudawana (Family Code)"
        }
        
        doc = Document(page_content=full_text, metadata=metadata)
        documents.append(doc)
    
    return documents

def create_vector_db():
    """
    Embeds the documents and stores them in a local Chroma vector database.
    """
    docs = load_documents()
    if not docs: 
        return

    # Automatically delete the old database to prevent data duplication/conflicts
    if os.path.exists(DB_PATH):
        import shutil
        print("🧹 Cleaning up old database...")
        shutil.rmtree(DB_PATH)

    print(f"⚙️ Loading embedding model: {MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )

    print("🚀 Building vector database...")
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print(f"✅ Successfully saved to: {DB_PATH}")
    
    # --- Safe Testing ---
    print("\n🔍 Quick Test:")
    # Keeping the query in Moroccan Arabic/Darija to test real-world performance
    query = "شنو كيوقع يلا مات الزوج؟" 
    results = vector_db.similarity_search(query, k=3)
    
    for i, res in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"📄 Article: {res.metadata.get('title', 'N/A')}")
        
        # Fix: Using .get() to prevent KeyError if metadata is missing
        sec = res.metadata.get('section', 'No Section')
        chap = res.metadata.get('chapter', 'No Chapter')
        
        print(f"📂 Context: {sec} | {chap}")
        print(f"📝 Text: {res.page_content[:100]}...")

if __name__ == "__main__":
    create_vector_db()