import os
import warnings
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder

# 🚀 Change here: Importing Groq instead of Google
from langchain_groq import ChatGroq

warnings.filterwarnings('ignore')

load_dotenv()

# Verify Groq API Key
if not os.getenv("GROQ_API_KEY"):
    print("❌ Error: GROQ_API_KEY not found in the .env file")
    exit()

print("\n⏳ Waking up the intelligent engine (Embeddings + VectorDB)...")

try:
    embedding_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
    
    print("⏳ Loading the Arabic Cross-Encoder model (ARA-Reranker)...")
    reranker = CrossEncoder("Omartificial-Intelligence-Space/ARA-Reranker-V1", device='cpu')
    
    # 🚀 Change here: Setting up Groq (Llama 3)
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=0.3
    )
    
except Exception as e:
    print(f"❌ Error during initialization: {e}")
    exit()

print("\n✅ System ready with Llama 3 (Groq)! The Legal Advisor is at your service.")
print("   (Type 'exit', 'quit', or 'q' to close)")
print("-" * 60)

# The prompt is translated to English for readability, but instructs the LLM to answer in Arabic
prompt = PromptTemplate.from_template("""
You are an expert Moroccan legal advisor specializing in the Family Code (Moudawana).
Answer the following question accurately in Arabic, relying *strictly* on the provided legal texts.

Legal Context:
{context}

User Question: {question}

Answer (mention article numbers, and if the answer is not found, state clearly 'لا توجد معلومات كافية'):
""")

while True:
    query = input("\n❓ Your question: ")
    if query.strip().lower() in ['خروج', 'exit', 'quit', 'q']:
        print("👋 Goodbye!")
        break
    if not query.strip(): continue

    print("🔍 Searching and evaluating results...")
    
    try:
        # Fetching 7 initial results to keep CPU processing fast
        initial_docs = vector_db.similarity_search(query, k=7)
        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = reranker.predict(pairs)
        
        for idx, doc in enumerate(initial_docs):
            doc.metadata['rerank_score'] = float(scores[idx])
            
        ranked_docs = sorted(initial_docs, key=lambda x: x.metadata['rerank_score'], reverse=True)[:4]
        
        context_text = "\n\n".join([f"{doc.metadata.get('title', '')}: {doc.page_content}" for doc in ranked_docs])
        
        chain = prompt | llm
        response = chain.invoke({"context": context_text, "question": query})
        
        print("\n⚖️ Answer:")
        print("=" * 60)
        print(response.content)
        print("=" * 60)
        
        print("\n📚 Relied Sources:")
        for i, doc in enumerate(ranked_docs):
            print(f"  {i+1}. {doc.metadata.get('title', 'Article')} (Score: {doc.metadata['rerank_score']:.2f})")
            
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")