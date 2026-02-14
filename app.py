import streamlit as st
import os
import warnings
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder
from langchain_groq import ChatGroq

warnings.filterwarnings('ignore')

# 1. Page Configuration
st.set_page_config(page_title="المستشار القانوني - مدونة الأسرة", page_icon="⚖️", layout="wide")

def inject_custom_css():
    st.markdown("""
    <style>
    /* 1. Import professional Arabic font */
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');

    /* 2. Set the entire application to RTL (Right-to-Left) */
    .stApp {
        direction: rtl;
    }

    /* Chat interface formatting */
    .stChatMessage {
        direction: rtl;
        text-align: right;
    }
    
    /* 3. Apply font to core elements 
       (span is excluded to preserve Streamlit material icons) */
    html, body, p, div, h1, h2, h3, h4, h5, h6, label, input, textarea, li {
        font-family: 'Cairo', sans-serif !important;
        text-align: right !important;
    }

    /* Protect icons so they render correctly and not as text */
    span[class*="material"] {
        font-family: 'Material Icons', 'Material Symbols Rounded', sans-serif !important;
    }

    /* 4. Fix spacing issues in bullet points for RTL */
    ul, ol {
        direction: rtl;
        text-align: right;
        padding-right: 35px !important; /* Push bullets to the right */
        padding-left: 0 !important;     /* Remove default left padding */
        margin-top: 10px;
    }
    li {
        direction: rtl;
        text-align: right;
        margin-bottom: 8px; /* Spacing between bullets for better readability */
    }

    /* 5. Page background formatting */
    .stApp {
        background-color: #F4F7F6;
    }

    /* 6. Sidebar formatting */
    [data-testid="stSidebar"] {
        background-color: #1A252C;
        color: #FFFFFF;
        padding-top: 2rem;
        border-left: 1px solid #e6e6e6;
        border-right: none;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] p {
        color: #F8F9FA !important;
    }

    /* 7. Main headers */
    h1 {
        color: #1B3B5C;
        font-weight: 700 !important;
        text-align: center !important;
        padding-bottom: 20px;
    }

    /* 8. Chat input box formatting */
    .stTextInput>div>div>input {
        border: 2px solid #1B3B5C;
        border-radius: 10px;
        padding: 15px;
        font-size: 16px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.05);
        transition: border-color 0.3s ease;
        direction: rtl;
    }
    .stTextInput>div>div>input:focus {
        border-color: #D4AF37;
        box-shadow: 0px 4px 10px rgba(212, 175, 55, 0.2);
    }

    /* 9. Answer box (Official document style) */
    div[data-testid="stNotification"] {
        background-color: #FFFFFF;
        border-right: 5px solid #D4AF37 !important;
        border-left: none !important;
        color: #2C3E50;
        box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.08);
        border-radius: 8px;
        padding: 20px;
        font-size: 17px;
        line-height: 1.8;
    }

    /* 10. Sources section (Expander) */
    .streamlit-expanderHeader {
        background-color: #E8ECEF;
        border-radius: 5px;
        color: #1B3B5C;
        font-weight: 600;
        direction: rtl;
    }
    
    /* 11. Quick suggestion buttons */
    .stButton>button {
        border-radius: 20px;
        border: 1px solid #1B3B5C;
        color: #1B3B5C;
        background-color: transparent;
        transition: all 0.3s ease;
        padding: 5px 15px;
    }
    .stButton>button:hover {
        background-color: #1B3B5C;
        color: #FFFFFF;
        border-color: #1B3B5C;
    }

    /* Hide default Streamlit menus for a cleaner UI */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Inject the custom CSS
inject_custom_css()
# ------------------------

# Load environment variables
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    st.error("⚠️ المرجو التأكد من وجود مفتاح GROQ_API_KEY في ملف .env")
    st.stop()

# 2. Load the System (Cached to run only once)
@st.cache_resource(show_spinner="جاري تحميل المحرك الذكي (Embeddings & Reranker)...")
def load_rag_system():
    # a. Embeddings + VectorDB
    embedding_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
    
    # b. Reranker (Arabic Cross-Encoder model)
    reranker = CrossEncoder("Omartificial-Intelligence-Space/ARA-Reranker-V1", device='cpu')
    
    # c. LLM (Llama 3 via Groq for high-speed inference)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
    
    return vector_db, reranker, llm

try:
    vector_db, reranker, llm = load_rag_system()
except Exception as e:
    st.error(f"❌ حدث خطأ أثناء تحميل النظام: {e}")
    st.stop()

# 3. UI Design Setup
st.title("⚖️ MouhamiAI: مستشارك في مدونة الأسرة")
st.markdown("---")

with st.sidebar:
    st.header("معلومات النظام ⚙️")
    st.success("✅ متصل بقاعدة بيانات مدونة الأسرة")
    st.success("✅ محرك البحث: Multilingual-E5")
    st.success("✅ إعادة الترتيب: ARA-Reranker-V1")
    st.success("✅ التوليد: Llama-3.3 (Groq)")
    st.warning("تنبيه: هذا الذكاء الاصطناعي للمساعدة القانونية ولا يعوض المحامي.")

# --- 4. Session State Setup ---
if "messages" not in st.session_state:
    st.session_state.messages = [] # This is the conversation memory

# --- Display previous conversation ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input Interface ---
if query := st.chat_input("اطرح سؤالك القانوني هنا (مثال: شنو شروط الزواج؟)..."):
    
    # 1. Display user query immediately and save to memory
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.chat_message("assistant"):
        # ==========================================
        # 🛡️ Step 1: AI Guardrail / Intent Router
        # ==========================================
        with st.spinner("🛡️ جاري التحقق من تخصص السؤال..."):
            try:
                guardrail_prompt = PromptTemplate.from_template("""
                You are a security guard (Guardrail) for a Moroccan legal consultation system (Moudawana).
                Your sole task is to classify the user's intent.
                
                Answer with exactly one word:
                - "LEGAL": If the question is about law, marriage, divorce, custody, inheritance, courts, or administrative legal inquiries.
                - "OTHER": If the question is general chat, sports, cooking, tech, politics, or anything unrelated to law.
                
                User Question: {question}
                Classification (one word):
                """)
                
                # Quickly classify the intent
                intent = llm.invoke(guardrail_prompt.format(question=query)).content.strip().upper()
                
            except Exception as e:
                intent = "LEGAL" # Fallback to LEGAL if an error occurs to avoid blocking the user
        
        # ==========================================
        # 🛑 Response based on classification
        # ==========================================
        if "OTHER" in intent:
            # If out of scope, apologize and stop processing
            refusal_msg = "عذراً، بصفتي **محامي AI** ⚖️، تخصصي يقتصر حصرياً على تقديم الاستشارات في **مدونة الأسرة المغربية**. لا يمكنني الإجابة على استفسارات خارج النطاق القانوني.\n\n المرجو طرح سؤال يخص الزواج، الطلاق، أو حقوق الأسرة."
            st.markdown(refusal_msg)
            st.session_state.messages.append({"role": "assistant", "content": refusal_msg})
            
        else:
            # If legal, proceed with the standard RAG pipeline
            with st.spinner("🔍 جاري البحث والتحليل القانوني..."):
                try:
                    # 2. Query Reformulation based on context
                    search_query = query
                    if len(st.session_state.messages) > 1:
                        history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:-1]])
                        rephrase_prompt = PromptTemplate.from_template("""
                        Based on the conversation history and the new user question, rephrase the question to be a standalone, clear Arabic question for searching a legal database.
                        Conversation History:
                        {history}
                        New Question: {question}
                        Standalone Question:
                        """)
                        search_query = llm.invoke(rephrase_prompt.format(history=history, question=query)).content.strip()

                    # 3. Search and Reranking
                    initial_docs = vector_db.similarity_search(search_query, k=7)
                    pairs = [[search_query, doc.page_content] for doc in initial_docs]
                    scores = reranker.predict(pairs)
                    
                    for idx, doc in enumerate(initial_docs):
                        doc.metadata['rerank_score'] = float(scores[idx])
                        
                    ranked_docs = sorted(initial_docs, key=lambda x: x.metadata['rerank_score'], reverse=True)[:4]
                    context_text = "\n\n".join([f"{doc.metadata.get('title', '')}: {doc.page_content}" for doc in ranked_docs])
                    
                    # 4. Generate the final answer
                    final_prompt = PromptTemplate.from_template("""
                    You are an expert Moroccan legal advisor specializing in the Family Code (Moudawana).
                    Available Legal Context:
                    {context}
                    User Question: {question}
                    
                    Instructions:
                    1. Answer in Arabic based strictly on the provided context.
                    2. Explain the article in a simple, organized manner (use bullet points).
                    3. Cite the relied-upon article numbers.
                    Answer:
                    """)
                    
                    response = (final_prompt | llm).invoke({"context": context_text, "question": query})
                    
                    # 5. Display and save the answer
                    st.markdown(response.content)
                    st.session_state.messages.append({"role": "assistant", "content": response.content})
                    
                    # 6. Display sources
                    with st.expander("📚 المصادر القانونية المعتمدة"):
                        for i, doc in enumerate(ranked_docs):
                            st.markdown(f"**{i+1}. {doc.metadata.get('title', 'مادة')}** *(دقة التوافق: {doc.metadata['rerank_score']:.2f})*")
                            st.caption(doc.page_content)
                            st.divider()
                            
                except Exception as e:
                    st.error(f"❌ حدث خطأ أثناء المعالجة: {e}")