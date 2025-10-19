import streamlit as st
from transformers import pipeline
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
import pytesseract
import tempfile
import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load models
@st.cache_resource(show_spinner=False)
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource(show_spinner=False)
def load_qa_pipeline():
    return pipeline("question-answering", model="deepset/roberta-base-squad2")

summarizer = load_summarizer()
embedder = load_embedder()
qa_pipeline = load_qa_pipeline()

MAX_INPUT_CHARS = 1000

def truncate_text(text, max_length=MAX_INPUT_CHARS):
    if len(text) > max_length:
        return text[:max_length]
    return text

def extract_text_from_pdf(file_path):
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception:
        return ""

def summarize_large_pdf(file_path, pages_per_batch=10, max_batches=10):
    reader = PdfReader(file_path)
    summaries = []
    num_pages = len(reader.pages)
    for i in range(0, min(num_pages, pages_per_batch * max_batches), pages_per_batch):
        text = ""
        for j in range(i, min(i + pages_per_batch, num_pages)):
            t = reader.pages[j].extract_text()
            if t:
                text += t
        text = truncate_text(text)
        if text.strip():
            chunk_summary = summarizer(text, max_length=250, min_length=60, do_sample=False)
            summaries.append(f"Pages {i+1}-{min(i+pages_per_batch, num_pages)}: {chunk_summary[0]['summary_text']}")
    return "\n\n".join(summaries)

def extract_text_from_image(file_path):
    try:
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)
    except Exception:
        return ""

def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception:
        return ""

def extract_text(file_path, file_ext):
    if file_ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif file_ext in [".jpg", ".jpeg", ".png"]:
        return extract_text_from_image(file_path)
    elif file_ext == ".docx":
        return extract_text_from_docx(file_path)
    else:
        return ""

def split_text_into_chunks(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def embed_chunks(chunks):
    embeddings = embedder.encode(chunks, convert_to_tensor=True)
    return embeddings

def find_most_similar_chunk(question, chunk_embeddings, chunks):
    question_vec = embedder.encode([question], convert_to_tensor=True)
    cosine_scores = cosine_similarity(question_vec.cpu().numpy(), chunk_embeddings.cpu().numpy())[0]
    top_idx = cosine_scores.argmax()
    return chunks[top_idx]

def answer_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Streamlit UI
st.set_page_config(page_title="AI Note Summarizer + Chatbot", page_icon="🧠", layout="centered")
st.title("🧠 AI Note Summarizer & Chatbot")

uploaded_file = st.file_uploader("Upload your file (PDF, Image, Word)", type=["pdf", "jpg", "jpeg", "png", "docx"])

# State for embeddings and chunks for QA
if 'chunks' not in st.session_state:
    st.session_state['chunks'] = []
if 'chunk_embeddings' not in st.session_state:
    st.session_state['chunk_embeddings'] = None

if uploaded_file:
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    st.subheader("File Information")
    st.write(f"Filename: {uploaded_file.name}")
    st.write(f"File Type: {ext}")

    if ext == ".pdf":
        reader = PdfReader(temp_path)
        num_pages = len(reader.pages)
        st.write(f"Number of pages: {num_pages}")

        if num_pages > 20:
            st.warning("Large PDF detected! Summarization in batches.")

            pages_per_batch = st.number_input("Pages per summary batch:", min_value=5, max_value=50, value=10)
            max_batches = st.number_input("Max batches (to process):", min_value=1, max_value=100, value=10)
            if st.button("Summarize Large PDF"):
                with st.spinner("Chunking and summarizing PDF..."):
                    summary = summarize_large_pdf(temp_path, pages_per_batch=int(pages_per_batch), max_batches=int(max_batches))
                    st.subheader("📘 Chunked Summary")
                    st.write(summary)
        else:
            extracted_text = extract_text_from_pdf(temp_path)
            st.subheader("Extracted Text Preview")
            st.text_area("", extracted_text[:1500] + "...", height=200)

            if st.button("Summarize PDF"):
                with st.spinner("Summarizing..."):
                    out = summarizer(truncate_text(extracted_text), max_length=250, min_length=60, do_sample=False)
                    st.subheader("📘 Summary")
                    st.write(out[0]['summary_text'])

            # Prepare QA embeddings
            if st.button("Prepare Chatbot Q&A from PDF"):
                with st.spinner("Generating embeddings for chatbot..."):
                    st.session_state['chunks'] = split_text_into_chunks(extracted_text)
                    st.session_state['chunk_embeddings'] = embed_chunks(st.session_state['chunks'])
                st.success("Embeddings ready! You can now ask questions below.")

    elif ext in [".jpg", ".jpeg", ".png"]:
        extracted_text = extract_text_from_image(temp_path)
        st.subheader("Extracted Text Preview (OCR)")
        st.text_area("", extracted_text[:1500] + "...", height=200)

        if st.button("Summarize Image Text"):
            with st.spinner("Summarizing..."):
                out = summarizer(truncate_text(extracted_text), max_length=250, min_length=60, do_sample=False)
                st.subheader("📘 Summary")
                st.write(out[0]['summary_text'])

    elif ext == ".docx":
        extracted_text = extract_text_from_docx(temp_path)
        st.subheader("Extracted Text Preview")
        st.text_area("", extracted_text[:1500] + "...", height=200)

        if st.button("Summarize Word Doc"):
            with st.spinner("Summarizing..."):
                out = summarizer(truncate_text(extracted_text), max_length=250, min_length=60, do_sample=False)
                st.subheader("📘 Summary")
                st.write(out[0]['summary_text'])

        if st.button("Prepare Chatbot Q&A from Word Doc"):
            with st.spinner("Generating embeddings for chatbot..."):
                st.session_state['chunks'] = split_text_into_chunks(extracted_text)
                st.session_state['chunk_embeddings'] = embed_chunks(st.session_state['chunks'])
            st.success("Embeddings ready! You can now ask questions below.")

    else:
        st.warning("Unsupported file format or extraction error.")

