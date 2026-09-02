import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
from google import genai


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Chatbot")
st.write("Upload a PDF and ask questions from its content.")


# ============================================================
# GEMMA CLIENT
# ============================================================

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


embedding_model = load_embedding_model()


# ============================================================
# PDF PROCESSING
# ============================================================

def process_pdf(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            text += page_text + "\n"


    # ========================================================
    # SECTION-BASED CHUNKING
    # ========================================================

    lines = text.splitlines()

    chunks = []

    current_section = []


    for line in lines:

        line = line.strip()

        if not line:
            continue


        # Detect headings like:
        # 1. Introduction
        # 2. Artificial Intelligence
        # 3. Machine Learning

        if re.match(r'^\d+\.\s+', line):

            if current_section:

                section_text = " ".join(
                    current_section
                ).strip()

                if section_text:

                    chunks.append(
                        section_text
                    )


            current_section = [line]

        else:

            current_section.append(line)


    # Last section

    if current_section:

        section_text = " ".join(
            current_section
        ).strip()

        if section_text:

            chunks.append(
                section_text
            )


    return chunks


# ============================================================
# CREATE FAISS INDEX
# ============================================================

def create_faiss_index(chunks):

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    ).astype("float32")


    dimension = embeddings.shape[1]


    index = faiss.IndexFlatL2(
        dimension
    )


    index.add(embeddings)


    return index


# ============================================================
# ASK QUESTION
# ============================================================

def answer_question(
    question,
    chunks,
    index,
    k=3
):

    k = min(
        k,
        len(chunks)
    )


    # ========================================================
    # QUESTION EMBEDDING
    # ========================================================

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")


    # ========================================================
    # FAISS SEARCH
    # ========================================================

    distances, indices = index.search(
        question_embedding,
        k
    )


    # ========================================================
    # THRESHOLD
    # ========================================================

    THRESHOLD = 1.5

    best_distance = distances[0][0]


    # If question is not relevant

    if best_distance > THRESHOLD:

        return (
            "Sorry, I don't have enough information "
            "to answer that.",
            [],
            []
        )


    # ========================================================
    # RETRIEVE RELEVANT CHUNKS
    # ========================================================

    retrieved_chunks = []

    valid_distances = []


    for idx, distance in zip(
        indices[0],
        distances[0]
    ):

        if (
            0 <= idx < len(chunks)
            and distance <= THRESHOLD
        ):

            retrieved_chunks.append(
                chunks[idx]
            )

            valid_distances.append(
                distance
            )


    # ========================================================
    # CREATE CONTEXT
    # ========================================================

    context = "\n\n".join(
        retrieved_chunks
    )


    # ========================================================
    # GEMMA PROMPT
    # ========================================================

    prompt = f"""
You are a helpful RAG chatbot.

Answer the user's question using ONLY
the context provided below.

Do not use your outside knowledge.

If the answer is not present in the context,
say exactly:

"Sorry, I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:
"""


    # ========================================================
    # GEMMA
    # ========================================================

    response = client.models.generate_content(
        model="gemma-4-26b-a4b-it",
        contents=prompt
    )


    return (
        response.text,
        retrieved_chunks,
        valid_distances
    )


# ============================================================
# PDF UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    st.success(
        f"PDF uploaded: {uploaded_file.name}"
    )


    with st.spinner(
        "Processing PDF..."
    ):

        chunks = process_pdf(
            uploaded_file
        )


        index = create_faiss_index(
            chunks
        )


    st.success(
        f"PDF processed successfully! "
        f"{len(chunks)} chunks created."
    )


    # ========================================================
    # QUESTION INPUT
    # ========================================================

    question = st.text_input(
        "Ask a question about the PDF:"
    )


    if question:

        with st.spinner(
            "Searching PDF and generating answer..."
        ):

            answer, retrieved_chunks, distances = answer_question(
                question,
                chunks,
                index,
                k=3
            )


        # ====================================================
        # ANSWER
        # ====================================================

        st.subheader("🤖 Answer")

        st.write(answer)


        # ====================================================
        # RETRIEVED CHUNKS
        # ====================================================

        if retrieved_chunks:

            #st.subheader("Retrieved PDF Chunks")


            for i, (
                chunk,
                distance
            ) in enumerate(
                zip(
                    retrieved_chunks,
                    distances
                )
            ):

                with st.expander(f"Chunk {i + 1} | Distance: {distance:.4f}"):

                    #st.write(chunk)

        else:
            st.info(
                "No relevant PDF chunks were found."
            )
