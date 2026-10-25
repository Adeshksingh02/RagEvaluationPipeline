import io
import os
import re
import json
import time
import html
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="RAG Evaluation Studio",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f8faff, #eef2ff);
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3, h4 {
        color: #111827 !important;
        font-weight: 800 !important;
    }

    .hero {
        background: linear-gradient(135deg, #312e81, #4f46e5, #7c3aed);
        border-radius: 22px;
        padding: 30px 34px;
        margin-bottom: 22px;
        box-shadow: 0 18px 45px rgba(79,70,229,.25);
    }

    .hero-title {
        color: white !important;
        font-size: 2.35rem;
        font-weight: 850;
    }

    .hero-subtitle {
        color: #e0e7ff !important;
        margin-top: 8px;
    }

    .card {
        background: white;
        border: 1px solid #e4e7ec;
        border-radius: 15px;
        padding: 18px;
        margin: 10px 0;
        box-shadow: 0 5px 15px rgba(16,24,40,.05);
    }

    .metric {
        background: white;
        border: 1px solid #e4e7ec;
        border-radius: 16px;
        padding: 18px 10px;
        text-align: center;
        min-height: 110px;
        box-shadow: 0 5px 15px rgba(16,24,40,.05);
    }

    .metric-value {
        color: #4f46e5;
        font-size: 1.85rem;
        font-weight: 850;
    }

    .metric-label {
        color: #475467;
        font-size: .85rem;
        font-weight: 600;
    }

    .chunk {
        background: white;
        border-left: 5px solid #6366f1;
        border-radius: 12px;
        padding: 16px;
        margin: 12px 0;
        line-height: 1.65;
        box-shadow: 0 5px 14px rgba(16,24,40,.05);
    }

    .chunk-meta {
        color: #4f46e5;
        font-size: .85rem;
        font-weight: 750;
        margin-bottom: 8px;
    }

    .answer {
        background: white;
        border: 1px solid #c7d2fe;
        border-radius: 15px;
        padding: 20px;
        line-height: 1.75;
        box-shadow: 0 7px 20px rgba(16,24,40,.05);
    }

    .status-ok {
        color: #027a48;
        font-weight: 800;
    }

    .status-bad {
        color: #b42318;
        font-weight: 800;
    }

    .status-warn {
        color: #b54708;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# TEXT UTILITIES
# =========================================================

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_tokens(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def token_set(text: str) -> set:
    return set(word_tokens(text))


def split_sentences(text: str) -> List[str]:
    return [
        x.strip()
        for x in re.split(r"(?<=[.!?।])\s+", text.strip())
        if x.strip()
    ]


def create_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> List[str]:
    text = clean_text(text)

    if not text:
        return []

    overlap = min(overlap, max(0, chunk_size - 1))
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# =========================================================
# FILE EXTRACTION
# =========================================================

def extract_text(uploaded_file) -> str:
    filename = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if filename.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="ignore")

    if filename.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if filename.endswith(".docx"):
        from docx import Document

        document = Document(io.BytesIO(data))
        return "\n".join(p.text for p in document.paragraphs)

    raise ValueError("Only PDF, TXT, MD and DOCX files are supported.")


# =========================================================
# RETRIEVER
# =========================================================

class Retriever:
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self.vectorizer = None
        self.matrix = None
        self.error = None

        try:
            texts = [x["text"] for x in chunks]

            if not texts:
                raise ValueError("No chunks available.")

            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=20000,
                sublinear_tf=True,
            )

            self.matrix = self.vectorizer.fit_transform(texts)

        except Exception as exc:
            self.error = str(exc)

    @property
    def ready(self) -> bool:
        return self.matrix is not None and self.vectorizer is not None

    def search(self, query: str, top_k: int = 4) -> List[Dict]:
        if not self.ready:
            return []

        if not query.strip():
            return []

        vector = self.vectorizer.transform([query])
        scores = cosine_similarity(vector, self.matrix).flatten()

        indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for rank, index in enumerate(indices, start=1):
            item = self.chunks[int(index)].copy()
            item["rank"] = rank
            item["score"] = float(scores[index])
            results.append(item)

        return results


# =========================================================
# GENERATOR
# =========================================================

def generate_local_answer(
    query: str,
    retrieved: List[Dict],
    max_sentences: int,
) -> Tuple[str, str]:
    """
    No-API fallback generator.
    It does not invent content. It selects relevant sentences
    only from retrieved context.
    """

    if not retrieved:
        return (
            "No relevant information was found in the uploaded documents.",
            "local-extractive",
        )

    query_words = token_set(query)
    candidates = []

    for source in retrieved:
        for sentence in split_sentences(source["text"]):
            sentence_words = token_set(sentence)

            if len(sentence_words) < 3:
                continue

            overlap = len(query_words.intersection(sentence_words))
            score = overlap / max(1, len(query_words))

            candidates.append((score, sentence))

    candidates.sort(key=lambda x: x[0], reverse=True)

    selected = []
    seen = set()

    for score, sentence in candidates:
        key = sentence.lower()

        if key in seen:
            continue

        if score <= 0 and selected:
            continue

        selected.append(sentence)
        seen.add(key)

        if len(selected) >= max_sentences:
            break

    if not selected:
        selected = [retrieved[0]["text"][:700]]

    return " ".join(selected), "local-extractive"


def generate_openai_answer(
    query: str,
    retrieved: List[Dict],
    model: str,
    max_sentences: int,
) -> Tuple[str, str]:
    """
    Optional OpenAI generator.
    Requires OPENAI_API_KEY environment variable.
    """

    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    client = OpenAI(api_key=api_key)

    context_parts = []

    for item in retrieved:
        context_parts.append(
            f"[Source {item['source']} | Chunk {item['chunk_id']}]\n"
            f"{item['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a strict document question-answering system.

Rules:
1. Answer only from the provided context.
2. Do not invent or assume facts.
3. If the answer is not present, say:
   "The answer is not available in the uploaded documents."
4. Keep the answer within {max_sentences} sentences.
5. Add source references like [Source: filename, Chunk: number].

Question:
{query}

Context:
{context}
"""

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You answer questions using only supplied document context.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    answer = response.choices[0].message.content.strip()
    return answer, f"openai:{model}"


def generate_answer(
    query: str,
    retrieved: List[Dict],
    mode: str,
    model: str,
    max_sentences: int,
) -> Tuple[str, str, str]:
    started = time.perf_counter()

    if mode == "OpenAI API":
        try:
            answer, engine = generate_openai_answer(
                query,
                retrieved,
                model,
                max_sentences,
            )
            return answer, engine, f"{time.perf_counter() - started:.3f}s"

        except Exception as exc:
            answer, engine = generate_local_answer(
                query,
                retrieved,
                max_sentences,
            )
            return (
                answer,
                f"{engine} fallback: {exc}",
                f"{time.perf_counter() - started:.3f}s",
            )

    answer, engine = generate_local_answer(
        query,
        retrieved,
        max_sentences,
    )

    return answer, engine, f"{time.perf_counter() - started:.3f}s"


# =========================================================
# EVALUATION
# =========================================================

def retrieval_score(query: str, retrieved: List[Dict]) -> float:
    if not retrieved:
        return 0.0

    query_words = token_set(query)

    if not query_words:
        return 0.0

    scores = []

    for item in retrieved:
        words = token_set(item["text"])
        overlap = len(query_words.intersection(words))
        scores.append(overlap / len(query_words))

    return float(min(1.0, max(scores)))


def faithfulness_score(answer: str, context: str) -> float:
    answer_sentences = split_sentences(answer)
    context_words = token_set(context)

    if not answer_sentences:
        return 0.0

    values = []

    for sentence in answer_sentences:
        words = token_set(sentence)

        if not words:
            continue

        values.append(len(words.intersection(context_words)) / len(words))

    return float(np.mean(values)) if values else 0.0


def answer_accuracy(answer: str, expected: str) -> float:
    answer_words = token_set(answer)
    expected_words = token_set(expected)

    if not answer_words or not expected_words:
        return 0.0

    common = answer_words.intersection(expected_words)

    precision = len(common) / len(answer_words)
    recall = len(common) / len(expected_words)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def diagnose(
    retrieval: float,
    faithfulness: float,
    accuracy: float,
    has_expected: bool,
    retrieval_threshold: float,
    faithfulness_threshold: float,
    accuracy_threshold: float,
) -> Tuple[str, str]:

    retrieval_failed = retrieval < retrieval_threshold
    faithfulness_failed = faithfulness < faithfulness_threshold
    accuracy_failed = has_expected and accuracy < accuracy_threshold

    if retrieval_failed and faithfulness_failed:
        return (
            "Retriever + Generator Issue",
            "Relevant context retrieve nahi hua aur generated answer bhi sufficiently supported nahi hai.",
        )

    if retrieval_failed:
        return (
            "Retriever Issue",
            "Query ke liye retrieved chunks ki relevance low hai.",
        )

    if faithfulness_failed:
        return (
            "Generator Faithfulness Issue",
            "Answer retrieved context se properly supported nahi hai.",
        )

    if accuracy_failed:
        return (
            "Answer Accuracy Issue",
            "Generated answer expected answer se sufficiently match nahi karta.",
        )

    return (
        "Passed",
        "Retriever aur generator configured thresholds ke according pass hain.",
    )


# =========================================================
# UI HELPERS
# =========================================================

def section(title: str, icon: str):
    st.markdown(
        f"## {icon} {html.escape(title)}"
    )


def metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div class="metric">
            <div class="metric-value">{html.escape(value)}</div>
            <div class="metric-label">{html.escape(label)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_chunk(item: Dict):
    st.markdown(
        f"""
        <div class="chunk">
            <div class="chunk-meta">
                Rank {item.get("rank", "-")}
                · {html.escape(item.get("source", "Unknown"))}
                · Chunk {item.get("chunk_id", "-")}
                · Similarity {item.get("score", 0):.3f}
            </div>
            <div>{html.escape(item["text"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🧪 RAG Evaluation Studio</div>
        <div class="hero-subtitle">
            Retriever, generator, citations, latency and root-cause diagnosis
            in one professional prototype.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("⚙️ Configuration")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
    )

    st.subheader("Generator")

    generator_mode = st.selectbox(
        "Generator mode",
        ["Local Extractive", "OpenAI API"],
    )

    openai_model = st.text_input(
        "OpenAI model",
        value="gpt-4o-mini",
    )

    if generator_mode == "OpenAI API":
        if os.getenv("OPENAI_API_KEY"):
            st.success("OPENAI_API_KEY detected")
        else:
            st.warning(
                "OPENAI_API_KEY nahi mila. Local fallback use hoga."
            )

    st.subheader("Chunking")

    chunk_size = st.slider(
        "Chunk size",
        300,
        2000,
        800,
        50,
    )

    overlap = st.slider(
        "Chunk overlap",
        0,
        400,
        120,
        20,
    )

    top_k = st.slider(
        "Top K retrieved chunks",
        1,
        10,
        4,
    )

    max_sentences = st.slider(
        "Maximum answer sentences",
        1,
        8,
        4,
    )

    st.subheader("Thresholds")

    retrieval_threshold = st.slider(
        "Retrieval threshold",
        0.0,
        1.0,
        0.30,
        0.05,
    )

    faithfulness_threshold = st.slider(
        "Faithfulness threshold",
        0.0,
        1.0,
        0.65,
        0.05,
    )

    accuracy_threshold = st.slider(
        "Accuracy threshold",
        0.0,
        1.0,
        0.60,
        0.05,
    )


# =========================================================
# EMPTY STATE
# =========================================================

if not uploaded_files:
    st.info(
        "Start karne ke liye sidebar se PDF, TXT, MD ya DOCX upload karein."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.info("📄 Multi-file document upload")

    with c2:
        st.info("🔍 Retriever inspection")

    with c3:
        st.info("📊 Generator and root-cause evaluation")

    st.stop()


# =========================================================
# LOAD FILES
# =========================================================

documents = []
file_status = []

for uploaded_file in uploaded_files:
    try:
        raw_text = extract_text(uploaded_file)
        text = clean_text(raw_text)

        if text:
            documents.append(
                {
                    "source": uploaded_file.name,
                    "text": text,
                }
            )

            file_status.append(
                {
                    "File": uploaded_file.name,
                    "Characters": len(text),
                    "Status": "Loaded",
                }
            )
        else:
            file_status.append(
                {
                    "File": uploaded_file.name,
                    "Characters": 0,
                    "Status": "No readable text",
                }
            )

    except Exception as exc:
        file_status.append(
            {
                "File": uploaded_file.name,
                "Characters": 0,
                "Status": f"Error: {exc}",
            }
        )


# =========================================================
# CREATE CHUNKS
# =========================================================

all_chunks = []
chunk_id = 0

for document in documents:
    document_chunks = create_chunks(
        document["text"],
        chunk_size,
        overlap,
    )

    for chunk in document_chunks:
        all_chunks.append(
            {
                "chunk_id": chunk_id,
                "source": document["source"],
                "text": chunk,
            }
        )
        chunk_id += 1


if not all_chunks:
    st.error("Uploaded files se readable text nahi mila.")
    st.stop()


retriever = Retriever(all_chunks)


# =========================================================
# DOCUMENT HEALTH
# =========================================================

section("System Health", "🩺")

h1, h2, h3, h4, h5 = st.columns(5)

with h1:
    metric_card("Documents", str(len(documents)))

with h2:
    metric_card("Chunks", str(len(all_chunks)))

with h3:
    metric_card(
        "Retriever",
        "READY" if retriever.ready else "FAILED",
    )

with h4:
    metric_card(
        "Generator",
        "READY" if generator_mode == "Local Extractive"
        or os.getenv("OPENAI_API_KEY")
        else "FALLBACK",
    )

with h5:
    metric_card(
        "Characters",
        f"{sum(len(x['text']) for x in documents):,}",
    )

with st.expander("File status"):
    st.dataframe(
        pd.DataFrame(file_status),
        use_container_width=True,
        hide_index=True,
    )

if not retriever.ready:
    st.error(f"Retriever failed: {retriever.error}")
    st.stop()


# =========================================================
# QUERY EVALUATION
# =========================================================

section("RAG Evaluation", "🔍")

query = st.text_input(
    "Question",
    placeholder="Example: What is the main topic of these documents?",
)

expected_answer = st.text_area(
    "Expected answer / ground truth (optional)",
    placeholder="Accuracy calculate karne ke liye expected answer enter karein.",
    height=100,
)

run = st.button(
    "Run Complete Evaluation",
    type="primary",
    use_container_width=True,
)


if run:
    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    retrieval_started = time.perf_counter()

    retrieved = retriever.search(
        query=query,
        top_k=top_k,
    )

    retrieval_latency = time.perf_counter() - retrieval_started

    context = "\n\n".join(
        item["text"] for item in retrieved
    )

    answer, generator_engine, generation_latency = generate_answer(
        query=query,
        retrieved=retrieved,
        mode=generator_mode,
        model=openai_model,
        max_sentences=max_sentences,
    )

    retrieval = retrieval_score(query, retrieved)
    faithfulness = faithfulness_score(answer, context)

    has_expected = bool(expected_answer.strip())

    accuracy = (
        answer_accuracy(answer, expected_answer)
        if has_expected
        else 0.0
    )

    root_cause, explanation = diagnose(
        retrieval=retrieval,
        faithfulness=faithfulness,
        accuracy=accuracy,
        has_expected=has_expected,
        retrieval_threshold=retrieval_threshold,
        faithfulness_threshold=faithfulness_threshold,
        accuracy_threshold=accuracy_threshold,
    )

    overall_pass = (
        retrieval >= retrieval_threshold
        and faithfulness >= faithfulness_threshold
        and (
            not has_expected
            or accuracy >= accuracy_threshold
        )
    )

    result = {
        "query": query,
        "expected_answer": expected_answer,
        "answer": answer,
        "retrieved": retrieved,
        "retrieval_score": retrieval,
        "faithfulness_score": faithfulness,
        "accuracy_score": accuracy,
        "root_cause": root_cause,
        "explanation": explanation,
        "overall_pass": overall_pass,
        "generator_engine": generator_engine,
        "retrieval_latency": retrieval_latency,
        "generation_latency": generation_latency,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    st.session_state["result"] = result


# =========================================================
# RESULTS
# =========================================================

if "result" in st.session_state:
    result = st.session_state["result"]

    section("Evaluation Results", "📊")

    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        metric_card(
            "Retrieval",
            f"{result['retrieval_score']:.0%}",
        )

    with m2:
        metric_card(
            "Faithfulness",
            f"{result['faithfulness_score']:.0%}",
        )

    with m3:
        metric_card(
            "Accuracy",
            "N/A" if not result["expected_answer"].strip()
            else f"{result['accuracy_score']:.0%}",
        )

    with m4:
        metric_card(
            "Retriever latency",
            f"{result['retrieval_latency']:.3f}s",
        )

    with m5:
        metric_card(
            "Generator latency",
            result["generation_latency"],
        )

    if result["overall_pass"]:
        st.success("✅ Overall pipeline passed")
    else:
        st.error("❌ Overall pipeline failed")

    st.subheader("Generated Answer")

    st.markdown(
        f"""
        <div class="answer">
            {html.escape(result["answer"])}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        f"Generator engine: {result['generator_engine']}"
    )

    st.subheader("Root-Cause Diagnosis")

    if result["root_cause"] == "Passed":
        st.success(f"✅ {result['root_cause']}")
    elif "Retriever" in result["root_cause"]:
        st.warning(f"⚠️ {result['root_cause']}")
    else:
        st.error(f"❌ {result['root_cause']}")

    st.write(result["explanation"])

    st.subheader("Retrieved Sources")

    if result["retrieved"]:
        for item in result["retrieved"]:
            show_chunk(item)
    else:
        st.warning("No context retrieved.")

    report_row = {
        "Timestamp": result["created_at"],
        "Query": result["query"],
        "Generated Answer": result["answer"],
        "Generator Engine": result["generator_engine"],
        "Retrieval Score": round(result["retrieval_score"], 4),
        "Faithfulness Score": round(result["faithfulness_score"], 4),
        "Accuracy Score": round(result["accuracy_score"], 4),
        "Retriever Latency": result["retrieval_latency"],
        "Generator Latency": result["generation_latency"],
        "Root Cause": result["root_cause"],
        "Explanation": result["explanation"],
        "Overall Pass": result["overall_pass"],
    }

    csv_data = pd.DataFrame([report_row]).to_csv(index=False)

    st.download_button(
        "Download CSV Report",
        data=csv_data,
        file_name="rag_evaluation_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.download_button(
        "Download JSON Report",
        data=json.dumps(
            {
                **report_row,
                "retrieved_sources": result["retrieved"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        file_name="rag_evaluation_report.json",
        mime="application/json",
        use_container_width=True,
    )


# =========================================================
# RETRIEVER INSPECTION
# =========================================================

section("Retriever Inspection", "🧩")

inspection_query = st.text_input(
    "Test retriever with another query",
    placeholder="Enter query to inspect retrieved chunks...",
)

if inspection_query.strip():
    inspection_results = retriever.search(
        inspection_query,
        top_k=top_k,
    )

    if inspection_results:
        for item in inspection_results:
            show_chunk(item)
    else:
        st.warning("No chunks retrieved.")


# =========================================================
# ALL CHUNKS
# =========================================================

section("All Document Chunks", "🗂️")

with st.expander("View all chunks"):
    chunks_table = pd.DataFrame(
        [
            {
                "Chunk ID": item["chunk_id"],
                "Source": item["source"],
                "Characters": len(item["text"]),
                "Text": item["text"],
            }
            for item in all_chunks
        ]
    )

    st.dataframe(
        chunks_table,
        use_container_width=True,
        hide_index=True,
    )


st.markdown("---")
st.caption(
    "RAG Evaluation Studio · Retriever + Generator observability prototype"
)
