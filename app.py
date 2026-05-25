import os
import re
import textwrap
from pathlib import Path
from typing import Optional

import streamlit as st
from google import genai


APP_TITLE = "AI-Augmented Technical Manuals"
DOC_DIR = Path(__file__).parent / "manuals"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CHUNK_SIZE = 1400
CHUNK_OVERLAP = 180
TOP_K = 6
STOP_WORDS = {
    "what",
    "which",
    "with",
    "from",
    "that",
    "this",
    "should",
    "required",
    "give",
    "tell",
    "show",
    "and",
    "the",
    "for",
    "are",
    "is",
    "in",
    "on",
    "to",
    "of",
    "a",
    "an",
}

DOC_FAMILY_BY_FILE = {
    "amm_brake_actuator_maintenance.md.txt": "AMM",
    "ipc_brake_housing_components.md.txt": "IPC",
    "wdm_brake_transducer_circuit.md.txt": "WDM",
    "fim_brake_system_fault.md.txt": "FIM",
}


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; max-width: 1180px;}
    div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 8px;}
    .config-box {
        border: 1px solid #2f3b52;
        border-radius: 8px;
        padding: 18px 18px 10px 18px;
        background: #0f1724;
        margin-bottom: 14px;
    }
    .config-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 14px;
        color: #f7f9fc;
    }
    .source-card {
        border: 1px solid #d9dee8;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        background: #ffffff;
    }
    .small-muted {font-size: 0.85rem; color: #5f6b7a;}
    </style>
    """,
    unsafe_allow_html=True,
)


def get_secret(name: str) -> str:
    if name in os.environ:
        return os.environ[name]
    try:
        return st.secrets[name]
    except Exception:
        return ""


def make_client(api_key: str):
    return genai.Client(api_key=api_key)


def clean_value(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "").strip()
    return value.replace("\u2013", "-").replace("\u2014", "-")


def first_match(pattern: str, text: str, default: str = "") -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    return clean_value(match.group(1)) if match else default


def extract_metadata(text: str, filename: str) -> dict:
    family = DOC_FAMILY_BY_FILE.get(filename, "MANUAL")
    return {
        "source": family,
        "filename": filename,
        "DMC": first_match(r"\|\s*\*\*DMC\*\*\s*\|\s*([^|]+)\|", text, "N/A"),
        "FIN": first_match(
            r"\|\s*\*\*(?:FIN|FIN \(Primary\)|Master Assembly FIN)\*\*\s*\|\s*([^|]+)\|",
            text,
            "N/A",
        ),
        "EFF": first_match(r"\|\s*\*\*(?:EFF|EFF \(Document\))\*\*\s*\|\s*([^|]+)\|", text, "N/A"),
        "zone": first_match(r"\|\s*\*\*Zone\*\*\s*\|\s*([^|]+)\|", text, "N/A"),
        "ATA": first_match(r"\|\s*\*\*ATA Chapter\*\*\s*\|\s*([^|]+)\|", text, "32-42"),
        "fault_code": first_match(r"\|\s*\*\*Fault Code\*\*\s*\|\s*`?([^`|]+)`?\s*\|", text, ""),
        "connector": first_match(r"\|\s*\*\*(?:Connector Ref|Connector)\*\*\s*\|\s*([^|]+)\|", text, ""),
    }


def load_manuals() -> list[dict]:
    manuals = []
    for path in sorted(DOC_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        manuals.append(
            {
                "filename": path.name,
                "content": text,
                "metadata": extract_metadata(text, path.name),
            }
        )
    return manuals


def chunk_text(text: str, metadata: dict) -> list[dict]:
    parts = re.split(r"(?=\n### |\n## )", text)
    chunks = []
    current = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(current) + len(part) <= CHUNK_SIZE:
            current = f"{current}\n\n{part}".strip()
        else:
            if current:
                chunks.append(current)
            current = part
            while len(current) > CHUNK_SIZE:
                chunks.append(current[:CHUNK_SIZE])
                current = current[CHUNK_SIZE - CHUNK_OVERLAP :]
    if current:
        chunks.append(current)

    enriched = []
    for index, chunk in enumerate(chunks):
        row_eff = first_match(r"\*\*EFF:\s*([^*]+)\*\*", chunk, metadata.get("EFF", "N/A"))
        enriched.append(
            {
                "text": chunk,
                "metadata": {
                    **metadata,
                    "row_eff": row_eff,
                    "chunk_index": index,
                    "chunk_id": f"{metadata['source']}-{index:03d}",
                },
            }
        )
    return enriched


def build_chunks() -> list[dict]:
    chunks = []
    for manual in load_manuals():
        chunks.extend(chunk_text(manual["content"], manual["metadata"]))
    return chunks


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9][a-z0-9-]{1,}", text.lower())
    return [token for token in tokens if token not in STOP_WORDS]


def prepare_search_text(text: str, metadata: dict) -> str:
    return " ".join(
        [
            text,
            metadata.get("source", ""),
            metadata.get("DMC", ""),
            metadata.get("FIN", ""),
            metadata.get("EFF", ""),
            metadata.get("row_eff", ""),
            metadata.get("fault_code", ""),
            metadata.get("connector", ""),
        ]
    ).lower()


def exact_phrase_score(question: str, search_text: str) -> float:
    score = 0.0
    quoted_terms = re.findall(r"`([^`]+)`|\\b([A-Z]{2,}-[A-Z0-9-]+|[A-Z0-9]+-[A-Z0-9-]+)\\b", question)
    for groups in quoted_terms:
        term = next((group for group in groups if group), "")
        if term and term.lower() in search_text:
            score += 3.0
    return score


def lexical_score(question: str, item: dict) -> float:
    search_text = item["search_text"]
    tokens = tokenize(question)
    if not tokens:
        return 0.0

    score = exact_phrase_score(question, search_text)
    for token in tokens:
        count = search_text.count(token)
        if not count:
            continue
        if any(char.isdigit() for char in token) or "-" in token:
            score += min(count, 4) * 1.25
        else:
            score += min(count, 4) * 0.55

    compact_question = " ".join(tokens)
    for phrase_len in (4, 3, 2):
        words = compact_question.split()
        for start in range(0, max(len(words) - phrase_len + 1, 0)):
            phrase = " ".join(words[start : start + phrase_len])
            if phrase and phrase in search_text:
                score += phrase_len * 0.75
    return score


def reset_index():
    st.session_state.pop("knowledge_index", None)


def build_index(api_key: Optional[str] = None) -> int:
    chunks = build_chunks()
    st.session_state["knowledge_index"] = [
        {
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "search_text": prepare_search_text(chunk["text"], chunk["metadata"]),
        }
        for chunk in chunks
    ]
    return len(chunks)


def collection_ready() -> bool:
    return bool(st.session_state.get("knowledge_index"))


def retrieve(api_key: str, question: str, families: list[str], top_k: int = TOP_K) -> list[dict]:
    rows = []
    for item in st.session_state.get("knowledge_index", []):
        if families and item["metadata"].get("source") not in families:
            continue
        score = lexical_score(question, item)
        rows.append({"text": item["text"], "metadata": item["metadata"], "score": score})
    return sorted(rows, key=lambda row: row["score"], reverse=True)[:top_k]


def build_context(retrieved: list[dict]) -> str:
    blocks = []
    for index, item in enumerate(retrieved, 1):
        meta = item["metadata"]
        blocks.append(
            textwrap.dedent(
                f"""
                [SOURCE {index}]
                Family: {meta.get('source')}
                DMC: {meta.get('DMC')}
                FIN: {meta.get('FIN')}
                Effectivity: {meta.get('row_eff') or meta.get('EFF')}
                Zone: {meta.get('zone')}
                Filename: {meta.get('filename')}
                Text:
                {item['text']}
                """
            ).strip()
        )
    return "\n\n---\n\n".join(blocks)


def answer_with_gemini(api_key: str, question: str, config: dict, retrieved: list[dict]) -> str:
    client = make_client(api_key)
    context = build_context(retrieved)
    prompt = f"""
You are an expert Airbus A320 maintenance engineer supporting a proof of concept.
Use ONLY the retrieved context. Do not invent values, part numbers, limits, authority, or steps.
If the answer is absent from the context, say that clearly.

Aircraft configuration:
- Program family: {config['program_family']}
- ATA chapter: {config['ata_chapter']}
- Document families selected: {', '.join(config['families'])}
- Effectivity: {config['effectivity']}

Required answer format:
1. Direct answer
2. Safety / caution items, if any
3. Procedure or diagnostic steps, if relevant
4. Source citations with DMC, FIN, document family, and filename

Retrieved context:
{context}

Technician question:
{question}
"""
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text or "No response text returned from Gemini."


def render_config_panel():
    st.markdown('<div class="config-box"><div class="config-title">Aircraft Configuration</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([1.2, 1.6, 1.8, 1.2])
    with c1:
        program_family = st.selectbox("Program Family", ["A320"], index=0)
    with c2:
        ata_chapter = st.selectbox("ATA Chapter", ["32-42 Braking System"], index=0)
    with c3:
        st.caption("Document Family")
        f1, f2, f3, f4 = st.columns(4)
        amm = f1.checkbox("AMM", value=True)
        ipc = f2.checkbox("IPC", value=True)
        wdm = f3.checkbox("WDM", value=True)
        fim = f4.checkbox("FIM", value=True)
    with c4:
        effectivity = st.selectbox("Effectivity", ["MSN 001-200", "MSN 201-500", "ALL"], index=0)
    st.markdown("</div>", unsafe_allow_html=True)
    families = [name for name, enabled in [("AMM", amm), ("IPC", ipc), ("WDM", wdm), ("FIM", fim)] if enabled]
    return {
        "program_family": program_family,
        "ata_chapter": ata_chapter,
        "families": families,
        "effectivity": effectivity,
    }


def render_architecture():
    st.subheader("POC Architecture")
    st.code(
        """
Aircraft config + technician question
        |
        v
Technical publication access layer
  - Program: A320
  - ATA: 32-42 Braking System
  - Approved document families
  - Effectivity: MSN range
        |
        v
Local retrieval and effectivity filtering
        |
        v
Relevant controlled-publication excerpts
        |
        v
Gemini answer generator
        |
        v
Grounded answer + warnings + DMC/FIN citations
        """.strip(),
        language="text",
    )
    st.markdown(
        """
This follows the deck idea of a smart knowledge layer over isolated technical manuals.
The visible experience abstracts away the underlying source count and presents the system as a controlled technical-publication layer.

Recommended production upgrades:
- Replace local markdown files with an approved technical-publication connector.
- Persist retrieval indexes to controlled storage and version them by manual revision.
- Add authentication, audit logs, and feedback capture.
- Keep human-in-the-loop approval for any maintenance release decision.
        """
    )


def render_sources(retrieved: list[dict]):
    for index, item in enumerate(retrieved, 1):
        meta = item["metadata"]
        st.markdown(
            f"""
            <div class="source-card">
            <b>Reference {index}: {meta.get('DMC')}</b><br/>
            <span class="small-muted">FIN: {meta.get('FIN')} | Effectivity: {meta.get('row_eff') or meta.get('EFF')}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Show cited excerpt"):
            st.text(item["text"][:1800])


st.title(APP_TITLE)
st.caption("Gemini-powered RAG POC for A320 ATA 32-42 braking manuals")

api_key_default = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")
with st.sidebar:
    st.header("Runtime")
    api_key = st.text_input("Gemini API key", value=api_key_default, type="password")
    st.caption("Gemini is used only for answer generation. Retrieval runs locally for demo stability.")
    reset = st.button("Reset knowledge layer")
    if reset:
        reset_index()
        st.success("Knowledge layer reset.")

config = render_config_panel()

tab_ask, tab_arch = st.tabs(["Ask Manuals", "Architecture"])

with tab_ask:
    if not api_key:
        st.warning("Enter a Gemini API key in the sidebar before asking questions.")

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Initialize knowledge layer", type="primary"):
            with st.spinner("Preparing the technical-publication knowledge layer..."):
                build_index()
            st.success("Knowledge layer is ready.")
    with c2:
        st.caption("First run prepares the local demonstration knowledge layer. Retrieval runs locally for stable demos.")

    examples = [
        "What torque is required for the structural retention bolts, and what sequence should I follow?",
        "CMS-FAULT-32-42-E12 is shown. What is the first isolation step and possible causes?",
        "For connector CN-LG32 Pin C, give the signal role, wire identifier, and termination.",
        "Which actuator part number applies to MSN 001-200 and what O-ring kit is required?",
        "What pressure must be confirmed before loosening any brake line union?",
    ]
    question = st.text_area("Technician question", value=examples[0], height=100)
    top_k = st.slider("Evidence depth", min_value=3, max_value=10, value=TOP_K)

    if st.button("Ask Gemini", disabled=not api_key or not config["families"]):
        if not collection_ready():
            with st.spinner("Preparing the knowledge layer..."):
                build_index()
        with st.spinner("Retrieving approved technical context..."):
            retrieved = retrieve(api_key, question, config["families"], top_k=top_k)
        with st.spinner("Generating grounded answer with Gemini..."):
            answer = answer_with_gemini(api_key, question, config, retrieved)
        st.subheader("Answer")
        st.markdown(answer)
        st.subheader("Traceability")
        render_sources(retrieved)

with tab_arch:
    render_architecture()
