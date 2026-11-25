"""
Agentic RAG
===========
Single-file LangChain + Streamlit app.

Flow:
  1. Retrieve from the uploaded PDF (FAISS).
  2. If the best match is strong AND the LLM confirms the context answers the
     question, answer from the PDF only.
  3. Otherwise hand the question to a tool-calling agent that chooses between
     Tavily (web), Wikipedia (background) and arXiv (papers).
  4. The source badge is derived from the tools the agent actually called.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import hashlib
import hmac
import html
import json
import logging
import os
import tempfile
import textwrap
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

# ── Third-party ───────────────────────────────────────────────────────────────
import streamlit as st
from dotenv import load_dotenv

# LangChain 1.x: splitters live in their own package; the classic AgentExecutor
# lives in langchain_classic.
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("agentic_rag")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
# Cosine similarity cut-off for "the PDF probably covers this". Tuned for
# text-embedding-3-small, where on-topic chunks usually score ~0.35-0.65 and
# unrelated ones fall below ~0.25. Adjustable in the sidebar.
DEFAULT_THRESHOLD = 0.35
MIN_PDF_CHARS = 200  # below this the PDF is almost certainly scanned/empty
# Where FAISS indexes are persisted (a Docker volume in production).
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
# Abuse / cost controls (per client address)
MAX_QUESTIONS_PER_HOUR = int(os.getenv("MAX_QUESTIONS_PER_HOUR", "30"))
MAX_LOGIN_FAILURES = 5          # per 15 minutes

ANSWERABLE = "[[ANSWERABLE]]"
NOT_ANSWERABLE = "[[NOT_ANSWERABLE]]"

CSS = """
    <style>
    /* ── Base ── */
    [data-testid="stAppViewContainer"] {
        background: #0f1117;
        color: #e2e8f0;
    }
    [data-testid="stSidebar"] {
        background: #161b27;
        border-right: 1px solid #2d3748;
    }
    [data-testid="stSidebar"] * { color: #cbd5e0 !important; }

    /* ── Title ── */
    .rag-title {
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 1.9rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.15rem;
    }
    .rag-subtitle {
        color: #64748b;
        font-size: 0.85rem;
        font-family: 'SF Mono', monospace;
        margin-bottom: 1.5rem;
    }

    /* ── Chat bubbles ── */
    .msg-user {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px 12px 2px 12px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.92rem;
    }
    .msg-bot {
        background: #0d1f3c;
        border: 1px solid #1e3a5f;
        border-left: 3px solid #3b82f6;
        border-radius: 2px 12px 12px 12px;
        padding: 14px 16px;
        margin: 8px 0;
        font-size: 0.92rem;
        line-height: 1.65;
    }
    .msg-bot.pdf   { border-left-color: #10b981; }
    .msg-bot.web   { border-left-color: #f59e0b; }
    .msg-bot.wiki  { border-left-color: #8b5cf6; }
    .msg-bot.arxiv { border-left-color: #ef4444; }
    .msg-bot.mixed { border-left-color: #06b6d4; }

    /* ── Source badge ── */
    .source-badge {
        display: inline-block;
        font-family: 'SF Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 2px 8px;
        border-radius: 999px;
        margin-bottom: 8px;
    }
    .badge-pdf   { background: #064e3b; color: #6ee7b7; border: 1px solid #10b981; }
    .badge-web   { background: #451a03; color: #fcd34d; border: 1px solid #f59e0b; }
    .badge-wiki  { background: #2e1065; color: #c4b5fd; border: 1px solid #8b5cf6; }
    .badge-arxiv { background: #450a0a; color: #fca5a5; border: 1px solid #ef4444; }
    .badge-mixed { background: #0c4a6e; color: #7dd3fc; border: 1px solid #06b6d4; }
    .badge-error { background: #1c1917; color: #a8a29e; border: 1px solid #57534e; }

    /* ── Chunk card ── */
    .chunk-card {
        background: #1a2035;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.8rem;
        color: #94a3b8;
        font-family: 'SF Mono', monospace;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .chunk-meta {
        font-size: 0.72rem;
        color: #475569;
        margin-bottom: 4px;
    }

    /* ── Sidebar labels ── */
    .sidebar-label {
        font-size: 0.72rem;
        font-family: 'SF Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #475569;
        margin: 1.2rem 0 0.25rem 0;
    }

    /* ── Status pills ── */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.75rem;
        padding: 3px 10px;
        border-radius: 999px;
        font-family: 'SF Mono', monospace;
    }
    .status-ok   { background: #064e3b; color: #6ee7b7; border: 1px solid #065f46; }
    .status-warn { background: #451a03; color: #fcd34d; border: 1px solid #78350f; }

    /* ── Misc ── */
    .divider { border-top: 1px solid #1e293b; margin: 1rem 0; }
    div[data-testid="stExpander"] { background: #161b27; border: 1px solid #2d3748; border-radius: 8px; }
    </style>
"""


# ──────────────────────────────────────────────────────────────────────────────
# PDF PROCESSING
# ──────────────────────────────────────────────────────────────────────────────
def load_and_index_pdf(pdf_bytes: bytes, embeddings) -> FAISS:
    """Parse a PDF, split it into chunks and build a FAISS index."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        pages: list[Document] = PyPDFLoader(tmp_path).load()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    total_chars = sum(len(p.page_content.strip()) for p in pages)
    if not pages or total_chars < MIN_PDF_CHARS:
        raise ValueError(
            "No readable text found. This PDF looks scanned or image-only; "
            "run OCR on it first, then upload it again."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = [c for c in splitter.split_documents(pages) if c.page_content.strip()]
    if not chunks:
        raise ValueError("The PDF produced no usable text chunks.")

    log.info("Indexing %d pages -> %d chunks", len(pages), len(chunks))
    return FAISS.from_documents(chunks, embeddings)


# ──────────────────────────────────────────────────────────────────────────────
# INDEX PERSISTENCE  (survives container restarts; re-uploads skip re-embedding)
# ──────────────────────────────────────────────────────────────────────────────
def pdf_digest(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]


def _index_dir(digest: str) -> Path:
    return DATA_DIR / "indexes" / digest


def _write_last(name: str, digest: str) -> None:
    (DATA_DIR / "last.json").write_text(json.dumps({"name": name, "digest": digest}))


def index_pdf_cached(pdf_bytes: bytes, name: str, embeddings) -> FAISS:
    """Load this exact PDF's index from disk if present, else build and save it."""
    digest = pdf_digest(pdf_bytes)
    path = _index_dir(digest)
    try:
        if (path / "index.faiss").exists():
            store = FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
            log.info("Loaded cached index %s", digest)
        else:
            store = load_and_index_pdf(pdf_bytes, embeddings)
            path.mkdir(parents=True, exist_ok=True)
            store.save_local(str(path))
            log.info("Saved index %s", digest)
        _write_last(name, digest)
        return store
    except OSError:
        # A read-only or full disk must not break the app; fall back to in-memory.
        log.exception("Index persistence failed; continuing in memory")
        return load_and_index_pdf(pdf_bytes, embeddings)


def load_last_index(embeddings) -> Optional[tuple[FAISS, str]]:
    """Restore the most recently used PDF index after a restart, if there is one."""
    try:
        meta = json.loads((DATA_DIR / "last.json").read_text())
        path = _index_dir(meta["digest"])
        if not (path / "index.faiss").exists():
            return None
        # The pickle is read only from DATA_DIR, which this app alone writes.
        store = FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
        return store, meta["name"]
    except (OSError, ValueError, KeyError):
        return None


# ──────────────────────────────────────────────────────────────────────────────
# ACCESS CONTROL
# ──────────────────────────────────────────────────────────────────────────────
class RateLimiter:
    """Thread-safe sliding-window limiter keyed by client id."""

    def __init__(self, limit: int, window_s: float, clock=time.monotonic):
        self.limit, self.window_s, self._clock = limit, window_s, clock
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, q: deque, now: float) -> None:
        while q and now - q[0] > self.window_s:
            q.popleft()

    def blocked(self, key: str) -> bool:
        with self._lock:
            now = self._clock()
            q = self._hits[key]
            self._prune(q, now)
            return len(q) >= self.limit

    def hit(self, key: str) -> None:
        with self._lock:
            now = self._clock()
            q = self._hits[key]
            self._prune(q, now)
            q.append(now)

    def retry_after(self, key: str) -> int:
        with self._lock:
            q = self._hits[key]
            if not q:
                return 0
            return max(0, int(self.window_s - (self._clock() - q[0])) + 1)


@st.cache_resource
def get_limiters() -> tuple[RateLimiter, RateLimiter]:
    """Process-wide limiters (shared by every browser session)."""
    return RateLimiter(MAX_LOGIN_FAILURES, 900), RateLimiter(MAX_QUESTIONS_PER_HOUR, 3600)


def client_id() -> str:
    """Caller address. Behind Caddy, X-Forwarded-For carries the real client IP."""
    try:
        fwd = st.context.headers.get("X-Forwarded-For", "")
        return fwd.split(",")[0].strip() or "local"
    except Exception:
        return "local"


def password_ok(entered: str, expected: str) -> bool:
    """Constant-time comparison so response timing doesn't leak the password."""
    return hmac.compare_digest(entered.encode(), expected.encode())


def password_gate() -> bool:
    """True when access is allowed. If APP_PASSWORD is unset the app is open."""
    expected = os.getenv("APP_PASSWORD", "")
    if not expected or st.session_state.get("authed"):
        return True
    st.markdown('<p class="rag-title">Agentic RAG</p>', unsafe_allow_html=True)
    login_limiter, _ = get_limiters()
    cid = client_id()
    if login_limiter.blocked(cid):
        st.error(f"Too many failed attempts. Try again in about "
                 f"{login_limiter.retry_after(cid) // 60 + 1} minutes.")
        return False
    entered = st.text_input("Password", type="password")
    if entered:
        if password_ok(entered, expected):
            st.session_state.authed = True
            st.rerun()
        login_limiter.hit(cid)
        log.warning("Failed login from %s", cid)
        st.error("Incorrect password.")
    return False


def key_field(label: str, widget_key: str, env_name: str, placeholder: str) -> str:
    """
    Use the server-side key if one is set (never sent to the browser),
    otherwise show an input so a local user can paste their own.
    """
    env_val = os.getenv(env_name, "")
    if env_val:
        st.caption(f"{label}: configured on the server")
        return env_val
    st.markdown(f'<p class="sidebar-label">{html.escape(label)}</p>', unsafe_allow_html=True)
    return st.text_input(widget_key, type="password", placeholder=placeholder,
                         label_visibility="collapsed")


# ──────────────────────────────────────────────────────────────────────────────
# RETRIEVAL + PDF ANSWERING
# ──────────────────────────────────────────────────────────────────────────────
def l2_to_cosine(squared_l2: float) -> float:
    """
    FAISS IndexFlatL2 returns SQUARED L2 distance. OpenAI embeddings are unit
    length, so squared_l2 = 2 - 2*cos  =>  cos = 1 - squared_l2 / 2.
    """
    return 1.0 - float(squared_l2) / 2.0


def retrieve_from_pdf(query: str, store: FAISS) -> tuple[list[Document], float]:
    """Return the top-K chunks and the best cosine similarity among them."""
    hits = store.similarity_search_with_score(query, k=TOP_K)
    if not hits:
        return [], 0.0
    docs = [d for d, _ in hits]
    best = max(l2_to_cosine(s) for _, s in hits)
    return docs, best


def format_pdf_context(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        # PyPDFLoader pages are 0-indexed; show human page numbers.
        page = doc.metadata.get("page")
        page_label = page + 1 if isinstance(page, int) else "?"
        parts.append(f"[Chunk {i} | Page {page_label}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


def answer_from_pdf(query: str, docs: list[Document], llm: ChatOpenAI) -> tuple[bool, str]:
    """
    Ask the LLM to answer strictly from the PDF context. The model must start
    its reply with a verdict token, so sufficiency is decided explicitly
    instead of by scanning the answer for phrases.
    Returns (answerable, answer_text).
    """
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You answer questions using ONLY the PDF context below.\n"
            f"Begin your reply with exactly one token: {ANSWERABLE} if the context "
            f"contains enough information to answer, or {NOT_ANSWERABLE} if it does not.\n"
            f"If {ANSWERABLE}: give the answer and cite page numbers like (p. 4). "
            "Do not use outside knowledge.\n"
            f"If {NOT_ANSWERABLE}: write nothing else.\n\n"
            "CONTEXT:\n{context}",
        ),
        ("human", "{question}"),
    ])
    raw = (prompt | llm | StrOutputParser()).invoke(
        {"context": format_pdf_context(docs), "question": query}
    ).strip()

    if raw.startswith(NOT_ANSWERABLE) or NOT_ANSWERABLE in raw[:40]:
        return False, ""
    if raw.startswith(ANSWERABLE):
        return True, raw[len(ANSWERABLE):].strip()
    # Model ignored the protocol: treat as not answerable rather than trust it.
    log.warning("PDF answer missing verdict token; falling back to agent")
    return False, ""


# ──────────────────────────────────────────────────────────────────────────────
# EXTERNAL TOOLS + AGENT
# ──────────────────────────────────────────────────────────────────────────────
def build_tools(tavily_api_key: Optional[str]) -> list:
    tools = []

    if tavily_api_key:
        os.environ["TAVILY_API_KEY"] = tavily_api_key
        tools.append(TavilySearchResults(
            max_results=4,
            description=(
                "Web search. Use for current events, recent news, pricing, "
                "company details, product updates, anything time-sensitive."
            ),
        ))

    tools.append(WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=2000),
        description=(
            "Wikipedia. Use for general background: definitions, history, "
            "people, organisations, broad explanations."
        ),
    ))

    tools.append(ArxivQueryRun(
        api_wrapper=ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=2000),
        description=(
            "arXiv search. Use for academic papers, AI/ML research, scientific "
            "concepts and technical citations."
        ),
    ))
    return tools


def build_agent(llm: ChatOpenAI, tools: list) -> AgentExecutor:
    tool_names = ", ".join(t.name for t in tools)
    system_msg = (
        f"You are a research assistant with these tools: {tool_names}.\n"
        "The user's question was not answerable from their uploaded PDF, or they "
        "asked for current/research information.\n"
        "Rules:\n"
        "- You MUST call at least one tool before answering. Never answer from "
        "memory alone.\n"
        "- Pick the tool that fits: web search for current/time-sensitive info, "
        "Wikipedia for background, arXiv for papers and research.\n"
        "- Base your answer only on what the tools return. If they return nothing "
        "useful, say you could not find a reliable answer. Do not guess.\n"
        "- Mention which source each key claim came from."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        return_intermediate_steps=True,   # lets us report the tools actually used
        max_iterations=5,
        max_execution_time=90,
    )


def source_from_steps(steps: list) -> str:
    """Map the tools the agent actually called to a source label."""
    used = set()
    for action, _obs in steps:
        name = getattr(action, "tool", "").lower()
        if "tavily" in name:
            used.add("web")
        elif "wiki" in name:
            used.add("wiki")
        elif "arxiv" in name:
            used.add("arxiv")
    if not used:
        return "none"
    return next(iter(used)) if len(used) == 1 else "mixed"


# ──────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR  (PDF first, agent as fallback)
# ──────────────────────────────────────────────────────────────────────────────
def orchestrate(
    query: str,
    store: Optional[FAISS],
    agent: Optional[AgentExecutor],
    llm: ChatOpenAI,
    history: list,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    result = {
        "answer": "", "source": "none", "chunks": [],
        "relevance": 0.0, "note": "",
    }

    # Step 1: PDF
    if store is not None:
        chunks, relevance = retrieve_from_pdf(query, store)
        result["chunks"], result["relevance"] = chunks, relevance

        if chunks and relevance >= threshold:
            try:
                ok, answer = answer_from_pdf(query, chunks, llm)
            except Exception as exc:
                log.exception("PDF answer failed")
                result["note"] = f"PDF answering failed ({exc}); trying external tools."
            else:
                if ok and answer:
                    result.update(answer=answer, source="pdf")
                    return result
                result["note"] = "The PDF did not contain a clear answer; used external tools."
        else:
            result["note"] = (
                f"Best PDF match was {relevance:.2f} (threshold {threshold:.2f}); "
                "used external tools."
            )

    # Step 2: agent
    if agent is None:
        result["answer"] = "No PDF is indexed and no external tools are available."
        return result

    messages = []
    for role, text, *_ in history[-6:]:
        messages.append(HumanMessage(content=text) if role == "user" else AIMessage(content=text))

    try:
        out = agent.invoke({"input": query, "chat_history": messages})
    except Exception as exc:
        log.exception("Agent call failed")
        result["answer"] = f"The external tool call failed: {exc}"
        return result

    result["answer"] = out.get("output", "").strip() or "No answer was produced."
    result["source"] = source_from_steps(out.get("intermediate_steps", []))
    if result["source"] == "none":
        result["note"] = "No tool was called, so this answer is not backed by a source."
    return result


# ──────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ──────────────────────────────────────────────────────────────────────────────
SOURCE_META = {
    "pdf":   ("PDF Document",   "badge-pdf",   "msg-bot pdf"),
    "web":   ("Tavily Web",     "badge-web",   "msg-bot web"),
    "wiki":  ("Wikipedia",      "badge-wiki",  "msg-bot wiki"),
    "arxiv": ("arXiv Papers",   "badge-arxiv", "msg-bot arxiv"),
    "mixed": ("Multiple Tools", "badge-mixed", "msg-bot mixed"),
    "none":  ("No Source",      "badge-error", "msg-bot"),
}


def render_message(role: str, text: str, source: str = "none"):
    safe = html.escape(text).replace("\n", "<br>")
    if role == "user":
        st.markdown(f'<div class="msg-user">👤 {safe}</div>', unsafe_allow_html=True)
        return
    label, badge_cls, bubble_cls = SOURCE_META.get(source, SOURCE_META["none"])
    st.markdown(
        f'<div class="{bubble_cls}"><span class="source-badge {badge_cls}">⬤ {label}</span>'
        f"<br>{safe}</div>",
        unsafe_allow_html=True,
    )


def render_chunks(chunks: list[Document], relevance: float):
    if not chunks:
        return
    with st.expander(f"📄 Retrieved PDF chunks (best similarity: {relevance:.2f})"):
        for i, doc in enumerate(chunks, 1):
            page = doc.metadata.get("page")
            page_label = page + 1 if isinstance(page, int) else "?"
            preview = html.escape(textwrap.shorten(doc.page_content, width=600, placeholder="…"))
            st.markdown(
                f'<div class="chunk-card"><div class="chunk-meta">Chunk {i} · Page {page_label}</div>'
                f"{preview}</div>",
                unsafe_allow_html=True,
            )


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str, model: str) -> ChatOpenAI:
    return ChatOpenAI(api_key=api_key, model=model, temperature=0.2, timeout=60, max_retries=2)


@st.cache_resource(show_spinner=False)
def get_agent(api_key: str, model: str, tavily_key: str) -> AgentExecutor:
    return build_agent(get_llm(api_key, model), build_tools(tavily_key or None))


# ──────────────────────────────────────────────────────────────────────────────
# STREAMLIT APP
# ──────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Agentic RAG", page_icon="🔍", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    for k, v in {"vector_store": None, "pdf_key": None, "pdf_name": None,
                 "history": [], "restore_tried": False}.items():
        st.session_state.setdefault(k, v)

    if not password_gate():
        st.stop()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🔍 Agentic RAG")
        openai_key = key_field("OpenAI API key", "openai", "OPENAI_API_KEY", "sk-...")
        tavily_key = key_field("Tavily API key (web search)", "tavily", "TAVILY_API_KEY", "tvly-...")
        st.markdown('<p class="sidebar-label">Chat model</p>', unsafe_allow_html=True)
        model_name = st.text_input("model", value=DEFAULT_CHAT_MODEL, label_visibility="collapsed")
        st.markdown('<p class="sidebar-label">PDF similarity threshold</p>', unsafe_allow_html=True)
        threshold = st.slider(
            "threshold", 0.10, 0.80, DEFAULT_THRESHOLD, 0.01, label_visibility="collapsed",
            help="Higher = trust the PDF only on strong matches. Lower = try the PDF more often.",
        )

        pdf_ok = st.session_state.vector_store is not None
        st.markdown(
            f'<span class="status-pill {"status-ok" if openai_key else "status-warn"}">{"✓" if openai_key else "✗"} OpenAI</span> '
            f'<span class="status-pill {"status-ok" if tavily_key else "status-warn"}">{"✓" if tavily_key else "–"} Tavily</span> '
            f'<span class="status-pill {"status-ok" if pdf_ok else "status-warn"}">{"✓" if pdf_ok else "–"} PDF</span>',
            unsafe_allow_html=True,
        )
        if not tavily_key:
            st.caption("No Tavily key: web search is disabled (Wikipedia + arXiv only).")
        if pdf_ok:
            st.caption(f"📄 {st.session_state.pdf_name}")

        st.markdown('<p class="sidebar-label">Upload PDF</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader("pdf", type=["pdf"], label_visibility="collapsed")

        if uploaded and not openai_key:
            st.warning("Add your OpenAI key to index the PDF.")
        elif uploaded:
            key = (uploaded.name, uploaded.size)
            if key != st.session_state.pdf_key:
                with st.spinner("Reading and embedding the PDF…"):
                    try:
                        store = index_pdf_cached(
                            uploaded.getvalue(), uploaded.name,
                            OpenAIEmbeddings(api_key=openai_key, model=EMBEDDING_MODEL),
                        )
                    except ValueError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        log.exception("Embedding failed")
                        st.error(f"Embedding failed: {exc}. Check your OpenAI key and quota.")
                    else:
                        st.session_state.update(vector_store=store, pdf_key=key,
                                                pdf_name=uploaded.name, history=[])
                        st.success(f"Indexed {uploaded.name}")
                        st.rerun()

        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    # ── Main ─────────────────────────────────────────────────────────────────
    st.markdown('<p class="rag-title">Agentic RAG</p>', unsafe_allow_html=True)
    st.markdown('<p class="rag-subtitle">PDF-first retrieval · Tavily · Wikipedia · arXiv</p>',
                unsafe_allow_html=True)

    if not openai_key:
        st.info("Add your OpenAI API key in the sidebar to get started.", icon="🔑")
        st.stop()
    if os.getenv("OPENAI_API_KEY") and not os.getenv("APP_PASSWORD"):
        st.warning("Server-side API keys are in use and APP_PASSWORD is not set: anyone who "
                   "can reach this page can spend your quota. Set APP_PASSWORD.", icon="⚠️")

    if not st.session_state.restore_tried and st.session_state.vector_store is None:
        st.session_state.restore_tried = True
        restored = load_last_index(OpenAIEmbeddings(api_key=openai_key, model=EMBEDDING_MODEL))
        if restored:
            st.session_state.update(vector_store=restored[0], pdf_name=restored[1], pdf_key=None)
            st.rerun()

    if not st.session_state.vector_store:
        st.info("No PDF yet. Upload one in the sidebar, or ask a general question to use "
                "Wikipedia, arXiv and web search.", icon="📄")

    llm = get_llm(openai_key, model_name)
    agent = get_agent(openai_key, model_name, tavily_key)

    for role, text, source, chunks, relevance, note in st.session_state.history:
        render_message(role, text, source)
        if role == "assistant":
            if note:
                st.caption(note)
            if chunks and source == "pdf":
                render_chunks(chunks, relevance)

    with st.form("ask", clear_on_submit=True):
        c1, c2 = st.columns([9, 1])
        question = c1.text_input("question", label_visibility="collapsed",
                                 placeholder="Ask about your PDF, or anything else…")
        submitted = c2.form_submit_button("➤", use_container_width=True)

    if submitted:
        q = question.strip()
        if not q:
            st.warning("Type a question first.")
            return
        _, question_limiter = get_limiters()
        cid = client_id()
        if question_limiter.blocked(cid):
            st.warning(f"Question limit reached ({MAX_QUESTIONS_PER_HOUR}/hour). "
                       f"Try again in about {question_limiter.retry_after(cid) // 60 + 1} minutes.")
            return
        question_limiter.hit(cid)
        prior = list(st.session_state.history)
        with st.spinner("Searching your PDF, then external sources if needed…"):
            res = orchestrate(q, st.session_state.vector_store, agent, llm, prior, threshold)
        st.session_state.history.append(("user", q, "none", [], 0.0, ""))
        st.session_state.history.append(
            ("assistant", res["answer"], res["source"], res["chunks"], res["relevance"], res["note"])
        )
        st.rerun()


if __name__ == "__main__":
    main()
