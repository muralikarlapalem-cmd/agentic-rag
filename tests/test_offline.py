"""Offline tests: real PDF parsing + real FAISS, with fake embeddings/LLM (no API keys)."""
import io
import zlib
import math
import re

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from reportlab.pdfgen import canvas
from pypdf import PdfWriter

import app


class KeywordEmbeddings(Embeddings):
    """Unit-length bag-of-words vectors, so cosine similarity is predictable."""
    DIM = 256

    def _vec(self, text):
        v = np.zeros(self.DIM)
        for w in re.findall(r"[a-z]+", text.lower()):
            v[zlib.crc32(w.encode()) % self.DIM] += 1
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def make_pdf(pages):
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for text in pages:
        y = 800
        for line in text.split("\n"):
            c.drawString(50, y, line)
            y -= 18
        c.showPage()
    c.save()
    return buf.getvalue()


PAGES = [
    "Quantum error correction protects qubits from decoherence.\n" * 8,
    "The transformer architecture uses self attention over token sequences.\n" * 8,
]


@pytest.fixture(scope="module")
def store():
    return app.load_and_index_pdf(make_pdf(PAGES), KeywordEmbeddings())


def test_l2_to_cosine_matches_true_cosine():
    e = KeywordEmbeddings()
    a, b = np.array(e.embed_query("transformer attention tokens")), np.array(e.embed_query("attention tokens model"))
    sq_l2 = float(np.sum((a - b) ** 2))
    assert math.isclose(app.l2_to_cosine(sq_l2), float(a @ b), abs_tol=1e-9)


def test_faiss_score_is_squared_l2_so_conversion_is_right(store):
    docs, best = app.retrieve_from_pdf("transformer self attention token sequences", store)
    q = np.array(KeywordEmbeddings().embed_query("transformer self attention token sequences"))
    true_best = max(float(q @ np.array(KeywordEmbeddings().embed_query(d.page_content))) for d in docs)
    assert math.isclose(best, true_best, abs_tol=1e-4)


def test_relevant_scores_above_irrelevant(store):
    _, on_topic = app.retrieve_from_pdf("how does self attention work in a transformer", store)
    _, off_topic = app.retrieve_from_pdf("recipe for sourdough bread with rye flour", store)
    assert on_topic > app.DEFAULT_THRESHOLD > off_topic


def test_page_numbers_are_one_indexed(store):
    docs, _ = app.retrieve_from_pdf("quantum error correction qubits", store)
    assert "Page 1]" in app.format_pdf_context(docs[:1])


def test_scanned_pdf_gives_clear_error():
    w = PdfWriter(); w.add_blank_page(width=200, height=200)
    buf = io.BytesIO(); w.write(buf)
    with pytest.raises(ValueError, match="scanned or image-only"):
        app.load_and_index_pdf(buf.getvalue(), KeywordEmbeddings())


def test_temp_file_removed_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(app.tempfile, "tempdir", str(tmp_path))
    with pytest.raises(Exception):
        app.load_and_index_pdf(b"not a pdf", KeywordEmbeddings())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("reply,expected", [
    (f"{app.ANSWERABLE} Self attention relates tokens (p. 2).", (True, "Self attention relates tokens (p. 2).")),
    (app.NOT_ANSWERABLE, (False, "")),
    ("Sure! Here is an answer with no verdict token.", (False, "")),
])
def test_verdict_parsing(store, reply, expected):
    docs, _ = app.retrieve_from_pdf("attention", store)
    assert app.answer_from_pdf("q", docs, FakeListChatModel(responses=[reply])) == expected


class StubAgent:
    def __init__(self, steps, output="agent answer"):
        self.steps, self.output, self.calls = steps, output, 0
    def invoke(self, _):
        self.calls += 1
        return {"output": self.output, "intermediate_steps": self.steps}


class Act:
    def __init__(self, tool): self.tool = tool


def test_strong_pdf_match_skips_agent(store):
    agent = StubAgent([])
    llm = FakeListChatModel(responses=[f"{app.ANSWERABLE} Answer (p. 2)."])
    r = app.orchestrate("self attention transformer tokens", store, agent, llm, [])
    assert r["source"] == "pdf" and agent.calls == 0


def test_weak_match_goes_to_agent(store):
    agent = StubAgent([(Act("wikipedia"), "obs")])
    r = app.orchestrate("sourdough bread rye flour", store, agent, FakeListChatModel(responses=["x"]), [])
    assert r["source"] == "wiki" and agent.calls == 1 and "threshold" in r["note"]


def test_pdf_says_not_answerable_falls_back(store):
    agent = StubAgent([(Act("arxiv"), "obs")])
    llm = FakeListChatModel(responses=[app.NOT_ANSWERABLE])
    r = app.orchestrate("self attention transformer tokens", store, agent, llm, [])
    assert r["source"] == "arxiv" and agent.calls == 1


def test_no_pdf_uses_agent():
    r = app.orchestrate("latest news", None, StubAgent([(Act("tavily_search_results_json"), "o")]), None, [])
    assert r["source"] == "web"


@pytest.mark.parametrize("tools,expected", [
    (["tavily_search_results_json"], "web"), (["wikipedia"], "wiki"), (["arxiv"], "arxiv"),
    (["wikipedia", "arxiv"], "mixed"), (["arxiv", "arxiv"], "arxiv"), ([], "none"),
])
def test_source_from_steps(tools, expected):
    assert app.source_from_steps([(Act(t), "o") for t in tools]) == expected


def test_agent_with_no_tool_call_is_flagged():
    r = app.orchestrate("hi", None, StubAgent([]), None, [])
    assert r["source"] == "none" and "not backed by a source" in r["note"]


def test_agent_failure_is_reported_not_raised():
    class Boom:
        def invoke(self, _): raise RuntimeError("tavily 401")
    r = app.orchestrate("q", None, Boom(), None, [])
    assert "tool call failed" in r["answer"] and "tavily 401" in r["answer"]


def test_tools_and_agent_construct():
    tools = app.build_tools("tvly-dummy")
    assert [t.name for t in tools] == ["tavily_search_results_json", "wikipedia", "arxiv"]
    assert [t.name for t in app.build_tools(None)] == ["wikipedia", "arxiv"]
    llm = app.ChatOpenAI(api_key="sk-dummy", model="gpt-4o-mini")
    ex = app.build_agent(llm, tools)
    assert ex.return_intermediate_steps is True


# ── persistence, access control, rate limiting ───────────────────────────────
def test_index_is_persisted_and_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "DATA_DIR", tmp_path)
    pdf = make_pdf(PAGES)
    calls = {"n": 0}
    real = app.load_and_index_pdf
    monkeypatch.setattr(app, "load_and_index_pdf", lambda b, e: (calls.__setitem__("n", calls["n"] + 1), real(b, e))[1])

    first = app.index_pdf_cached(pdf, "paper.pdf", KeywordEmbeddings())
    second = app.index_pdf_cached(pdf, "paper.pdf", KeywordEmbeddings())
    assert calls["n"] == 1                       # second call loaded from disk, no re-embed
    d1, s1 = app.retrieve_from_pdf("self attention transformer tokens", first)
    d2, s2 = app.retrieve_from_pdf("self attention transformer tokens", second)
    assert math.isclose(s1, s2, abs_tol=1e-6) and [d.page_content for d in d1] == [d.page_content for d in d2]


def test_last_index_restored_after_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "DATA_DIR", tmp_path)
    app.index_pdf_cached(make_pdf(PAGES), "paper.pdf", KeywordEmbeddings())
    restored = app.load_last_index(KeywordEmbeddings())     # simulates a fresh process
    assert restored is not None and restored[1] == "paper.pdf"


def test_restore_with_no_state_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "DATA_DIR", tmp_path)
    assert app.load_last_index(KeywordEmbeddings()) is None


def test_unwritable_data_dir_falls_back_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "file"; blocker.write_text("x")      # DATA_DIR is a file -> OSError
    monkeypatch.setattr(app, "DATA_DIR", blocker / "sub")
    assert app.index_pdf_cached(make_pdf(PAGES), "p.pdf", KeywordEmbeddings()) is not None


def test_password_check():
    assert app.password_ok("s3cret", "s3cret")
    assert not app.password_ok("s3cret!", "s3cret") and not app.password_ok("", "s3cret")
    assert app.password_ok("pässwörd", "pässwörd")           # non-ASCII must not crash


def test_rate_limiter_blocks_then_recovers():
    t = {"now": 0.0}
    rl = app.RateLimiter(3, 60, clock=lambda: t["now"])
    for _ in range(3):
        assert not rl.blocked("a"); rl.hit("a")
    assert rl.blocked("a") and not rl.blocked("b")             # per-client
    assert 0 < rl.retry_after("a") <= 61
    t["now"] = 61.0
    assert not rl.blocked("a")                                 # window expired


def test_rate_limiter_threadsafe():
    import threading
    rl = app.RateLimiter(1000, 60)
    ts = [threading.Thread(target=lambda: [rl.hit("k") for _ in range(100)]) for _ in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert len(rl._hits["k"]) == 800
