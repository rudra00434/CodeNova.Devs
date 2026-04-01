"""
CodeNova — utils.py
All backend utilities: database, LLM chains, helpers, visualizations.
"""

import sqlite3
import json
import os
import re
import math
import time
import io
import difflib
from datetime import datetime
from io import BytesIO
from contextlib import redirect_stdout

import numpy as np
import plotly.graph_objects as go
from fpdf import FPDF
import httpx
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
DB_PATH = os.path.join(BASE_DIR, "codenova.db")

load_dotenv(ENV_PATH if os.path.exists(ENV_PATH) else None)
load_dotenv()

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

API_KEY = os.getenv("GROQ_API_KEY", "")

GROQ_MODELS = {
    "⚡ LLaMA 3.1 8B  (Fast)":     "llama-3.1-8b-instant",
    "🧠 LLaMA 3.3 70B (Smart)":    "llama-3.3-70b-versatile",
    "🔬 Command R+    (Advanced)":  "command-r-plus",
}

LANGUAGES = [
    "Python", "Java", "JavaScript", "TypeScript",
    "C++", "Go", "Rust", "C#", "Ruby", "PHP",
    "Swift", "Kotlin", "React", "Next.js", "Node.js",
]

LANG_EXT = {
    "Python": "py",   "Java": "java",  "JavaScript": "js",
    "TypeScript": "ts","C++": "cpp",   "Go": "go",
    "Rust": "rs",     "C#": "cs",      "Ruby": "rb",
    "PHP": "php",     "Swift": "swift","Kotlin": "kt",
    "React": "jsx",   "Next.js": "js", "Node.js": "js",
}

LANG_HIGHLIGHT = {
    "Python": "python",       "Java": "java",         "JavaScript": "javascript",
    "TypeScript": "typescript","C++": "cpp",           "Go": "go",
    "Rust": "rust",           "C#": "csharp",          "Ruby": "ruby",
    "PHP": "php",             "Swift": "swift",         "Kotlin": "kotlin",
    "React": "jsx",           "Next.js": "javascript",  "Node.js": "javascript",
}

UNIT_TEST_PROMPTS = {
    "Python":     "You are a Python test engineer. Generate comprehensive unit tests using pytest for the function. Cover normal, edge cases, and exceptions. Return ONLY the test code.",
    "Java":       "You are a Java developer. Generate JUnit 5 test cases with @Test, descriptive names, and assertions. Return ONLY the test code.",
    "JavaScript": "You are a JS developer. Use Jest: describe/it/expect. Return ONLY the test code.",
    "TypeScript": "You are a TypeScript developer. Use Jest + TypeScript typed tests. Return ONLY the test code.",
    "C++":        "You are a C++ developer. Use Google Test (gtest) with TEST() macros. Return ONLY the test code.",
    "Go":         "You are a Go developer. Use table-driven tests with testing.T and t.Run. Return ONLY the test code.",
    "Rust":       "You are a Rust developer. Use #[cfg(test)] and #[test]. Return ONLY the test code.",
    "C#":         "You are a C# developer. Use NUnit with [Test] attributes. Return ONLY the test code.",
    "React":      "You are a frontend dev. Use React Testing Library + Jest. Test render, props, events. Return ONLY the test code.",
    "Next.js":    "You are a Next.js developer. Use Jest + React Testing Library. Test components and API routes. Return ONLY the test code.",
    "Node.js":    "You are a Node.js developer. Use Jest or Mocha+Chai with describe/it/expect. Return ONLY the test code.",
    "Ruby":       "You are a Ruby developer. Use RSpec with describe/context/it blocks. Return ONLY the test code.",
    "Swift":      "You are a Swift developer. Use XCTest with XCTAssert methods. Return ONLY the test code.",
    "Kotlin":     "You are a Kotlin developer. Use JUnit 5 with Kotlin idioms. Return ONLY the test code.",
    "PHP":        "You are a PHP developer. Use PHPUnit with @test annotations. Return ONLY the test code.",
}

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role       TEXT,
            content    TEXT,
            model      TEXT DEFAULT '',
            timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS snippets (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT UNIQUE,
            language   TEXT,
            code       TEXT,
            tags       TEXT DEFAULT '',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS api_profiles (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT UNIQUE,
            data       TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    return conn


def load_chat_history(conn, session_id):
    c = conn.cursor()
    c.execute(
        "SELECT role, content FROM chat_history WHERE session_id=? ORDER BY id ASC",
        (session_id,)
    )
    return [{"role": r, "content": ct} for r, ct in c.fetchall()]


def save_chat_message(conn, session_id, role, content, model=""):
    c = conn.cursor()
    c.execute(
        "INSERT INTO chat_history (session_id, role, content, model) VALUES (?,?,?,?)",
        (session_id, role, content, model)
    )
    conn.commit()


def clear_chat_history(conn, session_id):
    c = conn.cursor()
    c.execute("DELETE FROM chat_history WHERE session_id=?", (session_id,))
    conn.commit()


def list_sessions(conn):
    c = conn.cursor()
    c.execute(
        "SELECT session_id, MIN(timestamp) FROM chat_history GROUP BY session_id ORDER BY MIN(timestamp) DESC"
    )
    return c.fetchall()


def save_snippet(conn, name, language, code, tags=""):
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO snippets (name, language, code, tags) VALUES (?,?,?,?)",
        (name, language, code, tags)
    )
    conn.commit()


def load_snippets(conn):
    c = conn.cursor()
    c.execute("SELECT id, name, language, code, tags, created_at FROM snippets ORDER BY created_at DESC")
    return c.fetchall()


def delete_snippet(conn, snippet_id):
    c = conn.cursor()
    c.execute("DELETE FROM snippets WHERE id=?", (snippet_id,))
    conn.commit()


def save_api_profile(conn, name, data: dict):
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO api_profiles (name, data) VALUES (?,?)",
        (name, json.dumps(data))
    )
    conn.commit()


def load_api_profiles(conn) -> dict:
    c = conn.cursor()
    c.execute("SELECT name, data FROM api_profiles ORDER BY created_at DESC")
    return {name: json.loads(data) for name, data in c.fetchall()}


def delete_api_profile(conn, name):
    c = conn.cursor()
    c.execute("DELETE FROM api_profiles WHERE name=?", (name,))
    conn.commit()


# ─────────────────────────────────────────────
# LLM FACTORY
# ─────────────────────────────────────────────

def get_llm(model_name: str, temperature: float = 0.2) -> ChatGroq:
    model_id = GROQ_MODELS.get(model_name, "llama-3.3-70b-versatile")
    return ChatGroq(api_key=API_KEY, model_name=model_id, temperature=temperature)


# ─────────────────────────────────────────────
# LLM CHAINS
# ─────────────────────────────────────────────

def build_chat_chain(llm):
    p = ChatPromptTemplate.from_messages([
        ("system",
         "You are CodeNova — an elite & powerful AI coding assistant. "
         "For code requests: return clean, production-ready code in markdown fenced blocks with language tags.and also give a brief explanation of the code in bullet points."
         "For conceptual questions: be precise, developer-friendly, and concise. "
         "Never pad responses. Skip filler phrases like 'Great question!'."),
        ("user", "{question}"),
    ])
    return p | llm | StrOutputParser()


def build_explain_chain(llm):
    p = ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior code reviewer. Explain the code line-by-line with clarity, then give a high-level summary. "
         "Use plain language. Do NOT reprint the code. Output only the explanation."),
        ("user", "{code}"),
    ])
    return p | llm | StrOutputParser()


def build_generate_chain(llm):
    p = ChatPromptTemplate.from_messages([
        ("system",
         "You are a professional software engineer. Write clean, idiomatic, production-quality code from the description. "
         "Auto-detect the best language unless specified. Return ONLY the code, no markdown fences."),
        ("user", "{desc}"),
    ])
    return p | llm | StrOutputParser()


def build_error_chain(llm):
    p = ChatPromptTemplate.from_messages([
        ("system",
         "You are a debugging expert. Analyze the error/traceback: explain what it means in simple terms, "
         "identify the root cause, and give specific fix steps. Output only diagnosis and solution — do NOT reprint the traceback."),
        ("user", "{traceback}"),
    ])
    return p | llm | StrOutputParser()


def build_complexity_chain(llm):
    p = ChatPromptTemplate.from_template("""
You are a senior algorithm engineer. Analyze this code.

Code:
{code}

Respond EXACTLY in this format (no extra text):
Time Complexity: O(...)
Space Complexity: O(...)
Explanation: (1-2 sentences)

Allowed Big-O strings: O(1), O(log n), O(n), O(n log n), O(n^2), O(n^3), O(2^n), O(n!)
""")
    return p | llm | StrOutputParser()


def build_translate_chain(llm):
    p = ChatPromptTemplate.from_messages([
        ("system",
         "You are a polyglot programmer. Translate the code from {source_lang} to {target_lang} exactly and idiomatically. "
         "Preserve logic. Add brief comments only where the translation requires clarification. "
         "Return ONLY the translated code, no markdown fences."),
        ("user", "{code}"),
    ])
    return p | llm | StrOutputParser()


def build_sql_chain(llm):
    p = ChatPromptTemplate.from_messages([
        ("system",
         "You are a SQL expert. Convert the natural-language query to valid, optimized SQL. "
         "Use the schema if provided. Return ONLY the SQL query."),
        ("user", "Schema: {schema}\n\nQuery: {query}"),
    ])
    return p | llm | StrOutputParser()


def build_security_chain(llm):
    p = ChatPromptTemplate.from_messages([
        ("system",
         "You are an application security engineer (AppSec). Analyze code for vulnerabilities. "
         "For EACH finding output: [SEVERITY] CATEGORY: Description + line hint + recommended fix. "
         "Severity: CRITICAL / HIGH / MEDIUM / LOW / INFO. "
         "If nothing found, say 'No obvious vulnerabilities detected.' "
         "Output only findings."),
        ("user", "{code}"),
    ])
    return p | llm | StrOutputParser()


def build_refactor_chain(llm):
    p = ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior software engineer doing a thorough code review. "
         "Refactor the code for readability, performance, and best practices. "
         "Return ONLY the refactored code. No explanations. No markdown fences."),
        ("user", "{code}"),
    ])
    return p | llm | StrOutputParser()


def build_quality_chain(llm):
    p = ChatPromptTemplate.from_template("""
You are a code quality reviewer. Score this code on each criterion from 0 to 10:

Code:
{code}

Respond ONLY with valid JSON (no markdown, no extra text):
{{"readability": 7, "efficiency": 8, "error_handling": 5, "best_practices": 7, "documentation": 4, "overall": 6, "summary": "One or two sentences on main strengths and weaknesses."}}
""")
    return p | llm | StrOutputParser()


def build_git_commit_chain(llm):
    p = ChatPromptTemplate.from_messages([
        ("system",
         "You are a Git expert. Generate a concise conventional commit message (Conventional Commits spec) "
         "from the provided diff or change description. "
         "Format: type(scope): short description\\n\\nOptional body.\\n\\nReturn ONLY the commit message."),
        ("user", "{diff}"),
    ])
    return p | llm | StrOutputParser()


def build_docker_chain(llm):
    p = ChatPromptTemplate.from_messages([
        ("system",
         "You are a DevOps engineer. Generate a production-ready Dockerfile for the project described. "
         "Use multi-stage builds where appropriate. Follow best practices: non-root user, minimal base image, "
         "layer caching, .dockerignore hints as comments. Return ONLY the Dockerfile content."),
        ("user", "{description}"),
    ])
    return p | llm | StrOutputParser()


def build_unit_test_chain(llm, lang: str):
    system_prompt = UNIT_TEST_PROMPTS.get(lang, UNIT_TEST_PROMPTS["Python"])
    p = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{func}"),
    ])
    return p | llm | StrOutputParser()


def build_regex_explain_chain(llm):
    p = ChatPromptTemplate.from_template(
        "Explain briefly what this regex pattern does (under 5 lines, no code fences): {pattern}"
    )
    return p | llm | StrOutputParser()


def build_changelog_chain(llm):
    p = ChatPromptTemplate.from_messages([
        ("system",
         "You are a technical writer. Generate a clean, professional CHANGELOG entry in Keep-a-Changelog format "
         "from the list of changes. Group items under Added / Changed / Fixed / Removed where applicable. "
         "Return ONLY the changelog entry markdown."),
        ("user", "{changes}"),
    ])
    return p | llm | StrOutputParser()


# ─────────────────────────────────────────────
# PYTHON SANDBOX
# ─────────────────────────────────────────────

def run_python_sandbox(code: str) -> dict:
    stdout_buf = io.StringIO()
    result = {"output": "", "error": "", "success": False, "exec_time_ms": 0}
    start = time.perf_counter()
    try:
        safe_globals = {"__builtins__": __builtins__, "__name__": "__main__"}
        with redirect_stdout(stdout_buf):
            exec(compile(code, "<sandbox>", "exec"), safe_globals)  # noqa: S102
        result["success"] = True
        result["output"] = stdout_buf.getvalue()
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["exec_time_ms"] = round((time.perf_counter() - start) * 1000, 2)
    return result


# ─────────────────────────────────────────────
# COMPLEXITY VISUALIZATION
# ─────────────────────────────────────────────

_COMPLEXITY_FN = {
    "O(1)":      lambda x: np.ones_like(x),
    "O(log n)":  lambda x: np.log2(np.maximum(x, 1e-9)),
    "O(n)":      lambda x: x,
    "O(n log n)":lambda x: x * np.log2(np.maximum(x, 1e-9)),
    "O(n^2)":    lambda x: x ** 2,
    "O(n^3)":    lambda x: x ** 3,
    "O(2^n)":    lambda x: np.clip(2.0 ** x, 0, 1e15),
    "O(n!)":     lambda x: np.array([math.factorial(int(i)) if i < 18 else float("nan") for i in x]),
}

_COMPLEXITY_COLORS = [
    "#00ff88", "#00d4ff", "#ffcc00", "#ff9900",
    "#ff5533", "#cc33ff", "#ff0055", "#880000",
]


def parse_big_o(text: str) -> str | None:
    """Extract and normalise a Big-O token from LLM output."""
    m = re.search(r"O\([^)]+\)", text, re.IGNORECASE)
    if not m:
        return None
    raw = m.group().replace(" ", "").lower()
    mapping = {
        "o(1)": "O(1)", "o(logn)": "O(log n)", "o(n)": "O(n)",
        "o(nlogn)": "O(n log n)", "o(n^2)": "O(n^2)", "o(n^3)": "O(n^3)",
        "o(2^n)": "O(2^n)", "o(n!)": "O(n!)",
    }
    return mapping.get(raw)


def plot_complexity_curve(big_o: str) -> go.Figure | None:
    if big_o not in _COMPLEXITY_FN:
        return None
    x = np.linspace(1, 50, 300)
    fig = go.Figure()
    for i, (label, fn) in enumerate(_COMPLEXITY_FN.items()):
        y = fn(x)
        is_target = label == big_o
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name=label,
            line=dict(color=_COMPLEXITY_COLORS[i], width=3 if is_target else 1),
            opacity=1.0 if is_target else 0.22,
        ))
    fig.update_layout(
        title=f"Growth Curve: {big_o}  (highlighted)",
        xaxis_title="Input Size (n)",
        yaxis_title="Operations (log scale)",
        yaxis=dict(type="log", range=[0, 9]),
        height=420,
        plot_bgcolor="#0a0a0f",
        paper_bgcolor="#0a0a0f",
        font=dict(color="#e2e8f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
    )
    return fig


# ─────────────────────────────────────────────
# CODE QUALITY RADAR
# ─────────────────────────────────────────────

def plot_quality_radar(scores: dict) -> go.Figure:
    categories   = ["Readability", "Efficiency", "Error Handling", "Best Practices", "Documentation"]
    keys         = ["readability", "efficiency", "error_handling", "best_practices", "documentation"]
    values       = [scores.get(k, 5) for k in keys]
    values_closed = values + [values[0]]
    cats_closed   = categories + [categories[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed, theta=cats_closed, fill="toself",
        fillcolor="rgba(0,212,255,0.18)",
        line=dict(color="#00d4ff", width=2.5),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], color="#94a3b8", gridcolor="#1e293b"),
            bgcolor="#0a0a0f",
            angularaxis=dict(color="#94a3b8"),
        ),
        showlegend=False, height=340,
        paper_bgcolor="#0a0a0f",
        font=dict(color="#e2e8f0"),
    )
    return fig


# ─────────────────────────────────────────────
# CODE DIFF (for refactor view)
# ─────────────────────────────────────────────

def compute_diff_html(original: str, refactored: str) -> str:
    """Return an HTML side-by-side diff table."""
    differ = difflib.HtmlDiff(wrapcolumn=80)
    return differ.make_table(
        original.splitlines(),
        refactored.splitlines(),
        fromdesc="Original",
        todesc="Refactored",
        context=True,
        numlines=3,
    )


# ─────────────────────────────────────────────
# API HEALTH MONITOR
# ─────────────────────────────────────────────

def monitor_api(name: str, url: str, method: str = "GET", timeout: float = 10.0) -> dict:
    try:
        start = time.perf_counter()
        with httpx.Client(timeout=timeout) as client:
            resp = client.request(method=method, url=url)
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        return {
            "name": name, "url": url, "method": method,
            "status_code": resp.status_code,
            "response_time_ms": duration_ms,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ok": resp.status_code < 400, "error": None,
        }
    except Exception as exc:
        return {
            "name": name, "url": url, "method": method,
            "status_code": None, "response_time_ms": None,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ok": False, "error": str(exc),
        }


# ─────────────────────────────────────────────
# PDF EXPORT
# ─────────────────────────────────────────────

def export_chat_pdf(history: list, session_id: str = "") -> BytesIO:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, "CodeNova — Chat Export", ln=True, align="C")
    pdf.set_font("Arial", size=9)
    pdf.cell(0, 7, f"Session: {session_id}  |  Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             ln=True, align="C")
    pdf.ln(4)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    for msg in history:
        role = msg["role"].upper()
        pdf.set_font("Arial", "B", 10)
        pdf.set_fill_color(200, 220, 255) if role == "USER" else pdf.set_fill_color(210, 255, 220)
        pdf.cell(0, 7, f"  {role}", ln=True, fill=True)
        pdf.set_font("Courier", size=9)
        for line in msg["content"].split("\n"):
            safe = line.encode("latin-1", "replace").decode("latin-1")
            pdf.multi_cell(0, 5, safe)
        pdf.ln(3)

    buf = BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────
# REGEX HELPERS
# ─────────────────────────────────────────────

def clean_regex_pattern(text: str) -> str:
    txt = text.strip()
    if txt.startswith("```"):
        parts = txt.split("```")
        txt = parts[1].strip() if len(parts) >= 3 else txt.replace("```", "").strip()
    for start, end in [("r'", "'"), ('r"', '"'), ("'", "'"), ('"', '"'), ("/", "/")]:
        if txt.startswith(start) and txt.endswith(end) and len(txt) > len(start) + len(end):
            txt = txt[len(start):-len(end)]
            break
    if txt.lower().startswith("python"):
        txt = txt[6:].strip()
    return txt.strip("` ;")


# ─────────────────────────────────────────────
# VOICE TTS
# ─────────────────────────────────────────────

def text_to_speech(text: str, lang: str = "en") -> BytesIO | None:
    try:
        from gtts import gTTS
        tts = gTTS(text=text[:3000], lang=lang)
        buf = BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf
    except Exception:
        return None
