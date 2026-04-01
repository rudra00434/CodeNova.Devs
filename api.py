"""
CodeNova FastAPI wrapper.

Run from the project root with:
    uvicorn api:app --reload
"""

from __future__ import annotations

import difflib
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import utils

DEFAULT_MODEL = (
    list(utils.GROQ_MODELS.keys())[1]
    if len(utils.GROQ_MODELS) > 1
    else next(iter(utils.GROQ_MODELS))
)
HTTP_METHOD = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]

_origins_raw = ["*"]

app = FastAPI(
    title="CodeNova API",
    version="1.0.0",
    description=(
        "Single-file FastAPI backend that wraps the existing CodeNova "
        "Streamlit utilities and exposes them as JSON endpoints."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins_raw,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelConfig(BaseModel):
    model_name: str = DEFAULT_MODEL
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)


class ChatRequest(ModelConfig):
    message: str
    session_id: str | None = None
    save_history: bool = True


class TextRequest(ModelConfig):
    text: str


class ExplainRequest(ModelConfig):
    code: str


class TranslateRequest(ModelConfig):
    code: str
    source_lang: str
    target_lang: str


class RefactorRequest(ModelConfig):
    code: str


class SecurityRequest(ModelConfig):
    code: str


class QualityRequest(ModelConfig):
    code: str


class DebugRequest(ModelConfig):
    traceback: str


class SandboxRequest(BaseModel):
    code: str


class FixCodeRequest(ModelConfig):
    code: str


class ComplexityRequest(ModelConfig):
    code: str


class MonitorRequest(BaseModel):
    name: str
    url: str
    method: Literal["GET", "POST"] = "GET"
    timeout: float = Field(default=10.0, ge=1.0, le=60.0)


class ApiRequestPayload(BaseModel):
    url: str
    method: HTTP_METHOD = "GET"
    headers: dict[str, str] = Field(default_factory=dict)
    body: dict[str, Any] = Field(default_factory=dict)
    bearer_token: str = ""
    timeout_s: float = Field(default=10.0, ge=1.0, le=60.0)
    expected_status: int | None = None


class ApiProfilePayload(BaseModel):
    name: str
    url: str
    method: HTTP_METHOD = "GET"
    token: str = ""
    headers: dict[str, Any] = Field(default_factory=dict)
    body: dict[str, Any] = Field(default_factory=dict)


class UnitTestRequest(ModelConfig):
    language: str
    code: str


class RegexRequest(ModelConfig):
    description: str
    sample_text: str = ""


class SqlRequest(ModelConfig):
    schema: str = ""
    query: str


class DockerRequest(ModelConfig):
    description: str


class CommitMessageRequest(ModelConfig):
    diff: str


class ChangelogRequest(ModelConfig):
    changes: str
    version: str = ""


class SnippetPayload(BaseModel):
    name: str
    language: str
    code: str
    tags: str = ""


@contextmanager
def db_connection():
    conn = sqlite3.connect(utils.DB_PATH, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()


@app.on_event("startup")
def startup() -> None:
    conn = utils.init_db()
    conn.close()


def ensure_api_key() -> None:
    if not utils.API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is missing. Add it to the project's .env file.",
        )


def ensure_model_name(model_name: str) -> str:
    if model_name not in utils.GROQ_MODELS:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unknown model_name.",
                "allowed_models": list(utils.GROQ_MODELS.keys()),
            },
        )
    return model_name


def ensure_language(language: str) -> str:
    if language not in utils.LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unsupported language.",
                "allowed_languages": utils.LANGUAGES,
            },
        )
    return language


def invoke_chain(builder, payload: dict[str, Any], model_name: str, temperature: float = 0.2) -> str:
    ensure_api_key()
    model_name = ensure_model_name(model_name)
    chain = builder(utils.get_llm(model_name, temperature))
    return chain.invoke(payload)


def invoke_prompt(prompt: str, model_name: str, temperature: float = 0.2) -> str:
    ensure_api_key()
    model_name = ensure_model_name(model_name)
    response = utils.get_llm(model_name, temperature).invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


def plotly_to_json(fig) -> dict[str, Any] | None:
    if fig is None:
        return None
    return json.loads(fig.to_json())


def build_curl_command(
    method: str,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any] | None = None,
) -> str:
    parts = [f'curl -X {method} "{url}"']
    for key, value in headers.items():
        parts.append(f'  -H "{key}: {value}"')
    if body:
        parts.append(f"  -d '{json.dumps(body)}'")
    return " \\\n".join(parts)


def serialize_snippet(row: tuple[Any, ...]) -> dict[str, Any]:
    snippet_id, name, language, code, tags, created_at = row
    return {
        "id": snippet_id,
        "name": name,
        "language": language,
        "code": code,
        "tags": tags,
        "created_at": created_at,
        "extension": utils.LANG_EXT.get(language, "txt"),
        "highlight": utils.LANG_HIGHLIGHT.get(language, "text"),
    }


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "CodeNova API",
        "docs": "/docs",
        "health": "/health",
        "features": [
            "chat",
            "code-tools",
            "sandbox",
            "api-suite",
            "generators",
            "snippets",
        ],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "groq_api_key_configured": bool(utils.API_KEY),
        "models": utils.GROQ_MODELS,
        "languages": utils.LANGUAGES,
        "db_path": utils.DB_PATH,
    }


@app.get("/meta")
def meta() -> dict[str, Any]:
    return {
        "default_model": DEFAULT_MODEL,
        "models": utils.GROQ_MODELS,
        "languages": utils.LANGUAGES,
        "language_extensions": utils.LANG_EXT,
        "language_highlights": utils.LANG_HIGHLIGHT,
    }


@app.get("/sessions")
def get_sessions() -> dict[str, Any]:
    with db_connection() as conn:
        rows = utils.list_sessions(conn)
    return {
        "sessions": [
            {"session_id": session_id, "created_at": created_at}
            for session_id, created_at in rows
        ]
    }


@app.get("/sessions/{session_id}/history")
def get_session_history(session_id: str) -> dict[str, Any]:
    with db_connection() as conn:
        history = utils.load_chat_history(conn, session_id)
    return {"session_id": session_id, "history": history}


@app.delete("/sessions/{session_id}")
def delete_session_history(session_id: str) -> dict[str, Any]:
    with db_connection() as conn:
        utils.clear_chat_history(conn, session_id)
    return {"ok": True, "session_id": session_id}


@app.get("/sessions/{session_id}/export.pdf")
def export_session_pdf(session_id: str):
    with db_connection() as conn:
        history = utils.load_chat_history(conn, session_id)
    if not history:
        raise HTTPException(status_code=404, detail="No chat history found for that session.")

    pdf_buffer = utils.export_chat_pdf(history, session_id)
    headers = {
        "Content-Disposition": f'attachment; filename="codenova_{session_id}.pdf"'
    }
    return StreamingResponse(pdf_buffer, media_type="application/pdf", headers=headers)


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message is required.")

    session_id = request.session_id or uuid.uuid4().hex[:8]
    answer = invoke_chain(
        utils.build_chat_chain,
        {"question": request.message},
        request.model_name,
        request.temperature,
    )

    history: list[dict[str, str]]
    if request.save_history:
        with db_connection() as conn:
            utils.save_chat_message(
                conn, session_id, "user", request.message, request.model_name
            )
            utils.save_chat_message(
                conn, session_id, "assistant", answer, request.model_name
            )
            history = utils.load_chat_history(conn, session_id)
    else:
        history = [
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": answer},
        ]

    return {
        "session_id": session_id,
        "model_name": request.model_name,
        "answer": answer,
        "history": history,
    }


@app.post("/tts")
def text_to_speech(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text is required.")
    buffer = utils.text_to_speech(request.text)
    if buffer is None:
        raise HTTPException(status_code=503, detail="Unable to generate speech audio.")
    headers = {"Content-Disposition": 'attachment; filename="codenova-response.mp3"'}
    return StreamingResponse(buffer, media_type="audio/mpeg", headers=headers)


@app.post("/tools/explain")
def explain_code(request: ExplainRequest) -> dict[str, Any]:
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="code is required.")
    explanation = invoke_chain(
        utils.build_explain_chain,
        {"code": request.code},
        request.model_name,
        request.temperature,
    )
    return {"explanation": explanation}


@app.post("/tools/translate")
def translate_code(request: TranslateRequest) -> dict[str, Any]:
    ensure_language(request.source_lang)
    ensure_language(request.target_lang)
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="code is required.")

    translated = invoke_chain(
        utils.build_translate_chain,
        {
            "source_lang": request.source_lang,
            "target_lang": request.target_lang,
            "code": request.code,
        },
        request.model_name,
        request.temperature,
    )
    return {
        "translated_code": translated,
        "source_lang": request.source_lang,
        "target_lang": request.target_lang,
        "extension": utils.LANG_EXT.get(request.target_lang, "txt"),
        "highlight": utils.LANG_HIGHLIGHT.get(request.target_lang, "text"),
    }


@app.post("/tools/refactor")
def refactor_code(request: RefactorRequest) -> dict[str, Any]:
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="code is required.")

    refactored = invoke_chain(
        utils.build_refactor_chain,
        {"code": request.code},
        request.model_name,
        request.temperature,
    )
    unified_diff = "\n".join(
        difflib.unified_diff(
            request.code.splitlines(),
            refactored.splitlines(),
            fromfile="original",
            tofile="refactored",
            lineterm="",
        )
    )
    return {
        "refactored_code": refactored,
        "unified_diff": unified_diff,
        "diff_html": utils.compute_diff_html(request.code, refactored),
    }


@app.post("/tools/security")
def security_scan(request: SecurityRequest) -> dict[str, Any]:
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="code is required.")
    findings = invoke_chain(
        utils.build_security_chain,
        {"code": request.code},
        request.model_name,
        request.temperature,
    )
    return {
        "raw": findings,
        "findings": [line.strip() for line in findings.splitlines() if line.strip()],
    }


@app.post("/tools/quality")
def quality_score(request: QualityRequest) -> dict[str, Any]:
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="code is required.")

    raw = invoke_chain(
        utils.build_quality_chain,
        {"code": request.code},
        request.model_name,
        request.temperature,
    )
    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        scores = json.loads(clean)
    except json.JSONDecodeError:
        return {"parsed": False, "raw": raw}

    return {
        "parsed": True,
        "scores": scores,
        "chart": plotly_to_json(utils.plot_quality_radar(scores)),
    }


@app.post("/tools/debug")
def debug_traceback(request: DebugRequest) -> dict[str, Any]:
    if not request.traceback.strip():
        raise HTTPException(status_code=400, detail="traceback is required.")
    diagnosis = invoke_chain(
        utils.build_error_chain,
        {"traceback": request.traceback},
        request.model_name,
        request.temperature,
    )
    return {"diagnosis": diagnosis}


@app.post("/sandbox/run")
def run_sandbox(request: SandboxRequest) -> dict[str, Any]:
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="code is required.")
    return utils.run_python_sandbox(request.code)


@app.post("/sandbox/fix")
def fix_broken_code(request: FixCodeRequest) -> dict[str, Any]:
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="code is required.")

    prompt = (
        "You are an expert code debugger. Fix the broken code below. "
        "Structure your response EXACTLY like this:\n"
        "## What was wrong\n"
        "- bullet 1\n- bullet 2\n\n"
        "## Fixed Code\n"
        "```python\n<fixed code here>\n```\n\n"
        "## What changed\n"
        "Brief explanation of each fix.\n\n"
        f"Broken code:\n```\n{request.code}\n```"
    )
    fixed = invoke_prompt(prompt, request.model_name, request.temperature)
    return {"response": fixed}


@app.post("/sandbox/complexity")
def analyze_complexity(request: ComplexityRequest) -> dict[str, Any]:
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="code is required.")

    analysis = invoke_chain(
        utils.build_complexity_chain,
        {"code": request.code},
        request.model_name,
        request.temperature,
    )
    time_big_o = utils.parse_big_o(analysis.split("Time")[1] if "Time" in analysis else analysis)
    space_big_o = utils.parse_big_o(analysis.split("Space")[1] if "Space" in analysis else "")
    chart = utils.plot_complexity_curve(time_big_o) if time_big_o else None
    return {
        "analysis": analysis,
        "time_complexity": time_big_o,
        "space_complexity": space_big_o,
        "chart": plotly_to_json(chart),
    }


@app.post("/http/request")
def send_api_request(request: ApiRequestPayload) -> dict[str, Any]:
    headers = dict(request.headers)
    if request.bearer_token:
        headers["Authorization"] = f"Bearer {request.bearer_token}"

    json_body = request.body if request.method != "GET" and request.body else None
    start = time.perf_counter()
    try:
        with httpx.Client(timeout=request.timeout_s, follow_redirects=True) as client:
            response = client.request(
                method=request.method,
                url=request.url,
                headers=headers,
                json=json_body,
            )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Request failed: {exc}") from exc

    response_time_ms = round((time.perf_counter() - start) * 1000, 1)
    try:
        response_json = response.json()
        response_text = None
    except Exception:
        response_json = None
        response_text = response.text

    expected_match = None
    if request.expected_status is not None:
        expected_match = request.expected_status == response.status_code

    return {
        "status_code": response.status_code,
        "ok": response.status_code < 400,
        "response_time_ms": response_time_ms,
        "headers": dict(response.headers),
        "body_json": response_json,
        "body_text": response_text,
        "expected_status": request.expected_status,
        "expected_status_matched": expected_match,
        "curl": build_curl_command(request.method, request.url, headers, json_body),
    }


@app.post("/monitor")
def monitor_endpoint(request: MonitorRequest) -> dict[str, Any]:
    return utils.monitor_api(
        request.name,
        request.url,
        request.method,
        request.timeout,
    )


@app.get("/api-profiles")
def get_api_profiles() -> dict[str, Any]:
    with db_connection() as conn:
        profiles = utils.load_api_profiles(conn)
    return {"profiles": profiles}


@app.post("/api-profiles")
def save_api_profile(request: ApiProfilePayload) -> dict[str, Any]:
    with db_connection() as conn:
        utils.save_api_profile(
            conn,
            request.name,
            {
                "url": request.url,
                "method": request.method,
                "token": request.token,
                "headers": request.headers,
                "body": request.body if request.method != "GET" else {},
            },
        )
        profiles = utils.load_api_profiles(conn)
    return {"ok": True, "profiles": profiles}


@app.delete("/api-profiles/{name}")
def delete_api_profile(name: str) -> dict[str, Any]:
    with db_connection() as conn:
        utils.delete_api_profile(conn, name)
        profiles = utils.load_api_profiles(conn)
    return {"ok": True, "profiles": profiles}


@app.post("/generate/unit-tests")
def generate_unit_tests(request: UnitTestRequest) -> dict[str, Any]:
    ensure_language(request.language)
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="code is required.")

    tests = invoke_chain(
        lambda llm: utils.build_unit_test_chain(llm, request.language),
        {"func": request.code},
        request.model_name,
        request.temperature,
    )
    return {
        "language": request.language,
        "tests": tests,
        "extension": utils.LANG_EXT.get(request.language, "txt"),
        "highlight": utils.LANG_HIGHLIGHT.get(request.language, "text"),
    }


@app.post("/generate/regex")
def generate_regex(request: RegexRequest) -> dict[str, Any]:
    if not request.description.strip():
        raise HTTPException(status_code=400, detail="description is required.")

    raw_pattern = invoke_prompt(
        "Return ONLY a Python regular-expression pattern, no description, no fences:\n"
        + request.description,
        request.model_name,
        request.temperature,
    )
    pattern = utils.clean_regex_pattern(raw_pattern)
    explanation = invoke_chain(
        utils.build_regex_explain_chain,
        {"pattern": pattern},
        request.model_name,
        request.temperature,
    )

    matches: list[Any] = []
    regex_error = None
    if request.sample_text:
        try:
            compiled = __import__("re").compile(pattern)
            matches = compiled.findall(request.sample_text)
        except Exception as exc:
            regex_error = str(exc)

    return {
        "pattern": pattern,
        "explanation": explanation,
        "sample_text": request.sample_text,
        "matches": matches,
        "match_count": len(matches),
        "regex_error": regex_error,
    }


@app.post("/generate/sql")
def generate_sql(request: SqlRequest) -> dict[str, Any]:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query is required.")
    sql = invoke_chain(
        utils.build_sql_chain,
        {"schema": request.schema, "query": request.query},
        request.model_name,
        request.temperature,
    )
    return {"sql": sql}


@app.post("/generate/dockerfile")
def generate_dockerfile(request: DockerRequest) -> dict[str, Any]:
    if not request.description.strip():
        raise HTTPException(status_code=400, detail="description is required.")
    dockerfile = invoke_chain(
        utils.build_docker_chain,
        {"description": request.description},
        request.model_name,
        request.temperature,
    )
    return {"dockerfile": dockerfile}


@app.post("/generate/commit-message")
def generate_commit_message(request: CommitMessageRequest) -> dict[str, Any]:
    if not request.diff.strip():
        raise HTTPException(status_code=400, detail="diff is required.")
    commit_message = invoke_chain(
        utils.build_git_commit_chain,
        {"diff": request.diff},
        request.model_name,
        request.temperature,
    )
    return {"commit_message": commit_message}


@app.post("/generate/changelog")
def generate_changelog(request: ChangelogRequest) -> dict[str, Any]:
    if not request.changes.strip():
        raise HTTPException(status_code=400, detail="changes is required.")
    full_input = (f"Version: {request.version}\n\n" if request.version else "") + request.changes
    changelog = invoke_chain(
        utils.build_changelog_chain,
        {"changes": full_input},
        request.model_name,
        request.temperature,
    )
    return {"changelog": changelog}


@app.get("/snippets")
def get_snippets(search: str = Query(default="")) -> dict[str, Any]:
    with db_connection() as conn:
        rows = utils.load_snippets(conn)
    snippets = [serialize_snippet(row) for row in rows]
    if search.strip():
        q = search.lower()
        snippets = [
            snippet for snippet in snippets
            if q in snippet["name"].lower() or q in (snippet["tags"] or "").lower()
        ]
    return {"snippets": snippets}


@app.post("/snippets")
def save_snippet(request: SnippetPayload) -> dict[str, Any]:
    ensure_language(request.language)
    if not request.name.strip() or not request.code.strip():
        raise HTTPException(status_code=400, detail="name and code are required.")

    with db_connection() as conn:
        utils.save_snippet(
            conn,
            request.name.strip(),
            request.language,
            request.code.strip(),
            request.tags.strip(),
        )
        rows = utils.load_snippets(conn)
    return {"ok": True, "snippets": [serialize_snippet(row) for row in rows]}


@app.delete("/snippets/{snippet_id}")
def delete_snippet(snippet_id: int) -> dict[str, Any]:
    with db_connection() as conn:
        utils.delete_snippet(conn, snippet_id)
        rows = utils.load_snippets(conn)
    return {"ok": True, "snippets": [serialize_snippet(row) for row in rows]}
