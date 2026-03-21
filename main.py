"""
CodeNova — main.py
AI-powered developer suite.  Run with: streamlit run main.py
Requires: utils.py  |  .env with GROQ_API_KEY=...
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import json
import os
import re
import time
import uuid
from datetime import datetime
from io import BytesIO

# ── third-party ───────────────────────────────────────────────────────────────
import streamlit as st
import httpx

# ── local ─────────────────────────────────────────────────────────────────────
import utils

# ═════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be the very first Streamlit call)
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CodeNova — AI Dev Suite",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS  — terminal-glass aesthetic
# ═════════════════════════════════════════════════════════════════════════════
import streamlit.components.v1 as _components

_FONT = "https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;800&display=swap"
_components.html(f'<link href="{_FONT}" rel="stylesheet">', height=0)

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800;900&display=swap');

/* ── RESET ─────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
    color: #e2e8f0 !important;
}

/* ── ANIMATED MESH BACKGROUND ──────────────────────────── */
.stApp {
    background-color: #05060f !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(0,212,255,.13) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(124,58,237,.15) 0%, transparent 60%),
        radial-gradient(ellipse 50% 30% at 50% 50%, rgba(16,185,129,.05) 0%, transparent 70%);
}
.main .block-container {
    padding-top: 1.6rem; padding-bottom: 3rem;
    max-width: 1140px;
}

/* ── SCROLLBAR ──────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #00d4ff, #7c3aed);
    border-radius: 99px;
}

/* ── SIDEBAR ────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(170deg, #07091a 0%, #0b0e20 60%, #08091a 100%) !important;
    border-right: 1px solid rgba(0,212,255,.12) !important;
    box-shadow: 4px 0 30px rgba(0,0,0,.6) !important;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] .stButton button {
    background: rgba(255,255,255,.04) !important;
    border: 1px solid rgba(255,255,255,.08) !important;
    border-radius: 10px !important;
    color: #64748b !important;
    font-size: 0.79rem !important;
    font-family: 'Space Mono', monospace !important;
    transition: all .2s ease !important;
    text-align: left !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(0,212,255,.08) !important;
    border-color: rgba(0,212,255,.3) !important;
    color: #00d4ff !important;
    transform: translateX(3px) !important;
}

/* ── KEYFRAMES ──────────────────────────────────────────── */
@keyframes gradShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; } to { opacity: 1; }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 20px rgba(0,212,255,.2), 0 0 40px rgba(124,58,237,.1); }
    50%       { box-shadow: 0 0 35px rgba(0,212,255,.4), 0 0 60px rgba(124,58,237,.2); }
}
@keyframes scanline {
    0%   { transform: translateY(-100%); }
    100% { transform: translateY(100vh); }
}
@keyframes borderGlow {
    0%, 100% { border-color: rgba(0,212,255,.2); }
    50%       { border-color: rgba(0,212,255,.5); }
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-6px); }
}

/* ── HERO TITLE ─────────────────────────────────────────── */
.nova-title {
    font-family: 'Syne', sans-serif;
    font-weight: 900;
    font-size: 3.2rem;
    background: linear-gradient(270deg, #00d4ff, #a855f7, #10b981, #f59e0b, #ef4444, #00d4ff);
    background-size: 600% 600%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradShift 4s ease infinite, fadeUp .5s ease both;
    text-align: center;
    letter-spacing: -2px;
    line-height: 1;
    margin-bottom: 2px;
    filter: drop-shadow(0 0 30px rgba(0,212,255,.3));
}
.nova-sub {
    text-align: center;
    color: #334155 !important;
    font-size: 0.78rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    margin-top: 8px;
    animation: fadeUp .7s .1s ease both;
    font-family: 'Space Mono', monospace !important;
}

/* ── PAGE BANNER (per-page vibrant headers) ─────────────── */
.page-banner {
    position: relative;
    border-radius: 18px;
    padding: 24px 28px;
    margin-bottom: 24px;
    overflow: hidden;
    animation: fadeUp .4s ease both;
}
.page-banner::before {
    content: '';
    position: absolute; inset: 0;
    background: linear-gradient(135deg, var(--banner-a) 0%, var(--banner-b) 100%);
    opacity: .12;
}
.page-banner::after {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 30% 30%, var(--banner-a) 0%, transparent 50%);
    opacity: .07;
}
.page-banner-border {
    position: absolute; inset: 0;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,.1);
    background: transparent;
}
.page-banner-content { position: relative; z-index: 1; }
.page-banner-icon {
    font-size: 2.2rem;
    display: block;
    margin-bottom: 6px;
    animation: float 3s ease infinite;
}
.page-banner-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.7rem;
    letter-spacing: -.5px;
    background: linear-gradient(90deg, var(--banner-a), var(--banner-b));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}
.page-banner-desc {
    font-size: .82rem;
    color: #64748b;
    margin-top: 4px;
    font-family: 'Space Mono', monospace;
}

/* ── GLASS CARDS ────────────────────────────────────────── */
.glass {
    background: linear-gradient(135deg, rgba(0,212,255,.04) 0%, rgba(124,58,237,.04) 100%);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(0,212,255,.12);
    border-radius: 16px;
    padding: 18px 22px;
    margin-bottom: 12px;
    box-shadow: 0 4px 24px rgba(0,0,0,.4), inset 0 1px 0 rgba(255,255,255,.05);
    transition: all .25s ease;
}
.glass:hover {
    border-color: rgba(0,212,255,.28);
    box-shadow: 0 8px 32px rgba(0,0,0,.5), 0 0 20px rgba(0,212,255,.08);
    transform: translateY(-2px);
}

/* ── SECTION HEADERS ────────────────────────────────────── */
.sec-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.26em;
    text-transform: uppercase;
    padding: 6px 16px 6px 12px;
    border-radius: 6px;
    margin-bottom: 14px;
    margin-top: 22px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    position: relative;
    overflow: hidden;
}
.sec-header::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    border-radius: 99px;
    background: currentColor;
}

/* ── METRIC CARDS ───────────────────────────────────────── */
.metric-card {
    background: linear-gradient(145deg, rgba(255,255,255,.05) 0%, rgba(255,255,255,.02) 100%);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 16px;
    padding: 20px 14px;
    text-align: center;
    transition: all .25s ease;
    box-shadow: 0 4px 20px rgba(0,0,0,.3);
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,.15), transparent);
}
.metric-card:hover {
    border-color: rgba(0,212,255,.3);
    transform: translateY(-3px);
    box-shadow: 0 8px 28px rgba(0,0,0,.4), 0 0 16px rgba(0,212,255,.1);
}
.metric-val {
    font-size: 1.9rem;
    font-weight: 900;
    line-height: 1;
    font-family: 'Space Mono', monospace;
}
.metric-label {
    font-size: 0.68rem;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: .14em;
    margin-top: 6px;
    font-family: 'Space Mono', monospace;
}

/* ── CHAT MESSAGES ──────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: linear-gradient(135deg,rgba(0,212,255,.03),rgba(124,58,237,.02)) !important;
    border: 1px solid rgba(0,212,255,.09) !important;
    border-radius: 16px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 2px 16px rgba(0,0,0,.25) !important;
    transition: border-color .2s !important;
}
[data-testid="stChatMessage"]:hover {
    border-color: rgba(0,212,255,.2) !important;
}

/* ── INPUTS ─────────────────────────────────────────────── */
.stTextInput input,
.stTextArea textarea,
.stSelectbox > div > div,
.stNumberInput input {
    background: rgba(13,17,23,.9) !important;
    border: 1px solid rgba(255,255,255,.08) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.84rem !important;
    transition: all .2s ease !important;
}
.stTextInput input:focus,
.stTextArea textarea:focus,
.stNumberInput input:focus {
    border-color: #00d4ff !important;
    box-shadow: 0 0 0 3px rgba(0,212,255,.1), 0 0 20px rgba(0,212,255,.08) !important;
    background: rgba(0,212,255,.03) !important;
}

/* ── BUTTONS ─────────────────────────────────────────────── */
.stButton button {
    background: rgba(255,255,255,.04) !important;
    border: 1px solid rgba(255,255,255,.1) !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    transition: all .2s ease !important;
    letter-spacing: .04em !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton button:hover {
    background: rgba(0,212,255,.09) !important;
    border-color: rgba(0,212,255,.35) !important;
    color: #00d4ff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(0,212,255,.12) !important;
}
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #00bcd4 0%, #7c3aed 100%) !important;
    color: #fff !important;
    border: none !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 20px rgba(0,188,212,.3), 0 0 40px rgba(124,58,237,.15) !important;
    letter-spacing: .06em !important;
}
.stButton button[kind="primary"]:hover {
    background: linear-gradient(135deg, #00d4ff 0%, #9333ea 100%) !important;
    box-shadow: 0 6px 28px rgba(0,212,255,.4), 0 0 50px rgba(147,51,234,.2) !important;
    transform: translateY(-2px) !important;
    color: #fff !important;
}

/* ── DOWNLOAD BUTTON ────────────────────────────────────── */
.stDownloadButton button {
    background: rgba(16,185,129,.08) !important;
    border: 1px solid rgba(16,185,129,.25) !important;
    color: #34d399 !important;
    border-radius: 10px !important;
}
.stDownloadButton button:hover {
    background: rgba(16,185,129,.16) !important;
    border-color: rgba(52,211,153,.5) !important;
    box-shadow: 0 0 20px rgba(16,185,129,.2) !important;
    color: #6ee7b7 !important;
    transform: translateY(-1px) !important;
}

/* ── EXPANDERS ──────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,.03) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,.07) !important;
    transition: all .2s ease !important;
}
.streamlit-expanderHeader:hover {
    border-color: rgba(0,212,255,.25) !important;
    background: rgba(0,212,255,.04) !important;
    box-shadow: 0 0 16px rgba(0,212,255,.06) !important;
}
.streamlit-expanderContent {
    border: 1px solid rgba(255,255,255,.05) !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    background: rgba(255,255,255,.01) !important;
}

/* ── TABS ───────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,.02) !important;
    border-radius: 14px !important;
    padding: 5px !important;
    gap: 3px !important;
    border: 1px solid rgba(255,255,255,.06) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-radius: 10px !important;
    color: #475569 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.74rem !important;
    padding: 7px 18px !important;
    transition: all .2s ease !important;
    font-weight: 700 !important;
    letter-spacing: .04em !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(255,255,255,.05) !important;
    color: #94a3b8 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,212,255,.2), rgba(124,58,237,.2)) !important;
    color: #00d4ff !important;
    box-shadow: 0 0 14px rgba(0,212,255,.18), inset 0 1px 0 rgba(255,255,255,.08) !important;
    text-shadow: 0 0 12px rgba(0,212,255,.5) !important;
}

/* ── CODE BLOCKS ────────────────────────────────────────── */
.stCode, pre {
    background: #040608 !important;
    border: 1px solid rgba(0,212,255,.12) !important;
    border-radius: 12px !important;
    font-family: 'Space Mono', monospace !important;
    box-shadow: inset 0 2px 10px rgba(0,0,0,.4) !important;
}
code {
    font-family: 'Space Mono', monospace !important;
    color: #67e8f9 !important;
    background: rgba(0,212,255,.07) !important;
    padding: 2px 7px !important;
    border-radius: 5px !important;
    border: 1px solid rgba(0,212,255,.12) !important;
}

/* ── DIVIDER ────────────────────────────────────────────── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,.2), rgba(124,58,237,.2), transparent) !important;
    margin: 16px 0 !important;
}

/* ── ALERTS ─────────────────────────────────────────────── */
[data-testid="stSuccess"] {
    background: rgba(16,185,129,.07) !important;
    border: 1px solid rgba(52,211,153,.25) !important;
    border-left: 3px solid #10b981 !important;
    border-radius: 10px !important;
    color: #6ee7b7 !important;
}
[data-testid="stWarning"] {
    background: rgba(245,158,11,.07) !important;
    border: 1px solid rgba(245,158,11,.25) !important;
    border-left: 3px solid #f59e0b !important;
    border-radius: 10px !important;
}
[data-testid="stError"] {
    background: rgba(239,68,68,.07) !important;
    border: 1px solid rgba(239,68,68,.25) !important;
    border-left: 3px solid #ef4444 !important;
    border-radius: 10px !important;
}
[data-testid="stInfo"] {
    background: rgba(0,212,255,.06) !important;
    border: 1px solid rgba(0,212,255,.2) !important;
    border-left: 3px solid #00d4ff !important;
    border-radius: 10px !important;
    color: #7dd3fc !important;
}

/* ── SPINNER ─────────────────────────────────────────────── */
[data-testid="stSpinner"] > div {
    border-color: rgba(0,212,255,.15) !important;
    border-top-color: #00d4ff !important;
}

/* ── BADGES ─────────────────────────────────────────────── */
.badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 12px; border-radius: 99px;
    font-size: .68rem; font-weight: 700;
    letter-spacing: .08em; text-transform: uppercase;
    font-family: 'Space Mono', monospace;
}
.badge-cyan   { background:rgba(0,212,255,.1);   color:#00d4ff;  border:1px solid rgba(0,212,255,.25);  box-shadow:0 0 10px rgba(0,212,255,.15); }
.badge-purple { background:rgba(124,58,237,.1);  color:#a78bfa;  border:1px solid rgba(124,58,237,.25); box-shadow:0 0 10px rgba(124,58,237,.15); }
.badge-green  { background:rgba(16,185,129,.1);  color:#34d399;  border:1px solid rgba(16,185,129,.25); box-shadow:0 0 10px rgba(16,185,129,.15); }
.badge-amber  { background:rgba(245,158,11,.1);  color:#fbbf24;  border:1px solid rgba(245,158,11,.25); box-shadow:0 0 10px rgba(245,158,11,.15); }
.badge-red    { background:rgba(239,68,68,.1);   color:#f87171;  border:1px solid rgba(239,68,68,.25);  box-shadow:0 0 10px rgba(239,68,68,.15); }

/* ── NAV ACTIVE ─────────────────────────────────────────── */
.nav-active button {
    background: linear-gradient(135deg, rgba(0,188,212,.25), rgba(124,58,237,.25)) !important;
    border: 1px solid rgba(0,212,255,.4) !important;
    color: #00d4ff !important;
    font-weight: 700 !important;
    box-shadow: 0 0 16px rgba(0,212,255,.2), inset 0 0 12px rgba(0,212,255,.05) !important;
    text-shadow: 0 0 10px rgba(0,212,255,.4) !important;
}

/* ── AI RESPONSE CARD ───────────────────────────────────── */
.ai-response-card {
    background: linear-gradient(135deg, rgba(0,212,255,.04), rgba(124,58,237,.03));
    border: 1px solid rgba(0,212,255,.15);
    border-left: 3px solid #00d4ff;
    border-radius: 0 14px 14px 0;
    padding: 18px 22px;
    margin: 12px 0;
    box-shadow: 0 4px 24px rgba(0,0,0,.3), 0 0 20px rgba(0,212,255,.05);
    animation: fadeIn .3s ease both;
}

/* ── TERMINAL OUTPUT ────────────────────────────────────── */
.terminal-output {
    background: linear-gradient(135deg, #020305, #040608);
    border: 1px solid rgba(0,212,255,.15);
    border-radius: 14px;
    padding: 18px 22px;
    font-family: 'Space Mono', monospace;
    font-size: .82rem;
    color: #67e8f9;
    box-shadow: inset 0 2px 12px rgba(0,0,0,.6), 0 0 20px rgba(0,212,255,.05);
    white-space: pre-wrap;
    line-height: 1.8;
    position: relative;
    overflow: hidden;
}
.terminal-output::before {
    content: '● ● ●';
    position: absolute;
    top: 10px; left: 16px;
    font-size: .6rem;
    color: #1e3040;
    letter-spacing: 6px;
}
.terminal-output-inner { margin-top: 16px; }
.t-prompt { color: #10b981; font-weight: 700; }
.t-error  { color: #ef4444; font-weight: 700; }
.t-line   { color: #67e8f9; }

/* ── STAT ROW (chat page) ───────────────────────────────── */
.stat-row {
    display: flex; gap: 12px; flex-wrap: wrap;
    margin: 16px 0;
}
.stat-chip {
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 99px;
    padding: 5px 14px;
    font-family: 'Space Mono', monospace;
    font-size: .7rem;
    color: #64748b;
    display: flex; align-items: center; gap: 6px;
    transition: all .2s;
}
.stat-chip:hover { border-color: rgba(0,212,255,.3); color: #00d4ff; }
.stat-chip b { color: #94a3b8; }

/* ── FEATURE GRID ───────────────────────────────────────── */
.feat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin: 16px 0;
}
.feat-card {
    background: rgba(255,255,255,.03);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 12px;
    padding: 14px 16px;
    transition: all .2s ease;
    cursor: default;
}
.feat-card:hover {
    border-color: rgba(0,212,255,.25);
    background: rgba(0,212,255,.04);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,.3);
}
.feat-icon { font-size: 1.4rem; margin-bottom: 6px; }
.feat-name { font-size: .78rem; font-weight: 700; color: #cbd5e1; }
.feat-desc { font-size: .68rem; color: #475569; margin-top: 3px; font-family: 'Space Mono', monospace; }

/* ── PRO CODE BLOCK (VS Code style) ─────────────────────────────── */
.code-block-wrapper {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(0,212,255,.18);
    box-shadow: 0 0 0 1px rgba(0,0,0,.5), 0 8px 32px rgba(0,0,0,.55), 0 0 40px rgba(0,212,255,.06);
    margin: 14px 0;
    font-family: 'Space Mono', monospace;
    animation: fadeUp .3s ease both;
}
.code-block-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 9px 16px;
    background: linear-gradient(90deg, #0d1117, #111827);
    border-bottom: 1px solid rgba(0,212,255,.1);
}
.code-block-dots { display: flex; gap: 6px; align-items: center; }
.code-block-dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
.dot-red    { background: #ef4444; box-shadow: 0 0 6px rgba(239,68,68,.5); }
.dot-yellow { background: #f59e0b; box-shadow: 0 0 6px rgba(245,158,11,.5); }
.dot-green  { background: #10b981; box-shadow: 0 0 6px rgba(16,185,129,.5); }
.code-block-lang {
    font-size: .68rem; font-weight: 700; letter-spacing: .14em;
    text-transform: uppercase; color: #00d4ff;
    background: rgba(0,212,255,.1); border: 1px solid rgba(0,212,255,.2);
    border-radius: 99px; padding: 2px 12px;
    font-family: 'Space Mono', monospace;
    text-shadow: 0 0 10px rgba(0,212,255,.5);
}
.code-block-copy {
    font-size: .66rem; color: #334155;
    background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);
    border-radius: 6px; padding: 3px 10px; cursor: pointer;
    font-family: 'Space Mono', monospace; letter-spacing: .06em; transition: all .2s;
}
.code-block-copy:hover { background: rgba(0,212,255,.1); border-color: rgba(0,212,255,.3); color: #00d4ff; }
.code-block-body { display: flex; background: #020407; overflow-x: auto; padding: 14px 0; }
.code-line-numbers {
    padding: 0 14px 0 16px; text-align: right; user-select: none;
    border-right: 1px solid rgba(0,212,255,.07); min-width: 44px;
    color: #1e3a4a; font-size: .8rem; line-height: 1.75; letter-spacing: .02em;
}
.code-line-numbers span { display: block; }
.code-content {
    padding: 0 20px; font-size: .82rem; line-height: 1.75;
    flex: 1; overflow-x: auto; white-space: pre; color: #a5f3fc; letter-spacing: .01em;
}
.tok-kw      { color: #c084fc; font-weight: 700; }
.tok-builtin { color: #67e8f9; }
.tok-string  { color: #86efac; }
.tok-comment { color: #334155; font-style: italic; }
.tok-number  { color: #fbbf24; }
.tok-func    { color: #60a5fa; }
.tok-bool    { color: #f472b6; font-weight: 700; }
.tok-self    { color: #f97316; }
.tok-deco    { color: #a78bfa; }

/* ── CAPTION ────────────────────────────────────────────── */
.stCaption { color: #334155 !important; font-family: 'Space Mono', monospace !important; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def sec_header(title: str, icon: str = "◆", color: str = "#00d4ff") -> None:
    st.markdown(
        f'<div class="sec-header" style="color:{color};border:1px solid {color}33;'
        f'background:linear-gradient(90deg,{color}18,{color}08);'
        f'box-shadow:0 0 14px {color}18;padding-left:16px;">{icon} {title}</div>',
        unsafe_allow_html=True,
    )


def metric_box(col, value, label: str, color: str = "#00d4ff") -> None:
    col.markdown(
        f'<div class="metric-card" style="border-color:{color}22;'
        f'background:linear-gradient(145deg,{color}09,{color}04);">'
        f'<div style="width:32px;height:3px;background:linear-gradient(90deg,{color},{color}55);'
        f'border-radius:99px;margin:0 auto 10px;"></div>'
        f'<div class="metric-val" style="color:{color};text-shadow:0 0 20px {color}66;">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )




import html as _html_mod
import re as _re_mod

def _highlight_python(code: str) -> str:
    """
    Accurate Python syntax highlighter using the stdlib `tokenize` module.
    Returns HTML string. Each token is classified exactly once — no regex
    re-processing of already-emitted span tags.
    """
    import tokenize as _tok
    import io as _io
    import html as _h

    TOKEN_CLASS = {
        _tok.COMMENT:  "tok-comment",
        _tok.STRING:   "tok-string",
        _tok.NUMBER:   "tok-number",
    }

    KEYWORDS = frozenset({
        "False","None","True","and","as","assert","async","await","break",
        "class","continue","def","del","elif","else","except","finally",
        "for","from","global","if","import","in","is","lambda","nonlocal",
        "not","or","pass","raise","return","try","while","with","yield",
    })

    BUILTINS = frozenset({
        "abs","all","any","bool","bytes","callable","chr","dict","dir",
        "divmod","enumerate","eval","exec","filter","float","format",
        "frozenset","getattr","globals","hasattr","hash","help","hex",
        "id","input","int","isinstance","issubclass","iter","len","list",
        "locals","map","max","memoryview","min","next","object","oct",
        "open","ord","pow","print","property","range","repr","reversed",
        "round","set","setattr","slice","sorted","staticmethod","str",
        "sum","super","tuple","type","vars","zip",
    })

    # ── tokenize ──────────────────────────────────────────────────────────────
    try:
        tokens = list(_tok.generate_tokens(_io.StringIO(code).readline))
    except _tok.TokenError:
        return _h.escape(code)          # fallback: plain escaped text

    lines   = code.splitlines(keepends=True)
    result  = []
    row, col = 1, 0                      # cursor position in source

    def advance_to(trow, tcol):
        """Emit raw (escaped) source text between cursor and token start."""
        nonlocal row, col
        while row < trow or (row == trow and col < tcol):
            if row > len(lines):
                break
            line = lines[row - 1]
            if row < trow:
                result.append(_h.escape(line[col:]))
                row += 1
                col  = 0
            else:
                result.append(_h.escape(line[col:tcol]))
                col  = tcol

    for tok_type, tok_str, tok_start, tok_end, _ in tokens:
        if tok_type in (_tok.ENCODING, _tok.ENDMARKER, _tok.NEWLINE,
                        _tok.NL, _tok.INDENT, _tok.DEDENT):
            continue
        if tok_type == _tok.ERRORTOKEN:
            advance_to(*tok_start)
            result.append(_h.escape(tok_str))
            row, col = tok_end
            continue

        advance_to(*tok_start)

        escaped = _h.escape(tok_str)

        if tok_type in TOKEN_CLASS:
            cls = TOKEN_CLASS[tok_type]
            result.append(f'<span class="{cls}">{escaped}</span>')
        elif tok_type == _tok.NAME:
            if tok_str in KEYWORDS:
                result.append(f'<span class="tok-kw">{escaped}</span>')
            elif tok_str in BUILTINS:
                result.append(f'<span class="tok-builtin">{escaped}</span>')
            elif tok_str == "self":
                result.append(f'<span class="tok-self">{escaped}</span>')
            else:
                result.append(escaped)
        elif tok_type == _tok.OP:
            result.append(f'<span class="tok-op">{escaped}</span>')
        else:
            result.append(escaped)

        row, col = tok_end

    # emit anything after the last token
    for ln in lines[row - 1:]:
        result.append(_h.escape(ln[col:]))
        col = 0

    return "".join(result)



_LANG_LABELS = {
    "python":"Python","py":"Python","javascript":"JavaScript","js":"JavaScript",
    "typescript":"TypeScript","ts":"TypeScript","java":"Java","cpp":"C++",
    "go":"Go","rust":"Rust","rs":"Rust","csharp":"C#","cs":"C#","sql":"SQL",
    "bash":"Bash","sh":"Bash","json":"JSON","html":"HTML","css":"CSS",
    "docker":"Dockerfile","diff":"Diff","text":"Output","jsx":"React","tsx":"React",
}


def render_code(code: str, language: str = "python", show_lines: bool = True) -> None:
    """
    Render a professional VS Code-style code block.
    Uses streamlit.components.v1.html() to bypass Streamlit's HTML sanitizer
    so inner <span> syntax-highlight tags are preserved.
    """
    import streamlit.components.v1 as _cv1

    if not isinstance(code, str):
        code = str(code)

    lang_key   = language.lower().strip()
    lang_label = _LANG_LABELS.get(lang_key, language.upper() or "Code")

    highlighted = _highlight_python(code) if lang_key in ("python", "py") else _html_mod.escape(code)
    lines       = highlighted.split("\n")
    n           = len(lines)

    # line numbers
    line_nums_html = "".join(f"<span>{i}</span>" for i in range(1, n + 1)) if show_lines else ""
    line_nums_col  = f'<div class="ln">{line_nums_html}</div>' if show_lines else ""

    # safe copy content
    safe_copy = code.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")

    body = "\n".join(lines)

    # Calculate iframe height: ~22px per line + 60px header + 30px padding
    height = max(120, n * 22 + 90)

    full_html = f"""<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    background: transparent;
    font-family: 'Space Mono', 'Fira Code', 'Consolas', monospace;
  }}
  .wrapper {{
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(0,212,255,.22);
    box-shadow: 0 0 0 1px rgba(0,0,0,.6),
                0 8px 32px rgba(0,0,0,.6),
                0 0 40px rgba(0,212,255,.07);
    background: #020407;
  }}
  /* ── Header ── */
  .header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 16px;
    background: linear-gradient(90deg, #0d1117 0%, #111827 100%);
    border-bottom: 1px solid rgba(0,212,255,.12);
  }}
  .dots {{ display:flex; gap:7px; align-items:center; }}
  .dot  {{ width:12px; height:12px; border-radius:50%; }}
  .d-r  {{ background:#ef4444; box-shadow:0 0 7px rgba(239,68,68,.6); }}
  .d-y  {{ background:#f59e0b; box-shadow:0 0 7px rgba(245,158,11,.6); }}
  .d-g  {{ background:#10b981; box-shadow:0 0 7px rgba(16,185,129,.6); }}
  .lang-badge {{
    font-size:.68rem; font-weight:700; letter-spacing:.15em;
    text-transform:uppercase; color:#00d4ff;
    background:rgba(0,212,255,.1); border:1px solid rgba(0,212,255,.25);
    border-radius:99px; padding:3px 14px;
    text-shadow:0 0 12px rgba(0,212,255,.6);
  }}
  .copy-btn {{
    font-size:.66rem; color:#475569;
    background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.1);
    border-radius:7px; padding:4px 12px; cursor:pointer;
    letter-spacing:.07em; transition:all .2s;
    font-family: 'Space Mono', monospace;
  }}
  .copy-btn:hover {{
    background:rgba(0,212,255,.12); border-color:rgba(0,212,255,.35); color:#00d4ff;
  }}
  /* ── Body ── */
  .body {{
    display: flex;
    overflow-x: auto;
    padding: 14px 0;
    background: #020407;
  }}
  .ln {{
    padding: 0 14px 0 16px;
    text-align: right;
    user-select: none;
    border-right: 1px solid rgba(0,212,255,.07);
    min-width: 48px;
    color: #1e3a4a;
    font-size: .8rem;
    line-height: 1.75;
    flex-shrink: 0;
  }}
  .ln span {{ display:block; }}
  .code {{
    padding: 0 20px;
    font-size: .82rem;
    line-height: 1.75;
    flex: 1;
    white-space: pre;
    color: #a5f3fc;
    letter-spacing: .01em;
  }}
  /* ── Syntax tokens ── */
  .tok-kw      {{ color:#c084fc; font-weight:700; }}
  .tok-builtin {{ color:#67e8f9; }}
  .tok-string  {{ color:#86efac; }}
  .tok-comment {{ color:#334155; font-style:italic; }}
  .tok-number  {{ color:#fbbf24; }}
  .tok-bool    {{ color:#f472b6; font-weight:700; }}
  .tok-self    {{ color:#f97316; }}
  .tok-deco    {{ color:#a78bfa; }}
</style>
</head>
<body>
<div class="wrapper">
  <div class="header">
    <div class="dots">
      <span class="dot d-r"></span>
      <span class="dot d-y"></span>
      <span class="dot d-g"></span>
    </div>
    <span class="lang-badge">{lang_label}</span>
    <button class="copy-btn" id="copybtn">&#10232; Copy</button>
  </div>
  <div class="body">
    {line_nums_col}
    <div class="code">{body}</div>
  </div>
</div>
<script>
  document.getElementById('copybtn').addEventListener('click', function() {{
    var code = `{safe_copy}`;
    navigator.clipboard.writeText(code).then(function() {{
      document.getElementById('copybtn').textContent = '✓ Copied!';
      setTimeout(function() {{
        document.getElementById('copybtn').textContent = '⎘ Copy';
      }}, 1500);
    }});
  }});
</script>
</body>
</html>"""

    _cv1.html(full_html, height=height, scrolling=False)




def render_llm_response(text: str) -> None:
    """
    Smart renderer for LLM responses.
    Splits on fenced code blocks (```lang ... ```) and renders:
      - prose   → st.markdown()
      - code    → render_code()  (VS Code-style iframe block)
    """
    if not text:
        return

    # Pattern: optional lang tag, then code content
    FENCE = re.compile(r'```(\w*)\n?(.*?)```', re.DOTALL)
    pos   = 0

    for m in FENCE.finditer(text):
        # Render prose before this block
        prose = text[pos:m.start()].strip()
        if prose:
            st.markdown(prose)

        lang = m.group(1).strip().lower() or "python"
        code = m.group(2)
        # Remove a trailing newline that most LLMs add before closing ```
        if code.endswith('\n'):
            code = code[:-1]

        render_code(code, language=lang)
        pos = m.end()

    # Render any trailing prose after the last code block
    tail = text[pos:].strip()
    if tail:
        st.markdown(tail)


@st.cache_resource
def _get_db():
    return utils.init_db()


conn = _get_db()


# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ═════════════════════════════════════════════════════════════════════════════
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "history" not in st.session_state:
    st.session_state.history = utils.load_chat_history(conn, st.session_state.session_id)
if "selected_model" not in st.session_state:
    st.session_state.selected_model = list(utils.GROQ_MODELS.keys())[1]
if "page" not in st.session_state:
    st.session_state.page = "💬 AI Chat"
if "api_history" not in st.session_state:
    st.session_state.api_history = []
if "api_profiles" not in st.session_state:
    st.session_state.api_profiles = utils.load_api_profiles(conn)
if "monitor_history" not in st.session_state:
    st.session_state.monitor_history = []


def llm():
    """Always returns a fresh LLM matching the currently selected model."""
    return utils.get_llm(st.session_state.selected_model)


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
PAGES = [
    "💬 AI Chat",
    "🔬 Code Tools",
    "🖥️ Dev Sandbox",
    "🌐 API Suite",
    "⚙️ Generators",
    "📚 Snippets",
]

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 10px;">
        <div style="
            font-family:'Syne',sans-serif;font-weight:900;font-size:1.7rem;
            background:linear-gradient(135deg,#00d4ff,#7c3aed,#10b981);
            background-size:200% 200%;
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            background-clip:text;letter-spacing:-1px;
            filter:drop-shadow(0 0 16px rgba(0,212,255,.35));
        ">⚡ CodeNova</div>
        <div style="
            font-family:'Space Mono',monospace;font-size:.6rem;
            color:#1e3a4a;letter-spacing:.3em;text-transform:uppercase;
            margin-top:4px;
        ">AI DEV SUITE v2</div>
        <div style="
            width:60%;height:1px;
            background:linear-gradient(90deg,transparent,rgba(0,212,255,.4),transparent);
            margin:12px auto 0;
        "></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Navigation ──
    st.markdown('<div style="font-family:Space Mono,monospace;font-size:.6rem;color:#1e3a4a;letter-spacing:.22em;text-transform:uppercase;margin:4px 0 8px;padding-left:4px;">Navigation</div>', unsafe_allow_html=True)
    for page in PAGES:
        is_active = st.session_state.page == page
        if is_active:
            st.markdown('<div class="nav-active">', unsafe_allow_html=True)
        if st.button(page, key=f"nav_{page}", use_container_width=True):
            st.session_state.page = page
            st.rerun()
        if is_active:
            st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ── Model selector ──
    st.markdown('<div style="font-family:Space Mono,monospace;font-size:.6rem;color:#1e3a4a;letter-spacing:.22em;text-transform:uppercase;margin:4px 0 8px;padding-left:4px;">LLM Model</div>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Model", list(utils.GROQ_MODELS.keys()),
        index=list(utils.GROQ_MODELS.keys()).index(st.session_state.selected_model),
        label_visibility="collapsed",
    )
    if model_choice != st.session_state.selected_model:
        st.session_state.selected_model = model_choice
        st.rerun()
    st.caption(f"ID: `{utils.GROQ_MODELS[st.session_state.selected_model]}`")

    st.divider()

    # ── Chat utilities ──
    if st.session_state.page == "💬 AI Chat":
        st.markdown('<div style="font-family:Space Mono,monospace;font-size:.6rem;color:#1e3a4a;letter-spacing:.22em;text-transform:uppercase;margin:4px 0 8px;padding-left:4px;">Chat Utilities</div>', unsafe_allow_html=True)

        new_sess = st.button("🆕 New Session", use_container_width=True)
        if new_sess:
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.session_state.history = []
            st.rerun()

        if st.button("🧼 Clear History", use_container_width=True):
            utils.clear_chat_history(conn, st.session_state.session_id)
            st.session_state.history = []
            st.rerun()

        if st.button("📄 Export PDF", use_container_width=True):
            if st.session_state.history:
                pdf_buf = utils.export_chat_pdf(
                    st.session_state.history, st.session_state.session_id
                )
                st.download_button(
                    "⬇️ Download PDF", pdf_buf,
                    file_name=f"codenova_{st.session_state.session_id}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                st.warning("No messages yet.")

        if st.button("💾 Export JSON", use_container_width=True):
            if st.session_state.history:
                st.download_button(
                    "⬇️ Download JSON",
                    json.dumps(st.session_state.history, indent=2),
                    file_name=f"chat_{st.session_state.session_id}.json",
                    mime="application/json",
                    use_container_width=True,
                )

    st.divider()
    st.markdown(
        f'''<div style="
            margin-top:8px;padding:12px;
            background:rgba(0,212,255,.04);
            border:1px solid rgba(0,212,255,.1);
            border-radius:10px;text-align:center;
        ">
            <div style="font-family:Space Mono,monospace;font-size:.6rem;color:#1e3a4a;letter-spacing:.1em;margin-bottom:6px;">SESSION INFO</div>
            <div style="font-family:Space Mono,monospace;font-size:.65rem;">
                <span style="color:#334155;">id:</span>
                <span style="color:#0e7490;">{st.session_state.session_id}</span>
            </div>
            <div style="font-family:Space Mono,monospace;font-size:.65rem;margin-top:3px;">
                <span style="color:#334155;">llm:</span>
                <span style="color:#7c3aed;">{st.session_state.selected_model.split()[1]}</span>
            </div>
        </div>''',
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGES
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────
# 1. AI CHAT
# ─────────────────────────────────────────────

def page_chat():
    st.markdown('<div class="nova-title">CodeNova</div>', unsafe_allow_html=True)
    st.markdown('<div class="nova-sub">Your AI-Powered Developer Suite</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="stat-row" style="justify-content:center;margin-top:14px;">
        <div class="stat-chip">🧠 <b>3</b> LLM Models</div>
        <div class="stat-chip">⚡ <b>15+</b> AI Tools</div>
        <div class="stat-chip">🗄️ <b>SQLite</b> Persistence</div>
        <div class="stat-chip">🎤 <b>Voice</b> Input</div>
        <div class="stat-chip">📤 <b>PDF/JSON</b> Export</div>
    </div>
    <div style="width:100%;height:1px;
        background:linear-gradient(90deg,transparent,rgba(0,212,255,.2),rgba(124,58,237,.2),transparent);
        margin:18px 0 20px;"></div>
    """, unsafe_allow_html=True)

    # voice input (optional)
    with st.expander("🎤 Voice Input (optional)"):
        audio_val = st.audio_input("Record your question")
        voice_text = None
        if audio_val:
            try:
                import speech_recognition as sr
                recognizer = sr.Recognizer()
                with sr.AudioFile(audio_val) as src:
                    audio = recognizer.record(src)
                voice_text = recognizer.recognize_google(audio)
                st.info(f"Heard: **{voice_text}**")
            except Exception as exc:
                st.warning(f"Voice recognition failed: {exc}")

    # chat history
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            render_llm_response(msg["content"])

    # input
    user_input = st.chat_input("Ask a coding question…")
    prompt_text = voice_text if (voice_text if "voice_text" in dir() else None) else user_input

    if prompt_text:
        st.session_state.history.append({"role": "user", "content": prompt_text})
        utils.save_chat_message(conn, st.session_state.session_id, "user", prompt_text, st.session_state.selected_model)
        with st.chat_message("user"):
            st.markdown(prompt_text)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                chain  = utils.build_chat_chain(llm())
                answer = chain.invoke({"question": prompt_text})
            render_llm_response(answer)
            # subtle label badge
            st.markdown(
                f'<span class="badge badge-cyan">'
                f'⚡ {st.session_state.selected_model.split()[1]} {st.session_state.selected_model.split()[0]}</span>',
                unsafe_allow_html=True,
            )

            # optional TTS
            tts_buf = utils.text_to_speech(answer)
            if tts_buf:
                st.audio(tts_buf, format="audio/mp3")

        st.session_state.history.append({"role": "assistant", "content": answer})
        utils.save_chat_message(conn, st.session_state.session_id, "assistant", answer, st.session_state.selected_model)


# ─────────────────────────────────────────────
# 2. CODE TOOLS
# ─────────────────────────────────────────────

def page_code_tools():
    st.markdown('''
    <div class="page-banner" style="--banner-a:#7c3aed;--banner-b:#a855f7;">
        <div class="page-banner-border"></div>
        <div class="page-banner-content">
            <span class="page-banner-icon">🔬</span>
            <div class="page-banner-title">Code Tools</div>
            <div class="page-banner-desc">Explain · Translate · Refactor · Secure · Score · Debug</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    sec_header("Code Tools", "🔬", "#7c3aed")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🧵 Explain", "🔁 Translate", "♻️ Refactor",
        "🛡️ Security", "⭐ Quality", "❗ Debug Error",
    ])

    # ── Explain ──────────────────────────────────────────
    with tab1:
        sec_header("Code Explainer", "🧵", "#7c3aed")
        code_in = st.text_area("Paste code:", height=200, key="explain_in")
        uploaded = st.file_uploader("Or upload a file", type=["py","js","ts","java","cpp","go","rs","cs","rb"], key="explain_file")
        if uploaded:
            extra = uploaded.getvalue().decode("utf-8", errors="replace")
            render_code(extra[:3000], language="python")
            code_in = code_in + "\n" + extra
        if st.button("🧠 Explain Code", key="btn_explain", type="primary"):
            if code_in.strip():
                with st.spinner("Analyzing…"):
                    out = utils.build_explain_chain(llm()).invoke({"code": code_in})
                render_llm_response(out)
            else:
                st.warning("Paste some code first.")

    # ── Translate ─────────────────────────────────────────
    with tab2:
        sec_header("Code Translator", "🔁", "#00d4ff")
        c1, c2 = st.columns(2)
        src_lang = c1.selectbox("Source language", utils.LANGUAGES, key="trans_src")
        tgt_lang = c2.selectbox("Target language", [l for l in utils.LANGUAGES if l != src_lang], key="trans_tgt")
        code_in_t = st.text_area("Source code:", height=200, key="translate_in")
        if st.button("🔁 Translate", key="btn_translate", type="primary"):
            if code_in_t.strip():
                with st.spinner(f"Translating {src_lang} → {tgt_lang}…"):
                    result = utils.build_translate_chain(llm()).invoke({
                        "source_lang": src_lang, "target_lang": tgt_lang, "code": code_in_t
                    })
                render_code(result, language=utils.LANG_HIGHLIGHT.get(tgt_lang, "python"))
                st.download_button(
                    "⬇️ Download translated file",
                    result,
                    file_name=f"translated.{utils.LANG_EXT.get(tgt_lang,'txt')}",
                    mime="text/plain",
                )
            else:
                st.warning("Paste source code first.")

    # ── Refactor ──────────────────────────────────────────
    with tab3:
        sec_header("AI Refactor + Diff View", "♻️", "#10b981")
        code_in_r = st.text_area("Code to refactor:", height=220, key="refactor_in")
        if st.button("♻️ Refactor Code", key="btn_refactor", type="primary"):
            if code_in_r.strip():
                with st.spinner("Refactoring…"):
                    refactored = utils.build_refactor_chain(llm()).invoke({"code": code_in_r})
                rc1, rc2 = st.columns(2)
                rc1.markdown("**Original**")
                rc1.code(code_in_r, language="python")
                rc2.markdown("**Refactored**")
                rc2.code(refactored, language="python")
                with st.expander("📑 Show unified diff"):
                    diff_lines = list(__import__("difflib").unified_diff(
                        code_in_r.splitlines(), refactored.splitlines(),
                        fromfile="original", tofile="refactored", lineterm=""
                    ))
                    if diff_lines:
                        render_code("\n".join(diff_lines), language="diff")
                    else:
                        st.info("No differences found.")
                st.download_button("⬇️ Download refactored", refactored, file_name="refactored.py", mime="text/plain")

    # ── Security ──────────────────────────────────────────
    with tab4:
        sec_header("Security Scanner", "🛡️", "#ef4444")
        st.caption("Scans for common vulnerabilities: injection, hardcoded secrets, insecure calls, and more.")
        code_in_s = st.text_area("Paste code to scan:", height=220, key="sec_in")
        if st.button("🔍 Scan for Vulnerabilities", key="btn_sec", type="primary"):
            if code_in_s.strip():
                with st.spinner("Scanning…"):
                    findings = utils.build_security_chain(llm()).invoke({"code": code_in_s})

                # parse severity badges
                lines = findings.strip().split("\n")
                for line in lines:
                    if line.strip():
                        color = "#ef4444" if "CRITICAL" in line.upper() else \
                                "#f97316" if "HIGH"     in line.upper() else \
                                "#facc15" if "MEDIUM"   in line.upper() else \
                                "#94a3b8"
                        st.markdown(
                            f'<div style="border-left:3px solid {color};padding:8px 12px;'
                            f'margin:6px 0;background:{color}11;border-radius:0 8px 8px 0;">{line}</div>',
                            unsafe_allow_html=True,
                        )
            else:
                st.warning("Paste code to scan.")

    # ── Quality ───────────────────────────────────────────
    with tab5:
        sec_header("Code Quality Score", "⭐", "#f59e0b")
        code_in_q = st.text_area("Paste code to score:", height=220, key="qual_in")
        if st.button("📊 Score My Code", key="btn_qual", type="primary"):
            if code_in_q.strip():
                with st.spinner("Evaluating…"):
                    raw = utils.build_quality_chain(llm()).invoke({"code": code_in_q})
                try:
                    clean = re.sub(r"```[a-z]*", "", raw).strip().strip("`").strip()
                    scores = json.loads(clean)
                    overall = scores.get("overall", 5)
                    ov_color = "#10b981" if overall >= 7 else "#f59e0b" if overall >= 4 else "#ef4444"

                    m1, m2, m3, m4, m5, m6 = st.columns(6)
                    metric_box(m1, f"{scores.get('readability',0)}/10",  "Readability",    "#00d4ff")
                    metric_box(m2, f"{scores.get('efficiency',0)}/10",   "Efficiency",     "#7c3aed")
                    metric_box(m3, f"{scores.get('error_handling',0)}/10","Error Handling", "#ef4444")
                    metric_box(m4, f"{scores.get('best_practices',0)}/10","Best Practices", "#10b981")
                    metric_box(m5, f"{scores.get('documentation',0)}/10","Docs",            "#f59e0b")
                    metric_box(m6, f"{overall}/10", "Overall", ov_color)

                    st.markdown(f"**Summary:** {scores.get('summary','')}")
                    fig = utils.plot_quality_radar(scores)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    render_llm_response(raw)
            else:
                st.warning("Paste code to score.")

    # ── Debug Error ───────────────────────────────────────
    with tab6:
        sec_header("Error Debugger", "❗", "#f97316")
        tb_in = st.text_area("Paste error traceback:", height=200, key="err_in")
        if st.button("🔍 Diagnose Error", key="btn_err", type="primary"):
            if tb_in.strip():
                with st.spinner("Diagnosing…"):
                    diag = utils.build_error_chain(llm()).invoke({"traceback": tb_in})
                render_llm_response(diag)
            else:
                st.warning("Paste an error traceback first.")


# ─────────────────────────────────────────────
# 3. DEV SANDBOX
# ─────────────────────────────────────────────

def page_dev_sandbox():
    st.markdown('''
    <div class="page-banner" style="--banner-a:#10b981;--banner-b:#059669;">
        <div class="page-banner-border"></div>
        <div class="page-banner-content">
            <span class="page-banner-icon">🖥️</span>
            <div class="page-banner-title">Dev Sandbox</div>
            <div class="page-banner-desc">Live Python runner · AI Fixer · Complexity Analyzer</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    sec_header("Dev Sandbox", "🖥️", "#10b981")
    tab1, tab2 = st.tabs(["🐍 Python Runner", "📏 Complexity Analyzer"])

    # ── Python Runner ─────────────────────────────────────
    with tab1:
        sec_header("Live Python Sandbox", "🐍", "#10b981")
        st.caption("Execute Python snippets safely in-app.")

        default_code = "# CodeNova Sandbox\nimport math\nfor i in range(5):\n    print(f'√{i} = {math.sqrt(i):.4f}')"
        sandbox_code = st.text_area("Python code:", value=default_code, height=200, key="sandbox_code")

        col_run, col_clear = st.columns([1, 4])
        run_clicked = col_run.button("▶ Run", type="primary", key="run_sandbox")
        if col_clear.button("✖ Clear Output", key="clr_sandbox"):
            if "sandbox_output" in st.session_state:
                del st.session_state["sandbox_output"]

        if run_clicked:
            result = utils.run_python_sandbox(sandbox_code)
            st.session_state["sandbox_output"] = result

        if "sandbox_output" in st.session_state:
            res = st.session_state["sandbox_output"]
            st.markdown("<br>", unsafe_allow_html=True)
            t1, t2, t3 = st.columns(3)
            metric_box(t1, "✅ OK" if res["success"] else "❌ ERR", "Status",
                       "#10b981" if res["success"] else "#ef4444")
            metric_box(t2, f"{res['exec_time_ms']} ms", "Exec Time", "#00d4ff")
            metric_box(t3, f"{len(res['output'].splitlines())} lines", "Output Lines", "#7c3aed")
            st.markdown("<br>", unsafe_allow_html=True)
            if res["success"]:
                output_text = res["output"] or "(no output)"
                # Render as styled terminal output
                import html as _html
                escaped = _html.escape(output_text)
                st.markdown(
                    f'''<div class="terminal-output">'''
                    f'''<span class="t-prompt">▶ output</span>\n{escaped}'''
                    f'''</div>''',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'''<div class="terminal-output">'''
                    f'''<span class="t-error">✖ {_html.escape(res["error"])}</span>'''
                    f'''</div>''',
                    unsafe_allow_html=True,
                )

        st.divider()
        # AI Code Fixer
        sec_header("AI Code Fixer", "🤖", "#7c3aed")
        st.caption("Paste broken code — get a fix + explanation.")
        broken = st.text_area("Broken code:", height=150, key="fixer_in")
        if st.button("🔧 Fix It", key="btn_fix", type="primary"):
            if broken.strip():
                with st.spinner("Fixing…"):
                    fix_prompt = (
                        "You are an expert code debugger. Fix the broken code below. "
                        "Structure your response EXACTLY like this:\n"
                        "## What was wrong\n"
                        "- bullet 1\n- bullet 2\n\n"
                        "## Fixed Code\n"
                        "```python\n<fixed code here>\n```\n\n"
                        "## What changed\n"
                        "Brief explanation of each fix.\n\n"
                        f"Broken code:\n```\n{broken}\n```"
                    )
                    fixed = (llm()).invoke(fix_prompt).content
                # Render in a styled AI response card
                st.markdown(
                    '<div class="ai-response-card">',
                    unsafe_allow_html=True,
                )
                render_llm_response(fixed)
                st.markdown('</div>', unsafe_allow_html=True)

    # ── Complexity Analyzer ───────────────────────────────
    with tab2:
        sec_header("Time & Space Complexity Analyzer", "📏", "#00d4ff")
        st.caption("AI estimates Big-O + plots the growth curve against all complexities.")

        code_in = st.text_area("Paste your function:", height=200, key="complex_in")
        if st.button("🧠 Analyze Complexity", key="btn_complex", type="primary"):
            if code_in.strip():
                with st.spinner("Analyzing…"):
                    result = utils.build_complexity_chain(llm()).invoke({"code": code_in})

                render_llm_response(result)

                time_big_o = utils.parse_big_o(result.split("Time")[1] if "Time" in result else result)
                space_big_o = utils.parse_big_o(result.split("Space")[1] if "Space" in result else "")

                c1, c2 = st.columns(2)
                metric_box(c1, time_big_o or "?", "Time Complexity", "#00d4ff")
                metric_box(c2, space_big_o or "?", "Space Complexity", "#7c3aed")
                st.markdown("")

                if time_big_o:
                    fig = utils.plot_complexity_curve(time_big_o)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Paste a function first.")


# ─────────────────────────────────────────────
# 4. API SUITE
# ─────────────────────────────────────────────

def page_api_suite():
    st.markdown('''
    <div class="page-banner" style="--banner-a:#00d4ff;--banner-b:#0ea5e9;">
        <div class="page-banner-border"></div>
        <div class="page-banner-content">
            <span class="page-banner-icon">🌐</span>
            <div class="page-banner-title">API Suite</div>
            <div class="page-banner-desc">REST Tester · Profile Manager · Health Monitor</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    sec_header("API Suite", "🌐", "#00d4ff")
    tab1, tab2 = st.tabs(["🧪 API Tester", "📡 Health Monitor"])

    # ── API Tester ────────────────────────────────────────
    with tab1:
        sec_header("REST API Tester", "🧪", "#00d4ff")
        if "api_history" not in st.session_state:
            st.session_state.api_history = []

        # Profile load/save
        profiles = st.session_state.api_profiles
        with st.expander("📂 Saved Profiles", expanded=False):
            if profiles:
                sel_prof = st.selectbox("Load profile", ["— none —"] + list(profiles.keys()), key="load_prof")
                col_load, col_del = st.columns(2)
                if col_load.button("📥 Load", key="load_profile_btn") and sel_prof != "— none —":
                    p = profiles[sel_prof]
                    st.session_state["_api_url"]     = p.get("url", "")
                    st.session_state["_api_method"]  = p.get("method", "GET")
                    st.session_state["_api_token"]   = p.get("token", "")
                    st.session_state["_api_headers"] = json.dumps(p.get("headers", {}), indent=2)
                    st.session_state["_api_body"]    = json.dumps(p.get("body", {}), indent=2)
                    st.toast(f"Loaded '{sel_prof}'")
                if col_del.button("🗑️ Delete", key="del_profile_btn") and sel_prof != "— none —":
                    utils.delete_api_profile(conn, sel_prof)
                    del st.session_state.api_profiles[sel_prof]
                    st.rerun()
            else:
                st.caption("No saved profiles yet.")

        # Config form
        with st.expander("🛠 Request Configuration", expanded=True):
            api_url = st.text_input("URL", value=st.session_state.get("_api_url", ""), key="api_url_input")
            c1, c2 = st.columns(2)
            method  = c1.selectbox("Method", ["GET","POST","PUT","PATCH","DELETE"], key="api_method")
            auth_token = c2.text_input("Bearer Token", value=st.session_state.get("_api_token",""), type="password", key="api_token")
            headers_raw = st.text_area("Headers (JSON)", value=st.session_state.get("_api_headers", '{\n  "Content-Type": "application/json"\n}'), height=100, key="api_headers")
            body_raw = ""
            if method != "GET":
                body_raw = st.text_area("Body (JSON)", value=st.session_state.get("_api_body","{}"), height=120, key="api_body")
            c3, c4 = st.columns(2)
            exp_status = c3.text_input("Expected Status", placeholder="200", key="exp_status")
            timeout_s  = c4.number_input("Timeout (s)", value=10, min_value=1, max_value=60, key="api_timeout")

        # Save profile
        with st.expander("💾 Save as Profile"):
            pname = st.text_input("Profile name", key="new_profile_name")
            if st.button("💾 Save Profile") and pname:
                try:
                    utils.save_api_profile(conn, pname, {
                        "url": api_url, "method": method, "token": auth_token,
                        "headers": json.loads(headers_raw or "{}"),
                        "body": json.loads(body_raw or "{}") if method != "GET" else {},
                    })
                    st.session_state.api_profiles = utils.load_api_profiles(conn)
                    st.success(f"Saved '{pname}'")
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")

        # Send
        if st.button("🚀 Send Request", type="primary", key="send_api"):
            try:
                headers = json.loads(headers_raw or "{}")
            except json.JSONDecodeError as e:
                st.error(f"Bad headers JSON: {e}")
                return
            if auth_token:
                headers["Authorization"] = f"Bearer {auth_token}"
            data = None
            if method != "GET" and body_raw.strip():
                try:
                    data = json.loads(body_raw)
                except json.JSONDecodeError as e:
                    st.error(f"Bad body JSON: {e}")
                    return
            try:
                with st.spinner("Sending…"):
                    with httpx.Client(timeout=float(timeout_s)) as client:
                        resp = client.request(method=method, url=api_url, headers=headers, json=data)
                status_color = "#10b981" if resp.status_code < 400 else "#ef4444"
                st.markdown(
                    f'<div style="display:flex;gap:10px;align-items:center;margin:8px 0;">'
                    f'<span style="color:{status_color};font-weight:700;font-size:1.3rem;">{resp.status_code}</span>'
                    f'<span style="color:#64748b;">{resp.elapsed.total_seconds()*1000:.0f} ms</span>'
                    f'</div>', unsafe_allow_html=True
                )
                rtab1, rtab2, rtab3 = st.tabs(["📦 Response Body", "📋 Headers", "🔧 cURL"])
                with rtab1:
                    try:
                        st.json(resp.json())
                    except Exception:
                        render_code(resp.text, language="text")
                with rtab2:
                    st.json(dict(resp.headers))
                with rtab3:
                    curl_parts = [f'curl -X {method} "{api_url}"']
                    for k, v in headers.items():
                        curl_parts.append(f'  -H "{k}: {v}"')
                    if data:
                        curl_parts.append(f"  -d '{json.dumps(data)}'")
                    render_code(" \\\n".join(curl_parts), language="bash")

                if exp_status:
                    try:
                        if int(exp_status) == resp.status_code:
                            st.success("✅ Status matched expected.")
                        else:
                            st.error(f"❌ Expected {exp_status}, got {resp.status_code}")
                    except ValueError:
                        pass

                st.session_state.api_history.insert(0, {
                    "url": api_url, "method": method, "status": resp.status_code,
                    "time_ms": round(resp.elapsed.total_seconds() * 1000),
                    "response": resp.text[:4000],
                })

            except Exception as exc:
                st.error(f"Request failed: {exc}")

        # History
        if st.session_state.api_history:
            sec_header("Request History", "🗂️", "#4f46e5")
            if st.button("🧹 Clear history"):
                st.session_state.api_history = []
                st.rerun()
            for entry in st.session_state.api_history[:10]:
                color = "#10b981" if entry["status"] < 400 else "#ef4444"
                with st.expander(f'`{entry["method"]}` {entry["url"][:60]}  →  '
                                 f'<span style="color:{color};">{entry["status"]}</span>  '
                                 f'({entry["time_ms"]} ms)'):
                    try:
                        st.json(json.loads(entry["response"]))
                    except Exception:
                        render_code(entry["response"], language="text")
            st.download_button(
                "⬇️ Export history JSON",
                json.dumps(st.session_state.api_history, indent=2),
                file_name="api_history.json", mime="application/json",
            )

    # ── Health Monitor ────────────────────────────────────
    with tab2:
        sec_header("API Health Monitor", "📡", "#10b981")
        with st.expander("➕ Add endpoint to monitor", expanded=True):
            mc1, mc2, mc3 = st.columns([3, 1, 1])
            mon_name = mc1.text_input("Name", placeholder="My API", key="mon_name")
            mon_url  = mc1.text_input("URL",  placeholder="https://api.example.com/health", key="mon_url")
            mon_method = mc2.selectbox("Method", ["GET","POST"], key="mon_method")
            if mc3.button("🔁 Check Now", type="primary", key="btn_monitor"):
                if mon_name and mon_url:
                    with st.spinner("Pinging…"):
                        result = utils.monitor_api(mon_name, mon_url, mon_method)
                    st.session_state.monitor_history.insert(0, result)

        for res in st.session_state.monitor_history[:10]:
            color = "#10b981" if res["ok"] else "#ef4444"
            icon  = "✅" if res["ok"] else "❌"
            st.markdown(
                f'<div class="glass" style="border-left:3px solid {color};">'
                f'<b>{icon} {res["name"]}</b> &nbsp;|&nbsp; '
                f'<code>{res["method"]} {res["url"]}</code><br>'
                f'Status: <span style="color:{color};">{res["status_code"] or "ERR"}</span> &nbsp;|&nbsp; '
                f'Response time: <b>{res["response_time_ms"] or "—"} ms</b> &nbsp;|&nbsp; '
                f'<span style="color:#64748b;">{res["timestamp"]}</span>'
                + ('' if res['ok'] else f'<br><span style="color:#ef4444;">{res["error"]}</span>')
                + '</div>',
                unsafe_allow_html=True,
            )

        if not st.session_state.monitor_history:
            st.info("Add an endpoint and click **Check Now** to start monitoring.")


# ─────────────────────────────────────────────
# 5. GENERATORS
# ─────────────────────────────────────────────

def page_generators():
    st.markdown('''
    <div class="page-banner" style="--banner-a:#f59e0b;--banner-b:#ef4444;">
        <div class="page-banner-border"></div>
        <div class="page-banner-content">
            <span class="page-banner-icon">⚙️</span>
            <div class="page-banner-title">Generators</div>
            <div class="page-banner-desc">Unit Tests · Regex · SQL · Dockerfile · Git Commits · Changelog</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    sec_header("Generators", "⚙️", "#f59e0b")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🧪 Unit Tests", "🧩 Regex", "🗄️ SQL Builder",
        "🐳 Dockerfile", "📝 Git Commit", "📋 Changelog",
    ])

    # ── Unit Tests ────────────────────────────────────────
    with tab1:
        sec_header("Unit Test Generator", "🧪", "#f59e0b")
        lang = st.selectbox("Language / Framework", utils.LANGUAGES, key="ut_lang")
        func_code = st.text_area("Paste function / component:", height=200, key="ut_code")
        if st.button("🧪 Generate Tests", type="primary", key="btn_ut"):
            if func_code.strip():
                with st.spinner("Writing tests…"):
                    tests = utils.build_unit_test_chain(llm(), lang).invoke({"func": func_code})
                render_code(tests, language=utils.LANG_HIGHLIGHT.get(lang, "python"))
                st.download_button(
                    "⬇️ Download test file",
                    tests, file_name=f"test.{utils.LANG_EXT.get(lang,'txt')}",
                    mime="text/plain",
                )
            else:
                st.warning("Paste a function first.")

    # ── Regex ─────────────────────────────────────────────
    with tab2:
        sec_header("Regex Generator & Tester", "🧩", "#00d4ff")
        with st.popover("✨ Quick examples"):
            qc1, qc2, qc3 = st.columns(3)
            if qc1.button("Email"):         st.session_state["rx_desc"] = "match a standard email address"
            if qc2.button("Indian mobile"): st.session_state["rx_desc"] = "match Indian mobile numbers with optional +91"
            if qc3.button("URL"):           st.session_state["rx_desc"] = "match http or https URLs"

        rx_desc   = st.text_area("Describe the pattern:", value=st.session_state.get("rx_desc",""), key="rx_desc_area", height=80)
        rx_sample = st.text_area("Sample text to test (optional):", height=100, key="rx_sample")

        if st.button("🧠 Generate Regex", type="primary", key="btn_regex"):
            if rx_desc.strip():
                with st.spinner("Generating…"):
                    raw = llm().invoke(
                        "Return ONLY a Python regular-expression pattern, no description, no fences:\n" + rx_desc
                    ).content
                    pattern = utils.clean_regex_pattern(raw)
                if pattern:
                    st.session_state["rx_pattern"] = pattern
                    render_code(pattern, language="text")
                    with st.spinner("Explaining…"):
                        explain = utils.build_regex_explain_chain(llm()).invoke({"pattern": pattern})
                    st.markdown(f"**Explanation:** {explain}")

        if rx_sample.strip() and "rx_pattern" in st.session_state:
            try:
                pat = re.compile(st.session_state["rx_pattern"])
                matches = pat.findall(rx_sample)
                if matches:
                    st.success(f"✅ {len(matches)} match(es): `{matches}`")
                    highlighted = pat.sub(lambda m: f"**{m.group()}**", rx_sample)
                    st.markdown("**Highlighted:** " + highlighted)
                else:
                    st.warning("No matches in sample text.")
            except re.error as e:
                st.error(f"Regex error: {e}")

    # ── SQL Builder ───────────────────────────────────────
    with tab3:
        sec_header("SQL Query Builder", "🗄️", "#7c3aed")
        st.caption("Natural language → optimized SQL query")
        schema = st.text_area("Table schema (optional):", placeholder="users(id, name, email, created_at)\norders(id, user_id, amount, status)", height=100, key="sql_schema")
        nl_query = st.text_area("Your query in plain English:", placeholder="Find all users who placed more than 3 orders in the last 30 days", height=100, key="sql_query")
        if st.button("🗄️ Generate SQL", type="primary", key="btn_sql"):
            if nl_query.strip():
                with st.spinner("Generating SQL…"):
                    sql_out = utils.build_sql_chain(llm()).invoke({"schema": schema, "query": nl_query})
                render_code(sql_out, language="sql")
                st.download_button("⬇️ Download .sql", sql_out, file_name="query.sql", mime="text/plain")

    # ── Dockerfile ────────────────────────────────────────
    with tab4:
        sec_header("Dockerfile Generator", "🐳", "#0ea5e9")
        st.caption("Describe your project and get a production-ready Dockerfile.")
        docker_desc = st.text_area(
            "Project description:",
            placeholder="A Python 3.11 FastAPI REST API with PostgreSQL, Redis, and Celery workers. The app runs on port 8000.",
            height=140, key="docker_desc"
        )
        if st.button("🐳 Generate Dockerfile", type="primary", key="btn_docker"):
            if docker_desc.strip():
                with st.spinner("Generating Dockerfile…"):
                    dockerfile = utils.build_docker_chain(llm()).invoke({"description": docker_desc})
                render_code(dockerfile, language="docker")
                st.download_button("⬇️ Download Dockerfile", dockerfile, file_name="Dockerfile", mime="text/plain")

    # ── Git Commit ────────────────────────────────────────
    with tab5:
        sec_header("Git Commit Message Generator", "📝", "#10b981")
        st.caption("Paste a git diff or describe changes → get a conventional commit message.")
        diff_in = st.text_area("Git diff or change description:", height=200, key="commit_diff",
                               placeholder="feat: added user authentication\nChanged auth.py to use JWT tokens\nRemoved session-based auth")
        if st.button("📝 Generate Commit Message", type="primary", key="btn_commit"):
            if diff_in.strip():
                with st.spinner("Writing commit message…"):
                    commit_msg = utils.build_git_commit_chain(llm()).invoke({"diff": diff_in})
                render_code(commit_msg, language="text")
                st.download_button("⬇️ Copy as .txt", commit_msg, file_name="commit_msg.txt", mime="text/plain")

    # ── Changelog ─────────────────────────────────────────
    with tab6:
        sec_header("Changelog Generator", "📋", "#a855f7")
        st.caption("List your changes → get a formatted Keep-a-Changelog entry.")
        changes_in = st.text_area("List of changes:", height=160, key="changelog_in",
                                  placeholder="- Added OAuth2 login\n- Fixed null pointer in /api/users\n- Removed legacy XML parser\n- Updated dependencies")
        version = st.text_input("Version (optional):", placeholder="1.4.2", key="cl_version")
        if st.button("📋 Generate Changelog", type="primary", key="btn_cl"):
            if changes_in.strip():
                full_input = (f"Version: {version}\n\n" if version else "") + changes_in
                with st.spinner("Generating changelog…"):
                    cl_out = utils.build_changelog_chain(llm()).invoke({"changes": full_input})
                render_llm_response(cl_out)
                st.download_button("⬇️ Download CHANGELOG.md", cl_out, file_name="CHANGELOG.md", mime="text/plain")


# ─────────────────────────────────────────────
# 6. SNIPPETS
# ─────────────────────────────────────────────

def page_snippets():
    st.markdown('''
    <div class="page-banner" style="--banner-a:#a855f7;--banner-b:#7c3aed;">
        <div class="page-banner-border"></div>
        <div class="page-banner-content">
            <span class="page-banner-icon">📚</span>
            <div class="page-banner-title">Snippet Library</div>
            <div class="page-banner-desc">Save · Tag · Search · Reuse — backed by SQLite</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    sec_header("Snippet Library", "📚", "#a855f7")
    st.caption("Save, tag, and reuse code snippets across sessions. Stored persistently in SQLite.")

    # Save new snippet
    with st.expander("➕ Save New Snippet", expanded=False):
        sn_name = st.text_input("Name:", key="sn_name", placeholder="Binary search function")
        sn_lang = st.selectbox("Language:", utils.LANGUAGES, key="sn_lang")
        sn_tags = st.text_input("Tags (comma-separated):", key="sn_tags", placeholder="algorithms, search")
        sn_code = st.text_area("Code:", height=180, key="sn_code")
        if st.button("💾 Save Snippet", type="primary", key="btn_save_snip"):
            if sn_name.strip() and sn_code.strip():
                utils.save_snippet(conn, sn_name.strip(), sn_lang, sn_code.strip(), sn_tags.strip())
                st.success(f"Saved '{sn_name}'!")
                st.rerun()
            else:
                st.warning("Name and code are required.")

    # Search
    search = st.text_input("🔍 Search snippets:", key="snip_search", placeholder="Search by name or tag…")
    snippets = utils.load_snippets(conn)

    if search:
        q = search.lower()
        snippets = [s for s in snippets if q in s[1].lower() or q in (s[4] or "").lower()]

    if not snippets:
        st.info("No snippets yet. Save your first one above!")
        return

    st.markdown(f"**{len(snippets)} snippet(s)** found")

    for snip_id, name, lang, code, tags, created_at in snippets:
        with st.expander(f"📄 {name}  `{lang}`  — {created_at[:10]}"):
            if tags:
                tag_html = " ".join(
                    f'<span style="background:#1e2d4a;color:#94a3b8;border-radius:99px;'
                    f'padding:2px 10px;font-size:.75rem;margin-right:4px;">#{t.strip()}</span>'
                    for t in tags.split(",") if t.strip()
                )
                st.markdown(tag_html, unsafe_allow_html=True)
            render_code(code, language=utils.LANG_HIGHLIGHT.get(lang, "python"))

            col_dl, col_del, _ = st.columns([1, 1, 4])
            col_dl.download_button(
                "⬇️ Download",
                code,
                file_name=f"{name.replace(' ','_')}.{utils.LANG_EXT.get(lang,'txt')}",
                mime="text/plain",
                key=f"dl_{snip_id}",
            )
            if col_del.button("🗑️ Delete", key=f"del_{snip_id}"):
                utils.delete_snippet(conn, snip_id)
                st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# ROUTER
# ═════════════════════════════════════════════════════════════════════════════
_ROUTES = {
    "💬 AI Chat":    page_chat,
    "🔬 Code Tools": page_code_tools,
    "🖥️ Dev Sandbox": page_dev_sandbox,
    "🌐 API Suite":  page_api_suite,
    "⚙️ Generators": page_generators,
    "📚 Snippets":   page_snippets,
}

# Guard: check API key
if not utils.API_KEY:
    st.error(
        "⚠️ **GROQ_API_KEY not found.** Create a `.env` file with:\n"
        "```\nGROQ_API_KEY=your_key_here\n```\n"
        "Get a free key at https://console.groq.com"
    )
    st.stop()

# Render selected page
page_fn = _ROUTES.get(st.session_state.page, page_chat)
page_fn()
