import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from streamlit_lottie import st_lottie
import requests
from datetime import datetime
import speech_recognition as sr
from fpdf import FPDF
import json
import os
from dotenv import load_dotenv
load_dotenv()
from gtts import gTTS
import time
import zipfile
import io
from io import BytesIO
import httpx
import numpy as np
import math
import plotly.graph_objects as go
import re


API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="💬 CodeNova Your coding buddy", layout="centered")

# Theme
mode = st.sidebar.radio("🎨 Choose Theme Mode", ("🌞 Light", "🌙 Dark"))
if mode == "🌙 Dark":
    primary_color = "#00FFFF"
    background_color = "#0E1117"
    text_color = "white"
else:
    primary_color = "#000000"
    background_color = "#FFFFFF"
    text_color = "black"

st.markdown(f"""
    <style>
    body, .stApp {{
        background-color: {background_color};
        color: {text_color};
    }}
    .stTextInput > div > input,
    .stTextArea > div > textarea,
    .stSelectbox > div > div,
    .stExpander, .stButton button,
    .stDownloadButton button,
    .stMarkdown, .stCode, .stJson,
    .stTabs {{
        background-color: {background_color};
        color: {text_color};
        border-color: {primary_color};
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {text_color};
    }}
    </style>
""", unsafe_allow_html=True)

# Styling
st.markdown(f"""
    <style>
    body {{ background-color: {background_color}; color: {text_color}; }}
    .stApp {{ background-color: {background_color}; }}
    .big-title {{ font-size: 40px; font-weight: bold; color: {primary_color}; text-align: center; margin-bottom: 10px; }}
    .subtext {{ text-align: center; font-size: 16px; color: #AAAAAA; margin-bottom: 30px; }}
    </style>
""", unsafe_allow_html=True)

# Lottie title
lottie_ai = requests.get("https://assets2.lottiefiles.com/packages/lf20_j1adxtyb.json").json()
st.markdown('<div class="big-title fade-in">💻 CodeNova your coding buddy</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext fade-in">Your AI Coding Companion — Now with Multi-language support</div>', unsafe_allow_html=True)
st_lottie(lottie_ai, height=200, key="ai")


#Llm selection
# Groq models dictionary (unchanged)
groq_models = {
    "LLaMA 3-8B Instant": "llama-3.1-8b-instant",
    "LLaMA 3-70B Versatile": "llama-3.3-70b-versatile",
    "Command R+": "command-r-plus"
}

# Initialize session state
if "selected_model_name" not in st.session_state:
    st.session_state.selected_model_name = "LLaMA 3-70B Versatile"

with st.expander("🧠 Choose Your LLM Model", expanded=True):
    # ✨ Custom CSS with animation
    st.markdown("""
        <style>
        .model-button {
            display: inline-block;
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 16px;
            margin: 6px;
            border: 2px solid #00BFFF;
            text-align: center;
            cursor: pointer;
            transition: background-color 0.4s ease, color 0.4s ease, transform 0.2s ease;
        }
        .model-button:hover {
            transform: scale(1.05);
        }
        .active {
            background-color: #00BFFF;
            color: white;
        }
        .inactive {
            background-color: transparent;
            color: #00BFFF;
        }
        </style>
    """, unsafe_allow_html=True)

    # Create buttons in a responsive layout
    cols = st.columns(len(groq_models))
    for i, (model_name, _) in enumerate(groq_models.items()):
        with cols[i]:
            is_active = model_name == st.session_state.selected_model_name
            css_class = "model-button active" if is_active else "model-button inactive"
            button_html = f'<div class="{css_class}">{model_name}</div>'
            clicked = st.button(model_name, key=f"btn_{model_name}")
            st.markdown(button_html, unsafe_allow_html=True)
            if clicked:
                st.session_state.selected_model_name = model_name

# Get selected model ID
selected_model_name = st.session_state.selected_model_name
selected_model_id = groq_models[selected_model_name]

# Optional confirmation display
st.markdown(f"✅ **Selected Model:** `{selected_model_name}` (`{selected_model_id}`)")


# LangChain
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You're a highly skilled and reliable coding assistant. Your job is to return only clean, valid, executable, and production-ready code in response to the user's request. "
     "The code should follow best practices for readability, error handling, and performance in the chosen language. "
     "Do not include explanations, comments, or markdown formatting — return only the raw code. "
     "If the user's question is unclear or incomplete, make reasonable assumptions or return a minimal working solution."),
    ("user", "Question: {question}")
])
llm = ChatGroq(api_key=API_KEY, model_name=selected_model_id)
chain = prompt | llm | StrOutputParser()

# History
if "history" not in st.session_state:
    st.session_state.history = []
if os.path.exists("chat_memory.json"):
    try:
        with open("chat_memory.json", "r") as f:
            st.session_state.history = json.load(f)
    except json.JSONDecodeError:
        st.session_state.history = []

# Sidebar Save/Load API Calls
api_profile_file = "saved_api_calls.json"
if "saved_api_calls" not in st.session_state:
    if os.path.exists(api_profile_file):
        try:
            with open(api_profile_file, "r") as f:
                st.session_state.saved_api_calls = json.load(f)
        except json.JSONDecodeError:
            st.session_state.saved_api_calls = {}
    else:
        st.session_state.saved_api_calls = {}
profile_names = list(st.session_state.saved_api_calls.keys())
selected_profile = st.sidebar.selectbox("📂 Load Saved Profile", options=[""] + profile_names)
if selected_profile and selected_profile in st.session_state.saved_api_calls:
    saved = st.session_state.saved_api_calls[selected_profile]
    st.session_state["api_url"] = saved["url"]
    st.session_state["method"] = saved["method"]
    st.session_state["auth_token"] = saved.get("token", "")
    st.session_state["headers_input"] = json.dumps(saved.get("headers", {}), indent=2)
    st.session_state["body_input"] = json.dumps(saved.get("body", {}), indent=2)
new_profile_name = st.sidebar.text_input("💾 New Profile Name")
if st.sidebar.button("✅ Save Current API Request"):
    if new_profile_name:
        new_entry = {
            "url": st.session_state.get("api_url", ""),
            "method": st.session_state.get("method", "GET"),
            "token": st.session_state.get("auth_token", ""),
            "headers": json.loads(st.session_state.get("headers_input", "{}")),
            "body": json.loads(st.session_state.get("body_input", "{}")) if st.session_state.get("method") != "GET" else {}
        }
        st.session_state.saved_api_calls[new_profile_name] = new_entry
        with open(api_profile_file, "w") as f:
            json.dump(st.session_state.saved_api_calls, f, indent=2)
        st.sidebar.success(f"Saved as '{new_profile_name}' ✅")
    else:
        st.sidebar.warning("Enter a profile name to save.")

# Voice Input
voice_text = ""
if st.button("🎧 Speak Instead of typing"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ CodeNova is Listening...")
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
    try:
        voice_text = recognizer.recognize_google(audio)
        st.success(f"You said: {voice_text}")
    except sr.UnknownValueError:
        st.warning("Could not understand the audio.")
    except sr.RequestError as e:
        st.error(f"Speech Recognition error: {e}")

# Input
input_txt = st.text_input("💬 Ask your coding question:", value=voice_text if voice_text else "", key="user_input")

# Run Chain
if st.button("🚀 Generate Code", key="main_code_gen") and input_txt:
    with st.spinner("🤖 Generating response, please wait..."):
        response = chain.invoke({"question": input_txt})
        st.session_state.history.append((input_txt, response))
        with open("chat_memory.json", "w") as f:
            json.dump(st.session_state.history, f)
        st.success("✅ Response generated!")

        try:
            tts = gTTS(text=response, lang='en')
            voice_path = "voice_response.mp3"
            tts.save(voice_path)
            time.sleep(0.5)
            with open(voice_path, "rb") as audio_file:
                st.audio(audio_file.read(), format="audio/mp3")
        except Exception as e:
            st.warning(f"⚠️ Voice generation failed: {e}")

# Show History
if st.session_state.history:
    st.markdown("### 📜 Chat History")
    for idx, (q, a) in enumerate(reversed(st.session_state.history), 1):
        st.markdown(f"**Q{idx}:** {q}")
        lang = "python"
        st.code(a, language=lang)

# Export PDF
if st.button("📝 Export as PDF"):
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        lines = st.session_state.history[-1][1].split('\n')
        for line in lines:
            pdf.multi_cell(0, 10, txt=line.encode('latin-1', 'replace').decode('latin-1'))
        pdf_file = f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(pdf_file)
        with open(pdf_file, "rb") as f:
            st.download_button("Download .pdf", f, file_name=pdf_file, mime="application/pdf")
    except Exception as e:
        st.warning(f"⚠️ PDF export failed: {e}")

# Clear History
if st.button("🧹 Clear Chat History"):
    st.session_state.history = []
    if os.path.exists("chat_memory.json"):
        os.remove("chat_memory.json")
    if os.path.exists("voice_response.mp3"):
        os.remove("voice_response.mp3")
    st.rerun()
def styled_header(title, icon="🧪", color="#00BFFF"):
    st.markdown(
        f"""
        <div style="
            background-color: {color}20;
            border: 1px solid {color};
            padding: 10px 16px;
            font-size: 16px;
            font-weight: bold;
            border-radius: 8px;
            margin-top: 15px;
            color: white;
        ">
            {icon} {title}
        </div>
        """,
        unsafe_allow_html=True
    )


# 🔁 TWO-WAY CODE EXPLANATION TOOL
styled_header("Two-Way Code Explanation Tool", icon="🧠", color="#6366F1")
with st.expander("Expand", expanded=False):
    tabs = st.tabs(["🧵 Explain Code", "🛠 Generate Code", "📦 Generate Test Cases", "❗ Explain Errors"])
    with tabs[0]:
        st.markdown("### 🧵 Paste your code or upload a file:")
        code_input = st.text_area("📝 Paste Code", height=200, key="explain_code_input")
        code_file = st.file_uploader("📤 Or Upload Code File", type=["py", "cpp", "js", "java", "ts"], key="code_file")
        if code_file:
            uploaded_code = code_file.read().decode("utf-8")
            st.code(uploaded_code, language="python")
            code_input += f"\n# Uploaded File:\n{uploaded_code}"
        if st.button("🧠 Explain This Code"):
            with st.spinner("🔎 Analyzing code..."):
                explain_prompt = ChatPromptTemplate.from_messages([
                    ("system",
     "You're a helpful and experienced coding assistant. Carefully read the code provided by the user. "
     "Explain it line by line in a clear and concise manner, using simple language where possible. "
     "After the line-by-line explanation, provide a high-level summary describing what the code does overall. "
     "If the code uses any advanced or uncommon syntax, explain it briefly when it appears. "
     "Output only the explanation — do not reprint the code itself."),
    ("user", "{code}")
                ])
                explain_chain = explain_prompt | llm | StrOutputParser()
                explanation = explain_chain.invoke({"code": code_input})
                st.text_area("📘 Explanation", explanation, height=300)
    with tabs[1]:
        desc_input = st.text_area("📝 Describe what the code should do:", height=150, key="desc_input")
        if st.button("🚀 Generate Code", key="two_way_code_gen"):
            with st.spinner("⚙️ Generating code..."):
                gen_prompt = ChatPromptTemplate.from_messages([
                    ("system", 
     "You're a helpful and professional AI software engineer. Write clean, efficient, and production-quality code based on the user's description. "
     "Automatically choose the most appropriate programming language if not specified. "
     "Ensure the code is functional, idiomatic, and follows best practices for that language. "
     "If the task is ambiguous, make reasonable assumptions and add comments to clarify. "
     "Only output the code — no explanations, no markdown formatting."),
    ("user", "{desc}")
                ])
                gen_chain = gen_prompt | llm | StrOutputParser()
                code_output = gen_chain.invoke({"desc": desc_input})
                st.code(code_output, language="python")
    with tabs[2]:
        test_func = st.text_area("🧪 Paste your function here to generate tests:", height=180, key="test_input")
        if st.button("📦 Generate Test Cases"):
            with st.spinner("🧠 Writing tests for your function..."):
                test_prompt = ChatPromptTemplate.from_messages([
                    ("system",
     "You're a professional test engineer. Generate comprehensive and well-structured unit tests for the given function. "
     "Automatically detect the programming language and use the most appropriate testing framework (e.g., unittest or pytest for Python, Jest for JavaScript, JUnit for Java, Catch2 for C++, etc.). "
     "Cover valid cases, edge cases, invalid inputs, and exception handling. Mock external dependencies if necessary. "
     "Follow naming and structural best practices for the selected framework. Output only the complete test code without explanation or markdown."),
    ("user", "{func}")
                ])
                test_chain = test_prompt | llm | StrOutputParser()
                test_code = test_chain.invoke({"func": test_func})
                st.code(test_code, language="python")
    with tabs[3]:
        traceback_input = st.text_area("❗ Paste your error traceback:", height=180, key="error_input")
        if st.button("🔍 Explain Error"):
            with st.spinner("🛠️ Diagnosing error..."):
                error_prompt = ChatPromptTemplate.from_messages([
                    ("system",
     "You're a highly skilled debugging assistant. Analyze the following error message or traceback. "
     "Clearly explain what the error means in simple terms, identify the likely cause, and suggest specific steps to fix it. "
     "If the error includes a file and line number, use it to narrow down the issue. "
     "Provide only the explanation and solution — do not repeat the error message."),
    ("user", "{traceback}")
                ])
                error_chain = error_prompt | llm | StrOutputParser()
                error_explanation = error_chain.invoke({"traceback": traceback_input})
            st.text_area("📘 Explanation", error_explanation, height=250)

def estimate_time_complexity_with_gpt(code: str):
    llm = ChatGroq(
        temperature=0.2,
        api_key= API_KEY,
        model_name="llama3-8b-8192"
    )

    prompt = ChatPromptTemplate.from_template("""
You are a senior software engineer. Analyze the following Python function and estimate its time complexity in Big-O notation. Also give a 1-2 line explanation.

Function:
{code}

Format:
Time Complexity: O(...)
Explanation: ...
""")

    chain = prompt | llm | StrOutputParser()
    try:
        result = chain.invoke({"code": code})  

        return result
    except Exception as e:
        return f"❌ Error: {e}"


def plot_complexity_curve(big_o: str):
    x = np.linspace(1, 100, 100)

    complexity_map = {
        "O(1)": np.ones_like(x),
        "O(log n)": np.log2(x),
        "O(n)": x,
        "O(n log n)": x * np.log2(x),
        "O(n^2)": x ** 2,
        "O(n^3)": x ** 3,
        "O(2^n)": 2 ** x,
        "O(n!)": [math.factorial(int(i)) if i < 20 else np.nan for i in x],
    }

    if big_o not in complexity_map:
        st.warning(f"⚠️ Could not plot curve for: {big_o}")
        return

    y = complexity_map[big_o]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=big_o))
    fig.update_layout(
        title=f"Growth Curve: {big_o}",
        xaxis_title="Input Size (n)",
        yaxis_title="Operations",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

            
# 🧠 TIME COMPLEXITY ANALYZER SECTION
styled_header("Time Complexity Analyzer (AI-Powered)", icon="📏", color="#FF5733")
with st.expander("Expand",expanded=False):
    code_input = st.text_area("🔍 Paste your function code here:")
    if st.button("🧠 Analyze with GPT", key="analyze_complexity"):
        if code_input.strip():
            with st.spinner("Asking GPT to estimate time complexity..."):
                result = estimate_time_complexity_with_gpt(code_input)
                st.success("✅ GPT Analysis Complete")
                st.markdown(f"```\n{result.strip()}\n```")

                # Optional: try to extract Big-O notation for plotting
                match = re.search(r"O\([\w\s\^\!\*nlog]+\)", result)
                if match:
                    st.markdown("📊 Visualizing Time Complexity Growth")
                    plot_complexity_curve(match.group())
                else:
                    st.info("ℹ️ No recognizable Big-O notation found for plotting.")
        else:
            st.warning("⚠️ Please enter a valid Python function to analyze.")
            
#🧪API TESTING TOOL
def load_lottie():
    url = "https://assets9.lottiefiles.com/packages/lf20_jcikwtux.json"
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    return None

def api_testing_tool():
    styled_header("API Testing Tool (Advanced)", icon="🧪", color="#21C55D")
    st_lottie(load_lottie(), height=180, speed=1)

    if "api_history" not in st.session_state:
        st.session_state.api_history = []

    with st.expander("🛠️ API Configuration", expanded=True):
        col1, col2 = st.columns([2, 1])
        api_url = st.text_input("🔗 Enter API URL", value=st.session_state.get("api_url", ""))
        method = st.selectbox("HTTP Method", ["GET", "POST", "PUT", "DELETE"], index=0)
        auth_token = st.text_input("🔐 Bearer Token (Optional)", value=st.session_state.get("auth_token", ""), type="password")

        headers_input = st.text_area("🔧 Headers (JSON)", value=st.session_state.get("headers_input", '{\n  "Content-Type": "application/json"\n}'), height=120)
        body_input = st.text_area("📦 Request Body (JSON)", value=st.session_state.get("body_input", "{}"), height=150) if method != "GET" else ""

        expected_status = st.text_input("✅ Expected HTTP Status Code", value=st.session_state.get("expected_status", ""))
        expected_body = st.text_area("🧾 Expected JSON Response", value=st.session_state.get("expected_body", ""), height=150)

    if st.button("🚀 Send API Request"):
        try:
            headers = json.loads(headers_input)
        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON in headers:\n{str(e)}")
            return

        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        data = None
        if method != "GET":
            try:
                data = json.loads(body_input)
            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON in request body:\n{str(e)}")
                return

        try:
            with st.spinner("📡 Sending request..."):
                with httpx.Client(timeout=10.0) as client:
                    response = client.request(method=method, url=api_url, headers=headers, json=data)

            st.success(f"✅ Status Code: {response.status_code}")
            st.code(response.text, language="json")
            st.json(dict(response.headers))

            curl_parts = [f"curl -X {method} \"{api_url}\""]
            for k, v in headers.items():
                curl_parts.append(f"  -H \"{k}: {v}\"")
            if data:
                curl_parts.append(f"  -d '{json.dumps(data, indent=2)}'")
            curl_command = " \\\n".join(curl_parts)

            with st.expander("📋 View cURL Command"):
                st.code(curl_command, language="bash")
                st.button("📋 Copy to Clipboard (manual)", on_click=lambda: st.toast("Copied to clipboard (manually)!"))

            st.session_state.api_history.insert(0, {
                "url": api_url,
                "method": method,
                "status": response.status_code,
                "headers": headers,
                "body": data,
                "response": response.text
            })

            if expected_status:
                try:
                    if int(expected_status) == response.status_code:
                        st.success("✅ Status code matched.")
                    else:
                        st.error(f"❌ Expected {expected_status}, got {response.status_code}")
                except:
                    st.warning("⚠️ Expected status must be an integer.")

            if expected_body:
                try:
                    if json.loads(expected_body) == response.json():
                        st.success("✅ Response matches expected JSON.")
                    else:
                        st.error("❌ Response body mismatch.")
                except:
                    st.warning("⚠️ Error comparing expected and actual JSON.")

        except Exception as e:
            st.error(f"❌ Request failed: {str(e)}")

    styled_header("📜 Request History", icon="🧾", color="#4F46E5")
    if st.session_state.api_history:
        st.markdown("### 🗂️ View and manage history")
        if st.button("🧹 Clear History"):
            st.session_state.api_history = []
            st.success("🧹 History cleared successfully.")
            st.rerun()

        for i, entry in enumerate(st.session_state.api_history[:10]):
            with st.expander(f"{entry['method']} {entry['url']} → {entry['status']}"):
                st.code(json.dumps(entry['headers'], indent=2), language="json")
                if entry["body"]:
                    st.markdown("#### Request Body")
                    st.code(json.dumps(entry["body"], indent=2), language="json")
                st.markdown("#### Response")
                st.code(entry["response"], language="json")

        history_json = json.dumps(st.session_state.api_history, indent=2)
        st.download_button(
            label="⬇️ Download History as JSON",
            data=history_json,
            file_name="api_history.json",
            mime="application/json"
        )
    else:
        st.info("No request history yet.")
api_testing_tool()   
    

api_list = [
    {"name": "CodeNova Auth", "url": "https://jsonplaceholder.typicode.com/posts", "method": "GET"},
    {"name": "User Endpoint", "url": "https://jsonplaceholder.typicode.com/users", "method": "GET"},
    # Add more APIs here
]

# ------------------ MONITOR FUNCTION ------------------
def monitor_api(endpoint):
    try:
        start = time.time()
        response = httpx.request(endpoint["method"], endpoint["url"], timeout=10)
        duration = round(time.time() - start, 2)
        return {
            "name": endpoint["name"],
            "status_code": response.status_code,
            "response_time": duration,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ok": response.status_code < 400
        }
    except Exception as e:
        return {
            "name": endpoint["name"],
            "status_code": None,
            "response_time": None,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ok": False,
            "error": str(e)
        }
        

# ------------------ SECTION UI ------------------
styled_header("📡 REST API Monitoring", icon="📶", color="#10B981")  # ✅ Green header

# Inject adaptive dark-mode friendly styling
st.markdown("""
    <style>
    .api-monitor-box {
        border: 2px solid rgba(34, 197, 94, 0.5);  /* Green border with transparency */
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 15px;
        background-color: rgba(255, 255, 255, 0.05);  /* Light-transparent for dark theme */
        color: inherit;  /* Inherit text color from theme */
    }
    .api-monitor-box.error {
        border-color: rgba(239, 68, 68, 0.5);  /* Red border */
    }
    </style>
""", unsafe_allow_html=True)

with st.expander("🧪 Enter API Details to Monitor", expanded=True):
    api_name = st.text_input("API Name", placeholder="e.g. CodeNova Auth")
    api_url = st.text_input("API URL", placeholder="https://yourapi.com/endpoint")
    method = st.selectbox("Request Method", ["GET", "POST", "PUT", "DELETE"])

    if st.button("🔁 Check API Health"):
        if api_name and api_url:
            with st.spinner("📡 Monitoring API..."):
                res = monitor_api({"name": api_name, "url": api_url, "method": method})

            status_color = "#22C55E" if res["ok"] else "#EF4444"
            box_class = "api-monitor-box" + (" error" if not res["ok"] else "")

            st.markdown(f"""
                <div class="{box_class}">
                    <h4>🌐 <b>{res['name']}</b></h4>
                    <p><b>🕒 Time:</b> {res['timestamp']}</p>
                    <p><b>📤 Status Code:</b> {res['status_code']}</p>
                    <p><b>⚡ Response Time:</b> {res['response_time']}s</p>
                    <p><b>Status:</b> {'✅ API is Healthy' if res['ok'] else f"❌ API Failed: {res.get('error', 'Unknown Error')}"} </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Please provide both a valid name and URL.")


       
#Unit Testing Generator
styled_header("Unit Testing Generator", icon="📄", color="#F59E0B")
with st.expander("Expand",expanded=False):
    st.markdown("### 🧪 Paste a function/component to generate unit tests:")

    # Language/Framework Selector
    language = st.selectbox(
        "🔤 Choose the language or framework:",
        ["Python", "Java", "JavaScript", "C++", "Go", "Rust", "C#", "React", "Next.js", "Node.js"],
        key="unit_test_lang"
    )

    unit_input = st.text_area("🧩 Function/Component Code:", height=200, key="unit_code_input")

    # Extension mapping
    file_extensions = {
        "Python": "py", "Java": "java", "JavaScript": "js", "C++": "cpp", "Go": "go",
        "Rust": "rs", "C#": "cs", "React": "jsx", "Next.js": "js", "Node.js": "js"
    }

    lang_prompts = {
        "Python": (
                    "You're a Python test engineer. Generate unit tests using the unittest framework for the provided function."
                ),
                "Java": (
                    "You're a Java developer. Generate JUnit 5 test cases with descriptive test methods."
                ),
                "JavaScript": (
                    "You're a JavaScript developer. Use the Jest framework to write unit tests for the provided function."
                ),
                "C++": (
                    "You're a C++ developer. Generate Google Test (gtest) unit tests for the provided function."
                ),
                "Go": (
                    "You're a Go developer. Generate Go unit tests using Go's testing package (`testing`).\n"
                    "- Use `t.Run` and proper table-driven tests."
                ),
                "Rust": (
                    "You're a Rust developer. Generate Rust unit tests using `#[cfg(test)]` and `#[test]`.\n"
                    "- Cover typical and edge cases."
                ),
                "C#": (
                    "You're a C# developer. Generate unit tests using NUnit or MSTest for the provided method."
                ),
                "React": (
                    "You're a frontend developer. Given a React component, generate unit tests using `React Testing Library` and `Jest`.\n"
                    "- Test rendering, props, events, and effects."
                ),
                "Next.js": (
                    "You're a Next.js developer. Generate tests using Jest + React Testing Library for components and API routes.\n"
                    "- If it's an API route, test responses using `supertest` or similar."
                ),
                "Node.js": (
                    "You're a Node.js backend developer. Generate unit tests using `Mocha` and `Chai` or `Jest`.\n"
                    "- Use `describe`, `it`, and assertions like `expect`, `should`, or `assert`."
                )
            }

            
    lang_map = {
        "Python": "python", "Java": "java", "JavaScript": "javascript", "C++": "cpp",
        "Go": "go", "Rust": "rust", "C#": "csharp", "React": "javascript",
        "Next.js": "javascript", "Node.js": "javascript"
    }

    if st.button("🧪 Generate Unit Tests") and unit_input.strip():
        with st.spinner("🧠 Generating unit tests..."):

            unit_prompt = ChatPromptTemplate.from_messages([
                ("system", lang_prompts[language]),
                ("user", "{func}")
            ])
            unit_chain = unit_prompt | llm | StrOutputParser()
            unit_output = unit_chain.invoke({"func": unit_input})

            st.markdown("### ✅ Generated Unit Test:")
            st.code(unit_output, language=lang_map[language])

            # ------------------------------
            # Export Logic: Downloadable file
            # ------------------------------
            file_name = f"unit_test.{file_extensions[language]}"
            file_content = unit_output.encode("utf-8")
            file_buffer = BytesIO(file_content)

            # 📁 Download section
            with st.container():
                st.markdown("#### ⬇️ Export Test File:")
                st.download_button(
                    label="📤 Download Test File",
                    data=file_buffer,
                    file_name=file_name,
                    mime="text/plain",
                    use_container_width=True
                )
