# CodeNova.Devs
# CodeNova-Gen_AI_Project
CodeNova is a Streamlit-based chatbot powered by LangChain + Groq API that helps you generate multi-language code instantly using LLMs like Gemma:7b, LLaMA 3, and Mistral. Now supercharged with voice input, voice assistant, chat history, dark/light mode, and PDF/TXT exports , REST API testing Postman architecture , REST API monitoring , UNIT testing , Time complexity Analyzer etc multiple features .It can make your development smoother by giving🌐Multi-language Code Output
🔮 CodeNova - Your Real-Time AI Coding Buddy with real time feedback generator
CodeNova is a beautifully designed, real-time AI coding assistant built with Streamlit, LangChain, and Groq API. It supports multi-language code generation (Python, Java, C++, etc.) and provides interactive features for an enhanced developer experience.
<h1 align="center">Hi 👋, I'm Rudranil Goswami</h1>
<h3 align="center">A Generative AI enthusiastic and passionate python Developer</h3>

<p align="left"> <img src="https://komarev.com/ghpvc/?username=rudra00434&label=Profile%20views&color=0e75b6&style=flat" alt="rudra00434" /> </p>

- 🔭 I’m currently working on **CodeNova : Genai project**

- 🌱 I’m currently learning **Langchain 🔗, Langgraph 🌐**

<h3 align="left">Connect with me for Collaboration</h3>
<p align="left">
</p>

<h3 align="left">Languages and Tools:</h3>
<p align="left">
  <a href="https://www.python.org" target="_blank" rel="noreferrer">
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/>
  </a>
  
  <a href="https://streamlit.io/" target="_blank" rel="noreferrer">
    <img src="https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.svg" alt="streamlit" width="100" height="40"/>
  </a>
  
  <a href="https://www.langchain.com/" target="_blank" rel="noreferrer">
    <img src="https://avatars.githubusercontent.com/u/139895288?s=200&v=4" alt="langchain" width="40" height="40"/>
  </a>
  
  <a href="https://groq.com/" target="_blank" rel="noreferrer">
    <img src="https://avatars.githubusercontent.com/u/106233137?s=200&v=4" alt="groq" width="40" height="40"/>
  </a>

  <a href="https://pypi.org/project/gTTS/" target="_blank" rel="noreferrer">
    <img src="https://img.icons8.com/fluency/48/audio-wave.png" alt="gtts" width="40" height="40"/>
  </a>

  <a href="https://pypi.org/project/SpeechRecognition/" target="_blank" rel="noreferrer">
    <img src="https://img.icons8.com/color/48/microphone--v1.png" alt="speech recognition" width="40" height="40"/>
  </a>

  <a href="https://pyfpdf.github.io/fpdf2/" target="_blank" rel="noreferrer">
    <img src="https://img.icons8.com/external-flat-icons-inmotus-design/67/external-pdf-graphic-design-flat-icons-inmotus-design.png" alt="fpdf" width="40" height="40"/>
  </a>
</p>


🚀 Key Features — CodeNova (AI-Powered Developer Assistant)
💬 Chat & Language Support
🧠 Groq API-Powered LLMs — Choose from LLaMA3-8B, LLaMA3-70B, Mixtral, or Gemma models.

🌍 Multi-Language Support — Ask questions and generate code in Python, C++, Java, JavaScript, TypeScript, etc.

💡 Multi-Turn Chat Memory — Keeps track of past questions for better context and continuity.

🔁 Voice Input & Output — Speak your query, and hear the AI's response (powered by SpeechRecognition & gTTS).

🧠 Developer SuperTools (Built-in)
🧵 Two-Way Code Explanation Tool

Paste or upload code to get line-by-line explanations.

Describe functionality to generate complete code.

Generate unit tests or explain traceback errors.

📈 Time Complexity Analyzer

Paste a function and get Big-O estimation with a growth curve visualization.

🧪 API Testing Console
Send real-time GET, POST, PUT, DELETE requests.

Add auth tokens, custom headers, and JSON bodies.

Save and load API profiles for fast testing.

View cURL equivalents for easy CLI integration.

 _____REST API MONITORING ____
you can check you your API is healthy or not by giving input your API endpoint
it will show you the api time table (updated feature) of CodeNova

📦 Utilities & Export Options
🧾 PDF Export — Download responses or explanations as formatted PDF files.

🧠 Session Saving — Saves your chat history to chat_memory.json locally.

🔊 Voice Feedback — Get AI responses spoken aloud (great for accessibility).

🎨 UI & Customization
🌗 Light/Dark Theme Toggle

🎞️ Lottie Animations & Custom Fonts

📥 Drag & Drop File Uploads (for .py, .pdf, .txt, .cpp, etc.)

🧽 Clear Chat Button — Reset session memory anytime.


📦Codenova-chatbot file structure 
 ┣📜main.py
 ┣📜requirements.txt
 ┣📜chat_memory.json
 ┗📂assets (optional: animations, icons, etc.)
🔐 <h3 align="left">requirements.txt:</h3>

              streamlit
                |
                langchain
                  |
                langchain-community
                  |
                  langchain-core
                   |
                  Groq API
                   |
                  fpdf
                   |
                 gTTS
                   |
                SpeechRecognition
                  |
                 streamlit-lottie ai
                  |
                  httpx -(for api testing tool)
                  |
                  requests
                  |
                  plotly (from Time complexity graph)
                  |
                  numpy
                 


<h3 align="left">Groq API Setup:</h3>



    Get your API key from: https://console.groq.com
    Set it directly in code (or use st.secrets, .env, or config files for security):
    llm = ChatGroq(api_key="your_groq_api_key", model_name="llama3-70b-8192")

                
 ## ⚙️ Installation

```bash
# Clone this repository
git clone https://github.com/your-username/codenova-chatbot.git
cd codenova-chatbot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Start the app
streamlit run main.py
```
# App Visuals 
