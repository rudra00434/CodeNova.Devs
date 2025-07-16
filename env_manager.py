import streamlit as st

def select_environment():
    envs = {
        "Local": "http://localhost:8000",
        "Staging": "https://staging.api.com",
        "Prod": "https://api.example.com"
    }
    return envs[st.selectbox("🌍 Environment", list(envs.keys()))]
