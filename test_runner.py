import streamlit as st

def run_tests(code, response):
    try:
        local = {"response": response}
        exec(code, {}, local)
        st.success("✅ Test Passed!")
    except Exception as e:
        st.error(f"❌ Test Failed: {e}")
