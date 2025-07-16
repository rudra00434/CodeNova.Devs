import streamlit as st
from datetime import datetime

# Store request and response in session state
def store_request(url, method, headers, body, status, response):
    if "api_history" not in st.session_state:
        st.session_state["api_history"] = []

    st.session_state["api_history"].append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "url": url,
        "method": method,
        "headers": headers,
        "body": body,
        "status": status,
        "response": response
    })

# Render the history tab (used in main.py)
def render_history_tab():
    st.subheader("📜 API Request History")
    history = st.session_state.get("api_history", [])
    
    if not history:
        st.info("No requests made yet.")
        return

    for entry in reversed(history[-10:]):  # Show last 10 requests
        with st.expander(f"{entry['timestamp']} | {entry['method']} | {entry['status']}"):
            st.markdown(f"**🔗 URL:** {entry['url']}")
            st.markdown(f"**📨 Request Body:**")
            st.code(entry.get("body", ""), language="json")
            st.markdown(f"**📥 Response:**")
            st.code(entry.get("response", ""), language="json")
