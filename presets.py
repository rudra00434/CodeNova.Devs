import streamlit as st

# ✅ Save Preset UI
def save_preset_ui(tab_id):
    st.subheader("💾 Save Preset")
    preset_name = st.text_input(f"Preset Name for {tab_id}", key=f"{tab_id}_save_name")

    if st.button(f"💾 Save {tab_id} Preset", key=f"{tab_id}_save_btn"):
        if "api_presets" not in st.session_state:
            st.session_state["api_presets"] = {}

        st.session_state["api_presets"][preset_name] = {
            "url": st.session_state.get(f"{tab_id}_api_url", ""),
            "method": st.session_state.get(f"{tab_id}_method", ""),
            "headers": st.session_state.get(f"{tab_id}_headers_input", ""),
            "body": st.session_state.get(f"{tab_id}_body_input", ""),
            "token": st.session_state.get(f"{tab_id}_auth_token", ""),
        }
        st.success(f"Preset '{preset_name}' saved.")


# ✅ Load Preset UI
def load_preset_ui(tab_id):
    st.subheader(f"📂 Load Preset for {tab_id}")
    if "api_presets" in st.session_state:
        options = list(st.session_state["api_presets"].keys())
        preset = st.selectbox("Choose Preset", options, key=f"{tab_id}_select")
        
        if st.button(f"🔄 Load Preset", key=f"{tab_id}_load_btn"):
            p = st.session_state["api_presets"][preset]
            st.session_state[f"{tab_id}_api_url"] = p["url"]
            st.session_state[f"{tab_id}_method"] = p["method"]
            st.session_state[f"{tab_id}_headers_input"] = p["headers"]
            st.session_state[f"{tab_id}_body_input"] = p["body"]
            st.session_state[f"{tab_id}_auth_token"] = p["token"]
            st.success(f"Preset '{preset}' loaded into {tab_id}")


# ✅ Required for main.py — returns list of preset environments
def get_presets():
    # Option 1: If presets are saved in session state
    if "api_presets" in st.session_state:
        return [
            {"name": name, "base_url": preset.get("url", "")}
            for name, preset in st.session_state["api_presets"].items()
        ]
    
    # Option 2: Default fallback presets
    return [
        {"name": "Dev", "base_url": "http://localhost:8000"},
        {"name": "Prod", "base_url": "https://api.example.com"},
    ]
