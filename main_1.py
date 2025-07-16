import streamlit as st
from .presets import get_presets
from .tabs import render_api_tab
from .history import render_history_tab

def render_api_testing_tool() -> None:
    st.set_page_config(page_title="API Testing Tool", layout="wide")
    st.title("🧪 API Testing Tool")

    # Interface selection: Tabbed or Sidebar view
    interface_mode = st.radio("Select Interface", ["Tabbed View", "Sidebar View"], horizontal=True)

    # Load environments from presets
    environments = get_presets()

    if not environments:
        st.warning("⚠️ No presets found. Please define at least one environment in presets.py")
        return

    if interface_mode == "Tabbed View":
        tabs = st.tabs([env['name'] for env in environments])
        for idx, env in enumerate(environments):
            with tabs[idx]:
                render_api_tab(f"env{idx}", env['base_url'])

    elif interface_mode == "Sidebar View":
        env_names = [env['name'] for env in environments]
        selected_env_name = st.sidebar.selectbox("Choose Environment", env_names)
        selected_env = next(env for env in environments if env["name"] == selected_env_name)
        render_api_tab("sidebar_env", selected_env['base_url'])

    st.divider()
    with st.expander("📜 View API Request History", expanded=False):
        render_history_tab()
