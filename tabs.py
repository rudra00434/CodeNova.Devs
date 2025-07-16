
import streamlit as st
import requests
import json
from API_testing_tool.history import store_request
from requests.auth import HTTPBasicAuth

def render_api_tab(tab_id, base_url):
    url_key = f"{tab_id}_api_url"
    method_key = f"{tab_id}_method"
    headers_key = f"{tab_id}_headers"
    body_key = f"{tab_id}_body"
    response_key = f"{tab_id}_response"
    auth_type_key = f"{tab_id}_auth_type"
    api_key_key = f"{tab_id}_api_key"
    bearer_token_key = f"{tab_id}_bearer_token"
    basic_user_key = f"{tab_id}_basic_user"
    basic_pass_key = f"{tab_id}_basic_pass"

    st.subheader(f"🔗 API Request - {tab_id}")

    st.text_input("Request URL", key=url_key, value=base_url)
    st.selectbox("Method", ["GET", "POST", "PUT", "DELETE"], key=method_key)

    # Authentication Section
    st.markdown("### 🔐 Authentication")
    auth_type = st.selectbox(
        "Auth Type",
        options=["None", "API Key", "Bearer Token", "Basic Auth"],
        key=auth_type_key
    )

    if auth_type == "API Key":
        st.text_input("API Key Header Name", value="x-api-key", key=api_key_key)
    elif auth_type == "Bearer Token":
        st.text_input("Bearer Token", type="password", key=bearer_token_key)
    elif auth_type == "Basic Auth":
        st.text_input("Username", key=basic_user_key)
        st.text_input("Password", type="password", key=basic_pass_key)

    st.markdown("### 📬 Headers & Body")
    st.text_area("Headers (JSON)", key=headers_key, value='{\n  "Content-Type": "application/json"\n}')
    st.text_area("Request Body (JSON)", key=body_key, height=150)

    if st.button("🚀 Send Request", key=f"{tab_id}_send"):
        try:
            headers = json.loads(st.session_state[headers_key])
            url = st.session_state[url_key]
            method = st.session_state[method_key]
            body = st.session_state[body_key]
            auth_type_val = st.session_state[auth_type_key]

            auth = None
            if auth_type_val == "API Key":
                api_key_header = st.session_state.get(api_key_key, "x-api-key")
                headers[api_key_header] = st.session_state.get(api_key_key, "")
            elif auth_type_val == "Bearer Token":
                headers["Authorization"] = f"Bearer {st.session_state.get(bearer_token_key, '')}"
            elif auth_type_val == "Basic Auth":
                auth = HTTPBasicAuth(
                    st.session_state.get(basic_user_key, ""),
                    st.session_state.get(basic_pass_key, "")
                )

            if method == "GET":
                res = requests.get(url, headers=headers, auth=auth)
            elif method == "POST":
                res = requests.post(url, headers=headers, data=body, auth=auth)
            elif method == "PUT":
                res = requests.put(url, headers=headers, data=body, auth=auth)
            elif method == "DELETE":
                res = requests.delete(url, headers=headers, auth=auth)

            st.session_state[response_key] = res.text
            st.session_state[f"{tab_id}_status_code"] = res.status_code
            store_request(url, method, headers, body, res.status_code, res.text)

        except Exception as e:
            st.session_state[response_key] = f"❌ Error: {e}"

    if response_key in st.session_state:
        st.subheader("📦 Response")
        st.code(st.session_state[response_key], language="json")

        