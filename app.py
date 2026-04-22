import streamlit as st
import sys
import os
import importlib.util

st.set_page_config(
    page_title="Supply Chain Predictor",
    page_icon="SC",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Sidebar
st.sidebar.markdown("## Supply Chain Predictor")
st.sidebar.markdown("---")

PAGES = [
    "Dashboard",
    "Data Explorer",
    "Predict Disruption",
    "Drift Detection",
    "Model Diagnostics",
    "About",
]

PAGE_FILES = {
    "Dashboard": "dashboard",
    "Data Explorer": "data_explorer",
    "Predict Disruption": "predictor",
    "Drift Detection": "drift_detection",
    "Model Diagnostics": "diagnostics",
    "About": "about",
}

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Dashboard"

for label in PAGES:
    is_active = st.session_state["current_page"] == label
    if st.sidebar.button(
        label,
        key="nav_" + label,
        type="primary" if is_active else "secondary",
        width="stretch"
    ):
        st.session_state["current_page"] = label
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("Python | Scikit-learn | Streamlit")

# Load and run page
page = st.session_state["current_page"]
file_name = PAGE_FILES[page]
page_path = os.path.join(BASE_DIR, "pages", file_name + ".py")

try:
    spec = importlib.util.spec_from_file_location(file_name, page_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.render()
except Exception as e:
    st.error("Error loading page: " + page)
    st.exception(e)
