import streamlit as st

st.set_page_config(page_title="StableVITON reference", layout="wide")
st.title("StableVITON Reference: Virtual Try-On")

col1, col2, col3 = st.columns(3)

with col1:
    st.header("Step 1: Upload a garment image")
    st.file_uploader("Drop image here", type=["jpg", "jpeg", "png"], key ="person")
with col2:
    st.header("Step 2: Upload a person image")
    st.file_uploader("Drop image here", type=["jpg", "jpeg", "png"], key ="cloth")
with col3:
    st.header("Step 3: Press 'Generate' to get the result")
    output_placeholder = st.empty()
    run_button = st.button("Generate")
