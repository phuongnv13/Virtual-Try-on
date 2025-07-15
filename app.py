import streamlit as st
from PIL import Image
import subprocess

st.set_page_config(page_title="StableVITON reference", layout="wide")
st.title("StableVITON Reference: Virtual Try-On")

person_path = "person.jpg"
cloth_path = "cloth.jpg"

col1, col2, col3 = st.columns(3)

with col1:
    st.header("Step 1: Upload a garment image")
    person_file = st.file_uploader("Drop image here", type=["jpg", "jpeg", "png"], key ="person")
    if person_file:
        person_img = Image.open(person_file)
        person_img.save(person_path)
        st.image(person_img, caption="Person Image", use_column_width=True)
with col2:
    st.header("Step 2: Upload a person image")
    cloth_file = st.file_uploader("Drop image here", type=["jpg", "jpeg", "png"], key ="cloth")
    if cloth_file:
        cloth_img = Image.open(cloth_file)
        cloth_img.save(cloth_path)
        st.image(cloth_img, caption="Cloth Image", use_column_width=True)
with col3:
    st.header("Step 3: Press 'Generate' to get the result")
    output_placeholder = st.empty()

run_button = st.button("Generate")

if run_button and person_file and cloth_file:
    with st.spinner("Running inference, please wait...", show_time=True):
        command = [
            "python", "single_inference.py",
            "--img_name", person_path,
            "--cloth_name", cloth_path,
            "--output_name", "output_image.jpg",
            "--data_type", "test",
            "--repaint",
        ]
        subprocess.run(command, check=True)

        if "./single_samples/output_image.jpg":
            col3.image("./single_samples/output_image.jpg", caption="Virtual Try-On Result", use_column_width=True)
        else:
            st.warning("No output image found. Please check the inference process.")

