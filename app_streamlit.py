import streamlit as st
import os
import tempfile
import subprocess
from PIL import Image

st.set_page_config(page_title="StableVITON Unpair Inference", layout="wide")
st.title("StableVITON: Virtual Try-On (Unpair Mode)")

col1, col2, col3 = st.columns(3)

with col1:
    st.header("1. Upload Person Image")
    person_file = st.file_uploader("Choose a person image", type=["jpg", "jpeg", "png"], key="person")
    if person_file:
        person_img = Image.open(person_file)
        st.image(person_img, caption="Person Image", use_column_width=True)

with col2:
    st.header("2. Upload Cloth Image (Unpair)")
    cloth_file = st.file_uploader("Choose a cloth image", type=["jpg", "jpeg", "png"], key="cloth")
    if cloth_file:
        cloth_img = Image.open(cloth_file)
        st.image(cloth_img, caption="Cloth Image", use_column_width=True)

with col3:
    st.header("3. Output: Virtual Try-On Result")
    output_placeholder = st.empty()

run_button = st.button("Run Inference")

if run_button and person_file and cloth_file:
    with st.spinner("Running inference, please wait..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save uploaded images
            person_path = os.path.join(tmpdir, "person.jpg")
            cloth_path = os.path.join(tmpdir, "cloth.jpg")
            person_img.save(person_path)
            cloth_img.save(cloth_path)

            # Prepare input directory structure as expected by your dataset/inference
            # For this example, we assume you need to place them in a folder structure
            # and create a test_pairs.txt file for unpair mode
            data_root = os.path.join(tmpdir, "zalando-hd-resized")
            test_dir = os.path.join(data_root, "test")
            cloth_dir = os.path.join(data_root, "test")
            os.makedirs(test_dir, exist_ok=True)
            os.makedirs(cloth_dir, exist_ok=True)
            # Save images with unique names
            person_fn = "00001_00.jpg"
            cloth_fn = "10001_00.jpg"
            person_dst = os.path.join(test_dir, person_fn)
            cloth_dst = os.path.join(cloth_dir, cloth_fn)
            person_img.save(person_dst)
            cloth_img.save(cloth_dst)
            # Create test_pairs.txt for unpair
            pairs_txt = os.path.join(data_root, "test_pairs.txt")
            with open(pairs_txt, "w") as f:
                f.write(f"{person_fn} {cloth_fn}\n")

            # Output directory
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir, exist_ok=True)

            # Call inference.py as subprocess
            command = [
                "python", "inference.py",
                "--config_path", "configs/VITONHD.yaml",
                "--model_load_path", "VITONHD_PBE_pose.ckpt",
                "--batch_size", "1",
                "--data_root_dir", data_root,
                "--unpair",
                "--save_dir", output_dir
            ]
            subprocess.run(command, check=True)

            # Find output image
            result_dir = os.path.join(output_dir, "unpair")
            result_files = [f for f in os.listdir(result_dir) if f.endswith(".jpg")]
            if result_files:
                result_img_path = os.path.join(result_dir, result_files[0])
                result_img = Image.open(result_img_path)
                col3.image(result_img, caption="Virtual Try-On Result", use_column_width=True)
            else:
                col3.warning("No output image found.")
