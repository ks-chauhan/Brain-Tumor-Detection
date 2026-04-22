import streamlit as st
from Inference.segment import predict_mask
from PIL import Image
import numpy as np

st.title("🧠 Brain Tumour Segmentation Using MRI Scans")

uploaded_file = st.file_uploader(
    "Upload MRI Image", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Please upload your MRI scan")
else:
    st.success("File uploaded successfully")

    if st.button("Run Analysis"):
        with st.spinner("Analyzing MRI..."):
            try:
                overlay, mask = predict_mask(uploaded_file)

                input_img = Image.open(uploaded_file).convert("L").resize((128,128))
                input_img = np.array(input_img)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("### Input")
                    st.image(input_img, use_column_width=True, clamp=True)

                with col2:
                    st.markdown("### Prediction Mask")
                    st.image(mask, use_column_width=True, clamp=True)

                with col3:
                    st.markdown("### Overlay")
                    st.image(overlay, use_column_width=True)

            except Exception as e:
                st.error(f"Error: {e}")