import streamlit as st
from Inference.classifierPredict import predict


st.title("🧠 Brain Tumour Analysis Using MRI Scans")

uploaded_file = st.file_uploader(
    "Upload MRI Image", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Please upload your MRI scan")
else:
    st.success("File uploaded successfully")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Input Image")
        st.image(uploaded_file, use_column_width=True)

    with col2:
        st.markdown("### Prediction")

        if st.button("Run Analysis"):
            with st.spinner("Analyzing MRI..."):
                try:
                    prediction = predict(uploaded_file)

                    # Highlight result
                    if prediction == "notumor":
                        st.success(f"{prediction.upper()}")
                    else:
                        st.error(f"{prediction.upper()}")

                except Exception as e:
                    st.error(f"Error: {e}")