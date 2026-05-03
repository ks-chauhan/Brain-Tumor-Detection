import streamlit as st
from Inference.classifierPredict import predict as predict_mri
from Inference.ctClassifier import predict as predict_ct

st.title("🧠 Brain Tumour Analysis Using MRI & CT Scans")

# Scan type selector
scan_type = st.radio(
    "Select Scan Type",
    ["MRI", "CT Scan"],
    horizontal=True
)

uploaded_file = st.file_uploader(
    f"Upload {scan_type} Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info(f"Please upload your {scan_type} image")
else:
    st.success("File uploaded successfully")

    col1, col2 = st.columns(2)

    # LEFT: Input Image
    with col1:
        st.markdown("### Input Image")
        st.image(uploaded_file, use_column_width=True)

    # RIGHT: Prediction
    with col2:
        st.markdown("### Prediction")

        if st.button("Run Analysis", use_container_width=True):

            # MRI FLOW
            if scan_type == "MRI":
                with st.spinner("Analyzing MRI..."):
                    try:
                        prediction = predict_mri(uploaded_file)

                        if prediction == "notumor":
                            st.success(f"✅ {prediction.upper()}")
                        else:
                            st.error(f"⚠️ {prediction.upper()}")

                    except Exception as e:
                        st.error(f"❌ Error: {e}")

            # CT FLOW
            elif scan_type == "CT Scan":
                with st.spinner("Analyzing CT Scan..."):
                    try:
                        prediction = predict_ct(uploaded_file)

                        if prediction.lower() in ["notumor", "no tumor"]:
                            st.success(f"✅ {prediction.upper()}")
                        else:
                            st.error(f"⚠️ {prediction.upper()}")

                    except Exception as e:
                        st.error(f"❌ Error: {e}")