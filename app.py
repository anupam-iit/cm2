import streamlit as st
from transformers import pipeline

# Page setup
st.set_page_config(page_title="BERT Sentiment Analyzer", page_icon="🧠", layout="centered")

# Custom Header
st.markdown("""
    <h1 style='text-align: center; color: #4B8BBE;'>🤖 BERT Sentiment Analyzer</h1>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

analyzer = load_model()

# User Input
st.markdown("### ✍️ Enter some text below:")
text = st.text_area("Type here...", height=150)

if st.button("🔍 Analyze"):
    if text:
        with st.spinner("Analyzing..."):
            result = analyzer(text)[0]
            label = result['label']
            score = result['score']
            
            color = "green" if label == "POSITIVE" else "red"
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                <h3>📊 Result</h3>
                <p><strong>Sentiment:</strong> <span style='color:{color}'>{label}</span></p>
                <p><strong>Confidence Score:</strong> {score:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter some text.")

st.markdown("---")

# Custom Footer
st.markdown("""
    <p style='text-align: center; font-size: 13px; color: gray;'>
        Developed by Anupam Roy, A Student of IIT, JU
    </p>
""", unsafe_allow_html=True)
