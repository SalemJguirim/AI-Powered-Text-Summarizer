from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st

# Streamlit UI Setup
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📚",
    layout="centered"
)

# Sidebar
with st.sidebar:
    st.title("AI Text Summarizer")
    st.markdown("""
    **Features:**
    - Summarize long texts quickly and efficiently.
    - Powered by Hugging Face's BART model.
    - Suitable for articles, research papers, and more.
    """)
    st.info("Tip: Input at least a few sentences for the best results!")

# Main App Title
st.title("📚 AI-Powered Text Summarizer")
st.markdown("""
    Welcome to the **AI-Powered Text Summarizer**!  
    This app uses a pre-trained BART model to generate concise summaries of long texts.  
    Enter your text below or upload a file and click **Generate Summary**.
""")

# Step 1: Load Model and Tokenizer
model_name = "facebook/bart-large-cnn"  # Pre-trained model
try:
    with st.spinner("Loading the model..."):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")

# Step 2: Input Text or Upload File
st.header("📝 Input Your Text or Upload a File")
input_mode = st.radio("Select Input Mode", ["Enter Text", "Upload File"])
input_text = ""

if input_mode == "Enter Text":
    input_text = st.text_area(
        "Enter the text you'd like to summarize below:",
        placeholder="Paste your text here (e.g., a news article, report, or document).",
        height=200
    )
else:
    uploaded_file = st.file_uploader("Upload a .txt file for summarization:", type=["txt"])
    if uploaded_file is not None:
        input_text = uploaded_file.read().decode("utf-8")
        st.text_area("File Content", input_text, height=200, disabled=True)

# Step 3: Generate Summary
if st.button("🚀 Generate Summary"):
    if input_text.strip():
        with st.spinner("Generating summary..."):
            try:
                # Tokenize the input text
                inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=1024)

                # Generate the summary
                summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

                # Display the summary
                st.header("📖 Generated Summary")
                st.success(summary)

                # Option to Download Summary
                st.download_button(
                    label="📥 Download Summary",
                    data=summary,
                    file_name="summary.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"Error during summarization: {str(e)}")
    else:
        st.warning("Please enter some text or upload a file to summarize!")

# Footer
st.markdown("---")
st.markdown("""
    Made with ❤️ using [Streamlit](https://streamlit.io) and [Hugging Face Transformers](https://huggingface.co).  
    Developed by [Jguirim Salem](https://www.linkedin.com).
""")
