import streamlit as st
import google.generativeai as genai
import io
import os
import time  # --- ADDED: Import the time module ---
from dotenv import load_dotenv
from striprtf.striprtf import rtf_to_text

# --- Load Environment Variables ---
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="Gemini AI Text Processor",
    page_icon="✨",
    layout="wide",
)

# --- Application Title and Description ---
st.title("Gemini AI Text Processor")
st.markdown(
    "This application allows you to process text line-by-line using Google's Gemini AI. "
    "It accepts `.txt` and `.rtf` files or pasted text. The AI prompt will be applied to each line."
)

# --- Sidebar for API Key and Instructions ---
with st.sidebar:
    st.header("Configuration")
    api_key_from_env = os.getenv("GOOGLE_API_KEY")
    google_api_key = st.text_input(
        "Enter your Google API Key",
        type="password",
        value=api_key_from_env,
        help="Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)"
    )

    st.markdown("---")
    st.header("How to Use")
    st.markdown(
        """
        1.  Ensure your API key is in a `.env` file or enter it manually.
        2.  Choose your input: upload a `.txt`/.`rtf` file or paste text.
        3.  Write your detailed prompt for the AI.
        4.  Click "Process Text" to get the results.
        """
    )

# --- Main Content Area ---
st.header("1. Provide Your Text")

text_data = None

# --- Input Method Tabs ---
input_tab1, input_tab2 = st.tabs(["Upload a File", "Paste Text"])

with input_tab1:
    uploaded_file = st.file_uploader("Choose a .txt or .rtf file", type=["txt", "rtf"])

    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        try:
            if file_extension == ".txt":
                stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                text_data = stringio.read()
            elif file_extension == ".rtf":
                rtf_content = uploaded_file.getvalue().decode('ascii', errors='ignore')
                text_data = rtf_to_text(rtf_content)

            st.info("File uploaded successfully. Preview of the extracted text:")
            st.text_area("File Content Preview", text_data[:500], height=150, disabled=True)

        except Exception as e:
            st.error(f"Error reading file: {e}")

with input_tab2:
    pasted_text = st.text_area("Paste your text here (one sentence per line)", height=250)
    if pasted_text:
        text_data = pasted_text

st.header("2. Write Your AI Prompt")
ai_prompt = st.text_area(
    "What should the AI do with each line of text?",
    height=200,
    placeholder="e.g., 'Analyze the sentiment of the following text. Classify it as positive, negative, or neutral. Then, explain your reasoning in one sentence.'"
)

st.markdown("---")

# --- Processing and Output ---
if st.button("Process Text", type="primary"):
    if not google_api_key:
        st.error("Please enter your Google API Key in the sidebar or add it to a .env file.")
    elif not text_data or not text_data.strip():
        st.error("Please upload a file or paste some text to process.")
    elif not ai_prompt:
        st.error("Please enter a prompt for the AI.")
    else:
        try:
            genai.configure(api_key=google_api_key)
            model = genai.GenerativeModel('models/gemini-1.5-pro')

            lines = [line for line in text_data.strip().split('\n') if line.strip()]

            st.header("✅ Results")
            st.write(f"Processing {len(lines)} lines of text...")

            # --- ADDED: Start a timer for the total processing time ---
            total_start_time = time.monotonic()

            with st.spinner('The AI is processing your text... Please wait.'):
                for i, line in enumerate(lines):
                    full_prompt = f"PROMPT: '{ai_prompt}'\n\n---\n\nTEXT TO PROCESS: \"{line}\""

                    try:
                        # --- MODIFIED: Record start and end times for each API call ---
                        start_time = time.monotonic()
                        response = model.generate_content([{"role": "user", "parts": [full_prompt]}])
                        end_time = time.monotonic()
                        processing_time = end_time - start_time

                        with st.expander(f"**Line {i + 1}:** {line[:80]}...", expanded=True):
                            st.markdown("**AI Response:**")
                            st.markdown(response.text)
                            # --- ADDED: Display the processing time for the individual line ---
                            st.info(f"Processing time: {processing_time:.2f} seconds")

                    except Exception as e:
                        with st.expander(f"**Line {i + 1}:** {line[:80]}...", expanded=True):
                            st.error(f"Could not process this line. Error: {e}")

            # --- ADDED: Calculate total processing time ---
            total_end_time = time.monotonic()
            total_processing_time = total_end_time - total_start_time

            # --- MODIFIED: Display total time in the success message ---
            st.success(f"All {len(lines)} lines have been processed! Total time: {total_processing_time:.2f} seconds")

        except Exception as e:
            st.error(f"An error occurred while configuring the AI model: {e}")
            st.info("Please check your API key and try again.")