import streamlit as st
import google.generativeai as genai
import io
import os
import time
import pandas as pd
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
    "It accepts `.txt`, `.rtf`, and `.csv` files or pasted text. The AI prompt will be applied to each line."
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

    # --- LLM Model Selection ---
    st.header("Choose LLM Model")
    model_name = st.selectbox(
        "Select the model to use:",
        ("models/gemini-2.5-pro", "models/gemini-2.5-flash", "models/gemini-2.5-flash-lite", "models/gemini-1.5-pro",
         "models/gemini-1.5-flash", "models/gemini-pro")
    )
    st.markdown("---")
    st.header("How to Use")
    st.markdown(
        """
        1.  Ensure your API key is in a `.env` file or enter it manually.
        2.  Choose your desired LLM model.
        3.  Choose your input: upload a file or paste text.
            - If uploading a CSV, select the column to process.
        4.  Write your detailed prompt for the AI.
        5.  Select your desired output method.
        6.  Click "Process Text" to get the results.
        """
    )

# --- Main Content Area ---
st.header("1. Provide Your Text")

text_data = None
df = None  # --- ADDED: Initialize df to handle CSV data state

# --- Input Method Tabs ---
input_tab1, input_tab2 = st.tabs(["Upload a File", "Paste Text"])

with input_tab1:
    uploaded_file = st.file_uploader("Choose a .txt, .rtf, or .csv file", type=["txt", "rtf", "csv"])

    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        try:
            if file_extension == ".txt":
                stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                text_data = stringio.read()
            elif file_extension == ".rtf":
                rtf_content = uploaded_file.getvalue().decode('ascii', errors='ignore')
                text_data = rtf_to_text(rtf_content)
            elif file_extension == ".csv":
                df = pd.read_csv(uploaded_file)
                st.info("CSV file uploaded successfully. Please select the column to process.")
                st.dataframe(df.head())
                column_to_process = st.selectbox("Select the column containing the text to process:", df.columns)
                if column_to_process:
                    # Ensure all data in the column is treated as a string, handling potential empty values
                    text_data = "\n".join(df[column_to_process].dropna().astype(str).tolist())

            if file_extension != ".csv":
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

st.header("3. Choose Your Output Method")
output_method = st.radio(
    "How would you like to receive the results?",
    ("Display on screen", "Write to a file")
)

output_filename = ""
if output_method == "Write to a file":
    output_filename = st.text_input("Enter the name for your output file (e.g., results.csv):",
                                    "ai_processed_output.csv")

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
            model = genai.GenerativeModel(model_name)

            lines = [line for line in text_data.strip().split('\n') if line.strip()]
            total_lines = len(lines)

            st.header("✅ Results")
            st.write(f"Processing {total_lines} lines of text using {model_name}...")

            total_start_time = time.monotonic()
            results_for_file = []

            # --- ADDED: Initialize progress bar and status text ---
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, line in enumerate(lines):
                full_prompt = f"PROMPT: '{ai_prompt}'\n\n---\n\nTEXT TO PROCESS: \"{line}\""

                try:
                    start_time = time.monotonic()
                    response = model.generate_content([{"role": "user", "parts": [full_prompt]}])
                    end_time = time.monotonic()
                    processing_time = end_time - start_time

                    if output_method == "Display on screen":
                        with st.expander(f"**Line {i + 1}:** {line[:80]}...", expanded=True):
                            st.markdown("**AI Response:**")
                            st.markdown(response.text)
                            st.info(f"Processing time: {processing_time:.2f} seconds")
                    else:
                        results_for_file.append({"Input Text": line, "AI Response": response.text,
                                                 "Processing Time (s)": f"{processing_time:.2f}"})

                except Exception as e:
                    if output_method == "Display on screen":
                        with st.expander(f"**Line {i + 1}:** {line[:80]}...", expanded=True):
                            st.error(f"Could not process this line. Error: {e}")
                    else:
                        results_for_file.append(
                            {"Input Text": line, "AI Response": f"ERROR: {e}", "Processing Time (s)": "N/A"})

                # --- ADDED: Update progress bar and status text ---
                progress_percentage = (i + 1) / total_lines
                progress_bar.progress(progress_percentage)
                status_text.text(f"Processing line {i + 1}/{total_lines}... ({progress_percentage:.0%})")

            total_end_time = time.monotonic()
            total_processing_time = total_end_time - total_start_time

            # --- MODIFIED: Clear progress elements and show success message ---
            status_text.empty()
            progress_bar.empty()
            st.success(f"All {total_lines} lines have been processed! Total time: {total_processing_time:.2f} seconds")

            if output_method == "Write to a file":
                results_df = pd.DataFrame(results_for_file)
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name=output_filename,
                    mime='text/csv',
                )

        except Exception as e:
            st.error(f"An error occurred while configuring the AI model: {e}")
            st.info("Please check your API key and try again.")