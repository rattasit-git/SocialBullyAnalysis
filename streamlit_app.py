import streamlit as st
import google.generativeai as genai
import io
import os
import time
import pandas as pd
from dotenv import load_dotenv
from striprtf.striprtf import rtf_to_text
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    #model_name = st.selectbox(
    #    "Select the model to use:",
    #    ("models/gemini-2.5-pro", "models/gemini-2.5-flash", "models/gemini-2.5-flash-lite", "models/gemini-1.5-pro",
    #     "models/gemini-1.5-flash", "models/gemini-pro")
    #)
    model_name = st.selectbox(
        "Select the model to use:",
        ("models/gemini-2.5-flash", "models/gemini-2.5-flash-lite", "models/gemini-2.5-pro")
    )
    st.markdown("---")
    # --- Concurrency Setting ---
    st.header("Concurrency Settings")
    max_workers = st.slider(
        "Number of concurrent AI requests (threads):",
        min_value=1,
        max_value=20, # You can adjust this based on your API rate limits and machine's capability
        value=10,
        help="Higher values can speed up processing, but may hit API rate limits or consume more resources."
    )
    st.markdown("---")
    st.header("How to Use")
    st.markdown(
        """
        1.  Ensure your API key is in a `.env` file or enter it manually.
        2.  Choose your desired LLM model.
        3.  Adjust the number of concurrent AI requests.
        4.  Choose your input: upload a file or paste text.
            - If uploading a CSV, select the column to process.
        5.  Write your detailed prompt for the AI.
        6.  Select your desired output method.
        7.  Click "Process Text" to get the results.
        """
    )

# --- Main Content Area ---
st.header("1. Provide Your Text")

text_data = None
df = None

# --- Input Method Tabs ---
input_tab1, input_tab2 = st.tabs(["Upload a File", "Paste Text"])

with input_tab1:
    uploaded_file = st.file_uploader("Choose a .txt, .rtf, or .csv file", type=["txt", "rtf", "csv"])

    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        try:
            if file_extension == ".txt":
                # Ensure correct decoding for input files if they contain non-ASCII characters
                stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                text_data = stringio.read()
            elif file_extension == ".rtf":
                # RTF to text conversion often handles various encodings, but good to note
                rtf_content = uploaded_file.getvalue().decode('ascii', errors='ignore') # Assuming RTF generally uses ASCII or similar for control, content might need specific handling
                text_data = rtf_to_text(rtf_content)
            elif file_extension == ".csv":
                # For CSV input, ensure pandas reads with UTF-8 if it might contain non-ASCII
                df = pd.read_csv(uploaded_file, encoding='utf-8') # Added encoding='utf-8' here for reading CSV
                st.info("CSV file uploaded successfully. Please select the column to process.")
                st.dataframe(df.head())
                column_to_process = st.selectbox("Select the column containing the text to process:", df.columns)
                if column_to_process:
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
    ("Display on screen", "Download as a file")
)

output_filename = "ai_processed_output.csv"
if output_method == "Download as a file":
    output_filename = st.text_input(
        "Enter the desired filename (e.g., results.csv):",
        "ai_processed_output.csv",
        help="The file will be downloaded to your browser's default download location, or you will be prompted to choose one."
    )

st.markdown("---")

# --- Function to process a single line with AI ---
def process_single_line(line_index, line_text, ai_prompt_template, model_instance):
    full_prompt = f"PROMPT: '{ai_prompt_template}'\n\n---\n\nTEXT TO PROCESS: \"{line_text}\""
    start_time = time.monotonic()
    try:
        # Use parts array for multimodal and conversational context, even for text-only
        response = model_instance.generate_content([{"role": "user", "parts": [full_prompt]}])
        end_time = time.monotonic()
        processing_time = end_time - start_time
        return {
            "index": line_index,
            "Input Text": line_text,
            "AI Response": response.text,
            "Processing Time (s)": f"{processing_time:.2f}"
        }
    except Exception as e:
        end_time = time.monotonic()
        processing_time = end_time - start_time
        return {
            "index": line_index,
            "Input Text": line_text,
            "AI Response": f"ERROR: {e}",
            "Processing Time (s)": f"{processing_time:.2f}"
        }

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
            st.write(f"Processing {total_lines} lines of text using {model_name} with {max_workers} concurrent requests...")

            total_start_time = time.monotonic()
            results_for_file = [None] * total_lines

            progress_bar = st.progress(0)
            status_text = st.empty()
            processed_count = 0

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_line = {executor.submit(process_single_line, i, line, ai_prompt, model): i for i, line in enumerate(lines)}

                for future in as_completed(future_to_line):
                    result = future.result()
                    results_for_file[result["index"]] = result
                    processed_count += 1

                    progress_percentage = processed_count / total_lines
                    progress_bar.progress(progress_percentage)
                    status_text.text(f"Processed {processed_count}/{total_lines} lines... ({progress_percentage:.0%})")

            total_end_time = time.monotonic()
            total_processing_time = total_end_time - total_start_time

            status_text.empty()
            progress_bar.empty()
            st.success(f"All {total_lines} lines have been processed! Total time: {total_processing_time:.2f} seconds")

            if output_method == "Display on screen":
                combined_texts = []
                for result in results_for_file:
                    if result:
                        header = f"Line {result['index'] + 1}: {result['Input Text'][:80]}"
                        if "ERROR" in result["AI Response"]:
                            combined_texts.append(
                                f"{header}\nERROR: Could not process this line. Details: {result['AI Response']}\nProcessing time: {result['Processing Time (s)']} s\n")
                        else:
                            combined_texts.append(
                                f"{header}\nAI Response: {result['AI Response']}\nProcessing time: {result['Processing Time (s)']} s\n")

                all_results = "\n".join(combined_texts)
                st.text_area("All Results", all_results, height=600, max_chars=None)

            # Handle file output (download)
            if output_method == "Download as a file":
                results_df_data = [{k: v for k, v in res.items() if k != 'index'} for res in results_for_file if res is not None]
                results_df = pd.DataFrame(results_df_data)

                # --- START OF MODIFIED/ADDED CODE FOR ENCODING ---
                # Use StringIO to capture the CSV directly as a string with UTF-8 encoding.
                # This ensures pandas writes the characters correctly into the string buffer.
                csv_buffer = io.StringIO()
                results_df.to_csv(csv_buffer, index=False, encoding='utf-8')
                csv_string_for_download = csv_buffer.getvalue()

                # Prepend the UTF-8 Byte Order Mark (BOM) to the string before encoding to bytes.
                # The BOM helps applications like Microsoft Excel automatically detect UTF-8.
                # It's a specific sequence of bytes: b'\xef\xbb\xbf'
                csv_bytes_with_bom = b'\xef\xbb\xbf' + csv_string_for_download.encode('utf-8')

                st.download_button(
                    label=f"Download '{output_filename}'",
                    data=csv_bytes_with_bom, # Pass the byte string with BOM to the download button
                    file_name=output_filename,
                    mime='text/csv',
                    key='download_csv_button'
                )
                # --- END OF MODIFIED/ADDED CODE FOR ENCODING ---
                st.info("The file will be downloaded via your browser. You can usually choose the save location in the download dialog. If opening in Excel, ensure it's imported as UTF-8.")


        except Exception as e:
            st.error(f"An error occurred while configuring the AI model: {e}")
            st.info("Please check your API key and try again.")