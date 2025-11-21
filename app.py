import streamlit as st
import os
import string
from openai import OpenAI
import time

# -------------------------------
#   Prompt Templates
#   These templates shape how the LLM responds.
# -------------------------------

SYSTEM_PROMPT = """You are an AI assistant that provides concise, factual answers with optional citations."""

USER_PROMPT_BASIC = "{processed_question}"

USER_PROMPT_ENHANCED = """
Please answer the following question concisely:

Question: {processed_question}

Provide a clear and factual explanation. Cite credible sources when relevant.
"""

# -------------------------------
#   Streamlit Page Configuration
# -------------------------------

st.set_page_config(
    page_title="LLM Q&A System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
#   Custom CSS (Fully Rewritten)
#   Layout redesigned: neutral tones, framed sections, clean typography
# -------------------------------

st.markdown("""
<style>

/* Base font import */
@import url('https://fonts.googleapis.com/css2?family=Rubik:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Rubik', sans-serif;
}

/* Container background */
.main {
    background-color: #f5f6f7;
}

/* Header panel styling */
.custom-header {
    background: #1c2530;
    padding: 2.4rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    color: #ffffff;
    text-align: center;
}

.custom-header h1 {
    margin: 0;
    font-weight: 600;
    font-size: 2.3rem;
}

.custom-header p {
    margin-top: 0.4rem;
    font-size: 1.05rem;
    opacity: 0.85;
}

/* Result sections */
.result-box {
    background: #1c2530;
    border: 1px solid #d6d9dd;
    padding: 1.3rem 1.4rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}

.result-title {
    font-weight: 600;
    color: #2f3c4a;
    font-size: 0.95rem;
    margin-bottom: 0.4rem;
}

/* Button styling */
.stButton>button {
    background-color: #1c2530;
    color: #ffffff;
    padding: 0.55rem 2rem;
    border: none;
    border-radius: 6px;
    font-weight: 500;
    transition: background-color 0.2s ease;
}

.stButton>button:hover {
    background-color: #2b3948;
}

/* Sidebar history item */
.history-entry {
    background-color: #eef0f2;
    border-left: 4px solid #1c2530;
    padding: 0.9rem;
    border-radius: 6px;
    margin-bottom: 0.7rem;
}

/* Remove default expander border */
.streamlit-expanderHeader {
    font-weight: 500;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
#   Question Preprocessing
# -------------------------------

def preprocess_question(question: str) -> str:
    """
    Prepare the user's question for the prompt by:
    - Lowercasing text
    - Removing punctuation
    - Tokenizing and recombining
    """
    lowered = question.lower()
    no_punct = lowered.translate(str.maketrans('', '', string.punctuation))
    tokens = no_punct.split()
    return " ".join(tokens)

# -------------------------------
#   Prompt Builder
# -------------------------------

def build_prompt(processed_question: str, use_enhanced: bool = False) -> list:
    """
    Construct the list of messages used by the LLM API call.
    """
    user_content = USER_PROMPT_ENHANCED.format(processed_question=processed_question) if use_enhanced else USER_PROMPT_BASIC.format(processed_question=processed_question)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

# -------------------------------
#   LLM API Interaction
# -------------------------------

def call_llm_api(messages: list, use_mock: bool = False) -> str:
    """
    Communicate with the OpenAI model or fallback mock mode.
    """
    if use_mock:
        return mock_llm_response(messages[-1]["content"])

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Error: API key not found. Add OPENAI_API_KEY to your environment."

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=200,
            temperature=0.7,
            timeout=30
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_message = str(e).lower()
        if "timeout" in error_message:
            return "Error: Request timed out."
        if "rate_limit" in error_message:
            return "Error: Rate limit exceeded."
        if "authentication" in error_message:
            return "Error: Invalid API key."
        return f"Error: {str(e)}"

# -------------------------------
#   Mock Response Generator
# -------------------------------

def mock_llm_response(question: str) -> str:
    """
    Provide predictable test responses without API calls.
    """
    q = question.lower()
    if "capital" in q or "city" in q:
        return "Mock Response: Example – the capital of France is Paris."
    if "python" in q:
        return "Mock Response: Python is a general-purpose programming language."
    if "weather" in q:
        return "Mock Response: Weather information is unavailable offline."
    return "Mock Response: Simulated output."

# -------------------------------
#   Session History Initialization
# -------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "use_mock" not in st.session_state:
    st.session_state.use_mock = os.getenv("USE_MOCK", "false").lower() == "true"

# -------------------------------
#   Header Section
# -------------------------------

st.markdown("""
<div class="custom-header">
    <h1>LLM Q&A System</h1>
    <p>Submit a question and receive an instant AI-generated explanation.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
#   Main Layout
# -------------------------------

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Enter Your Question")

    user_question = st.text_area(
        "Question Input",
        height=110,
        placeholder="Example: What is machine learning?",
        label_visibility="collapsed"
    )

    if st.button("Get Answer", use_container_width=True):

        if user_question.strip():

            with st.spinner("Processing your request..."):
                processed = preprocess_question(user_question)
                messages = build_prompt(processed)
                answer = call_llm_api(messages, use_mock=st.session_state.use_mock)

                st.session_state.history.insert(0, {
                    "question": user_question,
                    "processed": processed,
                    "answer": answer,
                    "timestamp": time.strftime("%H:%M:%S")
                })

                if len(st.session_state.history) > 5:
                    st.session_state.history.pop()

            st.markdown("---")

            st.markdown(f"""
            <div class="result-box">
                <div class="result-title">Processed Question</div>
                {processed}
            </div>

            <div class="result-box">
                <div class="result-title">Answer</div>
                {answer}
            </div>
            """, unsafe_allow_html=True)

            st.button("Copy Answer", on_click=lambda: st.write("Answer copied."))
        else:
            st.warning("Please enter a question before submitting.")

# -------------------------------
#   Sidebar — History
# -------------------------------

with st.sidebar:
    st.header("Recent Questions")

    mock_toggle = st.checkbox("Enable Mock Mode", value=st.session_state.use_mock)
    st.session_state.use_mock = mock_toggle

    st.markdown("---")

    if st.session_state.history:
        for idx, item in enumerate(st.session_state.history):
            with st.expander(f"Q{idx+1}: {item['question'][:45]} ({item['timestamp']})"):
                st.markdown(f"**Processed:** {item['processed']}")
                st.markdown(f"**Answer:** {item['answer']}")
    else:
        st.info("No previous questions available.")

    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

# -------------------------------
#   Footer
# -------------------------------

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7a8087; font-size: 0.9rem;">
    CSC415/CSC331 AI Project 2 | Developed with Streamlit and OpenAI API
</div>
""", unsafe_allow_html=True)

