import json
import streamlit as st
import requests
from streamlit_js_eval import streamlit_js_eval
from audiorecorder import audiorecorder
import io
import numpy as np
import whisper
from dotenv import load_dotenv
import os

load_dotenv()

st.markdown("""
<style>
  /* Pin the last VerticalBlockBorderWrapper (your input row) to the bottom */
  [data-testid="stHorizontalBlock"] {
    position: fixed !important;
    bottom: 0 !important;
    left: 300px !important;  /* adjust to your actual sidebar width */
    width: calc(100% - 300px) !important;
    background: white !important;
    padding: 8px 20px !important;
    z-index: 1000 !important;
  }
  /* Give the main chat area extra bottom padding so messages don’t sit under the input row */
  .block-container {
    padding-bottom: 80px !important;
  }
</style>
""", unsafe_allow_html=True)

API_KEY = os.getenv("API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-3.5-turbo"

HEADER = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "X-title": "AI Chatbot Streamlit",
    "HTTP-Referer": "http://localhost:8501",
}

def handle_error():
    st.error(st.session_state.error)
    if st.button("🔁 Retry"):
        st.session_state.error = ''
        st.session_state.retry = True
        st.rerun()

def send_request():
    with st.spinner("Thinking..."):
        payload = {
            "model": st.session_state.model,
            "messages": st.session_state.chat_history,
        }
        try:
            response = requests.post(
                API_URL,
                headers=HEADER,
                json=payload,
                timeout=30,
            )
            if response.status_code == 200:
                bot_response = response.json()["choices"][0]["message"]["content"]
                st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(bot_response)
                store_session_chat(st.session_state.current_session, st.session_state.chat_history)
            else:
                st.session_state.error = f"Error: {response.status_code} - {response.text}"
        except requests.exceptions.RequestException as e:
            st.session_state.error = f"Network error: {e}"


def new_chat():
    st.session_state.chat_history = []
    st.session_state.error = ''
    st.session_state.retry = False
    st.rerun()

def retrieve_session_titles():
    saved_sessions = streamlit_js_eval(js_expressions="localStorage.getItem('session_titles');", key="get_session_titles")
    return json.loads(saved_sessions) if saved_sessions else {}

def store_session_title(session_titles, session_id, title):
    session_titles[session_id] = title
    streamlit_js_eval(
        js_expressions=f"localStorage.setItem('session_titles', JSON.stringify({json.dumps(session_titles)}));", key="save_session_titles"
    )

def retrieve_session_chat(session_id):
    saved_sessions = streamlit_js_eval(js_expressions="localStorage.getItem('session_chat_history');", key="get_session_chat_history")
    session_chat = json.loads(saved_sessions) if saved_sessions else {}
    return session_chat.get(session_id, [])

def store_session_chat(session_id, chat_history):
    streamlit_js_eval(
        js_expressions=f"""
        let history = JSON.parse(localStorage.getItem('session_chat_history') || '{{}}');
        history[{json.dumps(session_id)}] = {json.dumps(chat_history)};
        localStorage.setItem('session_chat_history', JSON.stringify(history));
        """,
        key="save_session_chat_history"
    )

def generate_title(input_text):
    words = input_text.split()
    title = " ".join(words[:5])
    if len(title) > 50:
        title = title[:50]
    return title

@st.cache_resource
def load_model(size="small"):
    return whisper.load_model(size, in_memory=True)
    
@st.dialog("Input audio")
def input_audio():
    audio = audiorecorder(start_prompt="🔴 Start", stop_prompt="⏹️ Stop", key="audio")
    if audio:
        # existing transcription logic
        audio_buffer = io.BytesIO()
        audio.export(audio_buffer, format="wav", parameters=["-ar", "16000"])
        audio_array = np.frombuffer(audio_buffer.getvalue()[44:], dtype=np.int16).astype(np.float32) / 32768.0
        with st.spinner("Transcribing..."):
            model = load_model()
            result = model.transcribe(audio_array, fp16=False)
        if result['text']:
            st.session_state.transcript = result['text']
            st.rerun()
        else:
            st.write('Transcribing failed. Please try again.')

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'error' not in st.session_state:
    st.session_state.error = ''
if 'retry' not in st.session_state:
    st.session_state.retry = False
if 'model' not in st.session_state:
    st.session_state.model = MODEL
if 'session_titles' not in st.session_state:
    st.session_state.session_titles = {}
if 'transcript' not in st.session_state:
    st.session_state.transcript = ''

with st.sidebar:
    if not st.session_state.session_titles:
        st.session_state.session_titles = retrieve_session_titles()
    
    titles = st.session_state.session_titles.copy()

    if not st.session_state.chat_history:
        titles[str(len(titles)+1)] = "New Session"
    titles = sorted(
        titles.items(),
        key=lambda kv: int(kv[0]),
        reverse=True
    )
    titles = dict(titles)

    st.subheader("🗂️ Chat Sessions")
    if st.button("🆕 New Session Chat"):
        new_chat()
    
    selected_session = st.radio(
        "Select a session",
        options=titles,
        format_func=lambda k: titles[k],
    )

    if selected_session:
        st.session_state.current_session = selected_session
        saved_sessions = retrieve_session_chat(selected_session) 
        if saved_sessions and isinstance(saved_sessions, list):
            st.session_state.chat_history = saved_sessions

    st.title("🤖 Chatbot Settings")
    model_choice = st.sidebar.selectbox("Choose Model", [
        "mistralai/mistral-7b-instruct"
    ])
    st.session_state.model = model_choice
    st.markdown("Powered by [OpenRouter AI](https://openrouter.ai)")
        
    st.markdown("---")
    
st.title("💬 AI Chatbot")
st.caption("Ask anything! The bot will try its best to help you.")

if not st.session_state.chat_history:
    st.markdown(
        "<div style='text-align:center; color:gray;'>"
        "👋 <b>Welcome! Start the conversation below.</b>"
        "</div>",
        unsafe_allow_html=True,
    )

for chat in st.session_state.chat_history:
    avatar = "🧑" if chat["role"] == "user" else "🤖"
    with st.chat_message(chat["role"], avatar=avatar):
        st.markdown(chat["content"])

input_cols = st.columns([9, 1, 1])

with input_cols[1]:
    if st.button("🎙️"):
        input_audio()

with input_cols[0]:
    user_input = st.chat_input(
        key="user_input"
    )

if user_input or st.session_state.transcript or st.session_state.retry:
    if st.session_state.retry:
        st.session_state.retry = False
    else:
        if st.session_state.transcript:
            user_input = st.session_state.transcript
            st.session_state.transcript = ''
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)
        if len(st.session_state.chat_history) == 1:
            title = generate_title(user_input)
            st.session_state.current_session = str(len(st.session_state.session_titles) + 1)
            st.session_state.session_titles[st.session_state.current_session] = title
            store_session_title(st.session_state.session_titles, st.session_state.current_session, title)

    send_request()
    
if st.session_state.error:
    handle_error()