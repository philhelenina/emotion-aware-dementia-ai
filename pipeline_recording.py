import streamlit as st
import numpy as np
import torch
import librosa
import whisper
from transformers import pipeline
import tempfile
import os
import requests
from gtts import gTTS
import io
from datetime import datetime
import glob
from pathlib import Path
import time

# ============================================================================
# SECURE Configuration and API Keys
# ============================================================================

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, skip

# Get OpenAI API Key from environment variable (SECURE - no hardcoded keys)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Try Streamlit secrets if environment variable not found
try:
    if not OPENAI_API_KEY and hasattr(st, 'secrets'):
        OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
except:
    pass

# Audio files folder path
AUDIO_FOLDER = "./sample-wav-files"  # Create this folder and add audio files

# ============================================================================
# Page Configuration and Initialization
# ============================================================================

st.set_page_config(
    page_title="🎤 Terminal Voice AI",
    page_icon="💻",
    layout="wide"
)

# Custom CSS for terminal look
st.markdown("""
<style>
    .stApp {
        background-color: #000000;
        color: #00ff00;
    }
    
    .terminal-container {
        background-color: #000000;
        color: #00ff00;
        font-family: 'Courier New', monospace;
        padding: 20px;
        border-radius: 5px;
        border: 1px solid #00ff00;
        min-height: 400px;
        max-height: 600px;
        overflow-y: auto;
        white-space: pre-wrap;
    }
    
    .user-input {
        color: #00ff00;
        font-weight: bold;
    }
    
    .ai-response {
        color: #00ffff;
        margin-left: 20px;
    }
    
    .emotion-info {
        color: #ffff00;
        font-size: 0.9em;
    }
    
    .timestamp {
        color: #888888;
        font-size: 0.8em;
    }
    
    .terminal-prompt {
        color: #00ff00;
        font-weight: bold;
    }
    
    .status-loading {
        color: #ff9900;
        animation: blink 1s infinite;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    
    #cursor {
        animation: blink 1s infinite;
        color: #00ff00;
        font-weight: bold;
    }
    
    .stButton > button {
        background-color: #003300;
        color: #00ff00;
        border: 1px solid #00ff00;
    }
    
    .stSelectbox > div > div {
        background-color: #003300;
        color: #00ff00;
    }
    
    .stSidebar {
        background-color: #001100;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "terminal_log" not in st.session_state:
    st.session_state.terminal_log = "VOICE AI TERMINAL v1.0\n" + "="*50 + "\n\n"
    st.session_state.terminal_log += "System: Voice AI Terminal initialized...\n"
    st.session_state.terminal_log += "System: Ready for voice input.\n\n"

if "conversation_count" not in st.session_state:
    st.session_state.conversation_count = 0

if "show_typing" not in st.session_state:
    st.session_state.show_typing = False

if "typing_message" not in st.session_state:
    st.session_state.typing_message = ""

if "play_input_audio" not in st.session_state:
    st.session_state.play_input_audio = None

if "replay_ai" not in st.session_state:
    st.session_state.replay_ai = False

if "user_api_key" not in st.session_state:
    st.session_state.user_api_key = ""

# ============================================================================
# Model Loading
# ============================================================================

@st.cache_resource
def load_models():
    """Load AI models for speech recognition and emotion detection"""
    try:
        # Load Whisper model
        st.info("Loading Whisper model...")
        whisper_model = whisper.load_model("base")
        st.success("✅ Whisper model loaded")
        
        # Load emotion classifier
        st.info("Loading emotion classifier...")
        audio_emotion_classifier = None
        try:
            audio_emotion_classifier = pipeline(
                "audio-classification", 
                model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                return_all_scores=True
            )
            st.success("✅ Emotion classifier loaded")
        except Exception as e:
            st.warning(f"⚠️ Emotion classifier failed to load: {e}")
            st.info("Will use neutral emotion as fallback")
        
        return whisper_model, audio_emotion_classifier
        
    except Exception as e:
        st.error(f"❌ Critical error loading models: {e}")
        return None, None

# Load models and store in session state
if "models_loaded" not in st.session_state:
    with st.spinner("Loading AI models..."):
        whisper_model, audio_emotion_classifier = load_models()
        
        if whisper_model is not None:
            st.session_state.whisper_model = whisper_model
            st.session_state.audio_emotion_classifier = audio_emotion_classifier
            st.session_state.models_loaded = True
            st.session_state.terminal_log += "System: AI models loaded successfully.\n\n"
        else:
            st.error("Failed to load models. Please check your internet connection and try again.")
            st.stop()

# Get models from session state
whisper_model = st.session_state.get('whisper_model')
audio_emotion_classifier = st.session_state.get('audio_emotion_classifier')

# ============================================================================
# Utility Functions
# ============================================================================

def get_current_api_key():
    """Get current API key from session state or environment"""
    if st.session_state.user_api_key:
        return st.session_state.user_api_key
    return OPENAI_API_KEY

def get_audio_files():
    """Get supported audio files from the audio folder"""
    if not os.path.exists(AUDIO_FOLDER):
        os.makedirs(AUDIO_FOLDER)
        return []
    
    supported_formats = ["*.wav", "*.mp3", "*.ogg", "*.m4a", "*.flac"]
    audio_files = []
    
    for format in supported_formats:
        audio_files.extend(glob.glob(os.path.join(AUDIO_FOLDER, format)))
    
    return [os.path.basename(f) for f in audio_files]

def transcribe_audio(audio_path):
    """Convert audio to text using Whisper"""
    try:
        whisper_model = st.session_state.get('whisper_model')
        if not whisper_model:
            return "[Error: Whisper model not loaded]"
        
        result = whisper_model.transcribe(audio_path)
        return result["text"].strip()
    except Exception as e:
        return f"[Error: Speech recognition failed - {e}]"

def get_emotion_from_audio(audio_path):
    """Analyze emotion from audio using pretrained model"""
    audio_emotion_classifier = st.session_state.get('audio_emotion_classifier')
    if not audio_emotion_classifier:
        return "neutral", 0.5
    
    try:
        # Load and preprocess audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Adjust audio length (1-10 seconds)
        if len(y) > 10 * sr:
            y = y[:10 * sr]
        elif len(y) < sr:
            y = np.pad(y, (0, sr - len(y)))
        
        # Normalize
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        
        # Predict emotion
        predictions = audio_emotion_classifier({"raw": y, "sampling_rate": sr})
        
        if predictions and len(predictions) > 0:
            top_emotion = max(predictions, key=lambda x: x['score'])
            return top_emotion['label'], top_emotion['score']
        
    except Exception as e:
        st.error(f"Emotion analysis error: {e}")
    
    return "neutral", 0.5

def generate_llm_response(text, emotion, confidence):
    """Generate response using LLM with current API key"""
    
    current_api_key = get_current_api_key()
    
    # Use fallback if no valid API key
    if not current_api_key or not current_api_key.startswith("sk-"):
        return generate_simple_response(emotion)
    
    try:
        headers = {
            "Authorization": f"Bearer {current_api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""User said: "{text}"
Detected emotion: {emotion} (confidence: {confidence:.1f})

Respond naturally and empathetically. Keep it conversational and 1-2 sentences max."""

        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Respond naturally and empathetically."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers, 
            json=data, 
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
            
    except Exception as e:
        pass
    
    return generate_simple_response(emotion)

def generate_simple_response(emotion):
    """Generate simple fallback responses"""
    responses = {
        "angry": "I understand your frustration. Let's work through this.",
        "sad": "I'm sorry you're feeling down. I'm here to help.",
        "happy": "That's great to hear! Your positive energy is contagious.",
        "fear": "I understand your concerns. We can figure this out together.",
        "surprise": "That's quite unexpected! Tell me more about it.",
        "disgust": "I can see why that would be unpleasant.",
        "neutral": "I'm listening. How can I help you today?"
    }
    return responses.get(emotion, "Thank you for sharing. What would you like to talk about?")

def text_to_speech(text):
    """Convert text to speech using Google TTS"""
    try:
        # Clean text for better TTS
        clean_text = text.replace("[", "").replace("]", "").replace("*", "")
        tts = gTTS(text=clean_text, lang="en", slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return mp3_fp.getvalue()
    except Exception as e:
        return None

def add_to_terminal(text, role="system", emotion=None, confidence=None, enable_typing=False):
    """Add text to terminal log with optional typing effect"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if role == "user":
        emotion_str = f" [{emotion}:{confidence:.2f}]" if emotion else ""
        st.session_state.terminal_log += f"[{timestamp}] You: {text}{emotion_str}\n"
    elif role == "assistant":
        if enable_typing:
            # Store the message for typing effect
            st.session_state.typing_message = f"[{timestamp}] AI:  {text}\n"
            st.session_state.show_typing = True
        else:
            st.session_state.terminal_log += f"[{timestamp}] AI:  {text}\n"
    else:
        st.session_state.terminal_log += f"[{timestamp}] {text}\n"
    
    if not enable_typing:
        st.session_state.terminal_log += "\n"

# ============================================================================
# Main UI
# ============================================================================

# Title with terminal style
st.markdown("# 💻 VOICE AI TERMINAL")
st.markdown("**Real-time voice interaction with emotion analysis**")

# Create two columns - main terminal and controls
col1, col2 = st.columns([3, 1])

with col1:
    # Single terminal display area
    terminal_placeholder = st.empty()
    
    # Handle typing effect or normal display
    if st.session_state.show_typing and "last_ai_audio" in st.session_state:
        # Create base64 audio for embedding
        import base64
        audio_b64 = base64.b64encode(st.session_state.last_ai_audio).decode()
        
        # Clean the typing message for JavaScript
        typing_text = st.session_state.typing_message.replace('\n', '').replace('"', '\\"')
        terminal_content = st.session_state.terminal_log.replace('\n', '<br>')
        
        # Use streamlit components for proper JavaScript execution
        import streamlit.components.v1 as components
        
        components.html(f"""
        <div style="background: black; color: #00ff00; font-family: 'Courier New', monospace; padding: 20px; border: 1px solid #00ff00; border-radius: 5px; height: 400px; overflow-y: auto;" id="terminal">
            {terminal_content}<span id="ai-typing"></span><span id="cursor" style="animation: blink 1s infinite;">|</span>
        </div>
        
        <audio id="ai-audio" preload="auto">
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
        </audio>
        
        <style>
            @keyframes blink {{
                0%, 50% {{ opacity: 1; }}
                51%, 100% {{ opacity: 0; }}
            }}
        </style>
        
        <script>
            const message = "{typing_text}";
            const typingElement = document.getElementById('ai-typing');
            const audio = document.getElementById('ai-audio');
            const cursor = document.getElementById('cursor');
            const terminal = document.getElementById('terminal');
            
            let index = 0;
            
            function typeCharacter() {{
                if (index < message.length) {{
                    typingElement.textContent += message[index];
                    index++;
                    terminal.scrollTop = terminal.scrollHeight;
                    setTimeout(typeCharacter, 60); // 60ms per character
                }} else {{
                    cursor.style.display = 'none';
                }}
            }}
            
            // Start audio and typing together
            setTimeout(() => {{
                if (audio) {{
                    audio.play().catch(e => console.log('Audio autoplay blocked'));
                }}
                typeCharacter();
            }}, 500);
        </script>
        """, height=450)
        
        # Reset state
        st.session_state.show_typing = False
        st.session_state.terminal_log += st.session_state.typing_message + "\n"
        
    else:
        # Normal terminal display
        terminal_placeholder.markdown(f'''
        <div class="terminal-container" id="terminal">
        {st.session_state.terminal_log}
        </div>
        <script>
            setTimeout(function() {{
                var terminal = document.getElementById('terminal');
                if (terminal) {{
                    terminal.scrollTop = terminal.scrollHeight;
                }}
            }}, 100);
        </script>
        ''', unsafe_allow_html=True)

with col2:
    st.markdown("### 🎛️ Controls")
    
    # ========================================================================
    # SECURE API Key Configuration Section
    # ========================================================================
    st.markdown("**🔑 API Configuration:**")
    
    # Secure API key input
    user_input_key = st.text_input(
        "OpenAI API Key:",
        value="",
        type="password",
        placeholder="sk-proj-...",
        help="Enter your OpenAI API key for advanced responses"
    )
    
    if user_input_key:
        st.session_state.user_api_key = user_input_key
        st.success("✅ API key saved for session")
    
    # API Status display
    current_api_key = get_current_api_key()
    if current_api_key and current_api_key.startswith("sk-"):
        st.success("🟢 API Ready")
    else:
        st.warning("🔴 No API Key (fallback mode)")
        st.info("💡 Add API key above for better responses")
    
    st.markdown("---")
    
    # Real-time recording section
    st.markdown("**🎤 Real-time Voice Recording:**")
    
    # Use HTML5 MediaRecorder for real-time recording
    import streamlit.components.v1 as components
    
    recording_html = """
    <div style="margin: 10px 0;">
        <button id="recordBtn" style="background: #003300; color: #00ff00; border: 1px solid #00ff00; padding: 10px 20px; border-radius: 5px; cursor: pointer; width: 100%;">
            🔴 Start Recording
        </button>
        <div id="status" style="color: #00ff00; margin-top: 10px; font-size: 14px;"></div>
        <audio id="audioPlayback" controls style="width: 100%; margin-top: 10px; display: none;"></audio>
    </div>
    
    <script>
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;
        
        const recordBtn = document.getElementById('recordBtn');
        const status = document.getElementById('status');
        const audioPlayback = document.getElementById('audioPlayback');
        
        recordBtn.addEventListener('click', async () => {
            if (!isRecording) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    
                    mediaRecorder.addEventListener('dataavailable', event => {
                        audioChunks.push(event.data);
                    });
                    
                    mediaRecorder.addEventListener('stop', () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        const audioUrl = URL.createObjectURL(audioBlob);
                        audioPlayback.src = audioUrl;
                        audioPlayback.style.display = 'block';
                        
                        // Save to download
                        const link = document.createElement('a');
                        link.href = audioUrl;
                        link.download = 'recorded_voice_' + Date.now() + '.wav';
                        link.textContent = 'Download Recording';
                        link.style.color = '#00ff00';
                        
                        status.innerHTML = '✅ Recording saved! You can play it above or <br>' + link.outerHTML;
                    });
                    
                    mediaRecorder.start();
                    isRecording = true;
                    recordBtn.textContent = '⏹️ Stop Recording';
                    recordBtn.style.background = '#660000';
                    status.textContent = '🔴 Recording... Click stop when done.';
                    
                } catch (err) {
                    status.textContent = '❌ Microphone access denied or not available.';
                }
            } else {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                isRecording = false;
                recordBtn.textContent = '🔴 Start Recording';
                recordBtn.style.background = '#003300';
                status.textContent = '⏹️ Recording stopped.';
            }
        });
    </script>
    """
    
    components.html(recording_html, height=200)
    
    st.markdown("---")
    
    # File-based input section
    st.markdown("**📁 Upload Audio File:**")
    
    # File uploader for immediate processing
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'ogg', 'm4a', 'flac'],
        key="audio_uploader"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        import tempfile
        import os
        
        # Create a temporary file with proper extension
        file_extension = uploaded_file.name.split('.')[-1]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}')
        temp_path = temp_file.name
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        temp_file.close()
        
        st.audio(uploaded_file, format=f"audio/{file_extension}")
        
        if st.button("🎤 PROCESS UPLOADED FILE", type="primary", use_container_width=True):
            # Process the uploaded file
            add_to_terminal("Processing uploaded voice...", "system")
            
            with st.spinner("Analyzing uploaded audio..."):
                # Step 1: Transcribe
                transcribed_text = transcribe_audio(temp_path)
                
                # Step 2: Emotion analysis
                emotion, confidence = get_emotion_from_audio(temp_path)
                
                # Add user input to terminal
                add_to_terminal(transcribed_text, "user", emotion, confidence)
                
                # Step 3: Generate response
                ai_response = generate_llm_response(transcribed_text, emotion, confidence)
                
                # Add AI response to terminal immediately
                add_to_terminal(ai_response, "assistant", enable_typing=True)
                
                # Step 4: Generate speech
                speech_audio = text_to_speech(ai_response)
                
                # Store audio in session state for synchronized playback
                if speech_audio:
                    st.session_state.last_ai_audio = speech_audio
                    st.session_state.ai_response_text = ai_response
                    add_to_terminal("🔊 Voice response ready", "system")
                    
                st.session_state.conversation_count += 1
                
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                
                st.success("✅ Uploaded file processed! AI response with voice is ready.")
                st.rerun()
    
    # Folder-based file selection
    st.markdown("**📂 Or Select from Folder:**")
    
    # Get audio file list
    audio_files = get_audio_files()
    
    if not audio_files:
        st.warning("⚠️ No audio files found")
        st.info(f"Add files to folder or use upload above")
        
        # Refresh button to check for new files
        if st.button("🔄 Refresh File List", use_container_width=True):
            st.rerun()
    else:
        # File count and refresh option
        file_col1, file_col2 = st.columns([2, 1])
        
        with file_col1:
            selected_file = st.selectbox(
                "🎵 Select Audio:",
                audio_files,
                key="file_selector"
            )
        
        with file_col2:
            if st.button("🔄", help="Refresh files", use_container_width=True):
                st.rerun()
        
        # Process button
        if st.button("🎤 PROCESS VOICE", type="primary", use_container_width=True):
            audio_path = os.path.join(AUDIO_FOLDER, selected_file)
            
            if os.path.exists(audio_path):
                # Add processing message to terminal
                add_to_terminal("Processing voice input...", "system")
                terminal_placeholder.markdown(f'<div class="terminal-container">{st.session_state.terminal_log}</div>', unsafe_allow_html=True)
                
                # Process audio
                with st.spinner("Analyzing..."):
                    # Step 1: Transcribe
                    transcribed_text = transcribe_audio(audio_path)
                    
                    # Step 2: Emotion analysis
                    emotion, confidence = get_emotion_from_audio(audio_path)
                    
                    # Add user input to terminal
                    add_to_terminal(transcribed_text, "user", emotion, confidence)
                    terminal_placeholder.markdown(f'<div class="terminal-container">{st.session_state.terminal_log}</div>', unsafe_allow_html=True)
                    
                    # Step 3: Generate response
                    ai_response = generate_llm_response(transcribed_text, emotion, confidence)
                    
                    # Add AI response to terminal immediately
                    add_to_terminal(ai_response, "assistant", enable_typing=True)
                    
                    # Step 4: Generate speech
                    speech_audio = text_to_speech(ai_response)
                    
                    # Store audio in session state for synchronized playback
                    if speech_audio:
                        st.session_state.last_ai_audio = speech_audio
                        st.session_state.ai_response_text = ai_response
                        add_to_terminal("🔊 Voice response ready", "system")
                        
                    st.session_state.conversation_count += 1
                    
                    # Show processing message first
                    st.success("🎤 Processing complete! Watch synchronized AI response in terminal!")
                    st.info("💡 You can also replay your input and AI response using the controls on the right →")
                    
                    # Don't rerun immediately - let the typing effect play
    
    # Audio preview and input player
    if audio_files and 'selected_file' in locals():
        audio_path = os.path.join(AUDIO_FOLDER, selected_file)
        if os.path.exists(audio_path):
            st.markdown("**🎵 Your Voice Input:**")
            
            # Create columns for play button and info
            play_col, info_col = st.columns([1, 2])
            
            with play_col:
                if st.button("▶️ Play Input", use_container_width=True):
                    st.session_state.play_input_audio = audio_path  # Store correct audio path
            
            with info_col:
                st.caption(f"📁 {selected_file}")
            
            # Audio player for the INPUT file (not AI response)
            st.audio(audio_path, format="audio/wav")
            
            # Show play notification if button was clicked
            if st.session_state.get('play_input_audio') == audio_path:
                st.info("🎵 Playing your voice input!")
                st.session_state.play_input_audio = None  # Reset after showing
    
    # AI Voice Response Player (Manual Controls)
    if "last_ai_audio" in st.session_state and st.session_state.last_ai_audio:
        st.markdown("---")
        st.markdown("**🔊 AI Voice Response:**")
        
        # Response controls
        resp_col1, resp_col2 = st.columns([1, 2])
        
        with resp_col1:
            if st.button("▶️ Replay AI", use_container_width=True):
                st.session_state.replay_ai = True
        
        with resp_col2:
            st.caption("🤖 AI's voice response")
        
        # Audio player for AI response
        st.audio(st.session_state.last_ai_audio, format="audio/mp3")
        
        if st.session_state.get('replay_ai', False):
            st.info("🎵 Replaying AI response!")
            st.session_state.replay_ai = False
        
        st.caption("💡 AI voice auto-plays with typing effect in terminal")
        
        # Download button for AI response
        st.download_button(
            label="💾 Download Response",
            data=st.session_state.last_ai_audio,
            file_name=f"ai_response_{datetime.now().strftime('%H%M%S')}.mp3",
            mime="audio/mp3",
            use_container_width=True
        )
    
    # Clear terminal
    if st.button("🗑️ CLEAR TERMINAL", use_container_width=True):
        st.session_state.terminal_log = "VOICE AI TERMINAL v1.0\n" + "="*50 + "\n\n"
        st.session_state.terminal_log += "System: Terminal cleared.\n\n"
        st.session_state.conversation_count = 0
        if "last_ai_audio" in st.session_state:
            del st.session_state.last_ai_audio
        st.session_state.play_input_audio = None
        st.session_state.replay_ai = False
    
    # Quick process all files
    if len(audio_files) > 1:
        if st.button("🚀 PROCESS ALL FILES", use_container_width=True):
            for file in audio_files:
                audio_path = os.path.join(AUDIO_FOLDER, file)
                if os.path.exists(audio_path):
                    # Quick processing
                    add_to_terminal(f"Processing {file}...", "system")
                    transcribed_text = transcribe_audio(audio_path)
                    emotion, confidence = get_emotion_from_audio(audio_path)
                    add_to_terminal(transcribed_text, "user", emotion, confidence)
                    ai_response = generate_llm_response(transcribed_text, emotion, confidence)
                    add_to_terminal(ai_response, "assistant")
                    st.session_state.conversation_count += 1
            
            st.success(f"Processed {len(audio_files)} files!")
            st.rerun()
    
    # Stats
    st.markdown("### 📊 Stats")
    st.metric("Conversations", st.session_state.conversation_count)
    
    # API Status
    current_api_key = get_current_api_key()
    api_status = "🟢 Ready" if (current_api_key and current_api_key.startswith("sk-")) else "🔴 No API Key"
    st.metric("API Status", api_status)

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown("""
**🔒 SECURE VERSION - Instructions:**

**📂 Folder Method:**
1. 📁 Add audio files to `./sample-wav-files/` folder
2. 🎵 Select file from dropdown menu
3. 🎤 Click 'PROCESS VOICE' to analyze
4. 💬 View real-time conversation in terminal
5. 🔊 AI response plays automatically with typing synchronization

**📤 Upload Method:**
1. 🎤 Click "Choose an audio file" button
2. 📁 Select audio file from your computer
3. 🎤 Click 'PROCESS UPLOADED FILE'
4. 💬 Watch AI response in terminal

**🔑 API Key Setup:**
- **Environment**: Set `OPENAI_API_KEY` environment variable
- **Sidebar**: Enter API key in the input field above
- **Streamlit Secrets**: Add to `.streamlit/secrets.toml`

**Audio Controls:**
- ▶️ Play Input: Replay your original voice
- ▶️ Replay AI: Replay AI's voice response
- 💾 Download: Save AI response as MP3
- 🔄 Refresh: Check for new files

**Supported formats:** WAV, MP3, OGG, M4A, FLAC
""")

# Live status indicator
st.markdown('<p class="status-loading">● SYSTEM READY - SELECT FILE FROM FOLDER OR UPLOAD</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    print("Starting Secure Voice AI Terminal...")
    print("Run with: streamlit run pipeline_final.py")