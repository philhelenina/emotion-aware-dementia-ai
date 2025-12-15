def add_to_terminal(text, role="system", emotion=None, confidence=None):
    """Add text to terminal log with robot-compatible formatting"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if role == "user":
        # Add user text first
        st.session_state.terminal_log += f"[{timestamp}] You: {text}\n"
        # Add emotion info on next line if available
        if emotion:
            st.session_state.terminal_log += f"           Detected emotion: {emotion}\n"
    elif role == "assistant":
        st.session_state
        
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
import re

# ============================================================================
# Configuration and API Keys
# ============================================================================

# OpenAI API Key - load from environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Audio files folder path
AUDIO_FOLDER = "./sample-wav-files"  # Create this folder and add audio files

# ============================================================================
# Page Configuration and Initialization
# ============================================================================

st.set_page_config(
    page_title="Empathetic AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for empathetic theme - ROBOT COMPATIBLE
st.markdown("""
<style>
    .stApp {
        background-color: #000000;
        color: #90EE90;
    }
    
    .terminal-container {
        background-color: #000000;
        color: #90EE90;
        font-family: 'Courier New', monospace;
        padding: 20px;
        border-radius: 5px;
        border: 1px solid #90EE90;
        min-height: 500px;
        max-height: 600px;
        overflow-y: auto;
        white-space: pre-wrap;
    }
    
    .user-input {
        color: #90EE90;
        font-weight: bold;
    }
    
    .ai-response {
        color: #98FB98;
        margin-left: 20px;
    }
    
    .emotion-info {
        color: #FFFFE0;
        font-size: 0.9em;
    }
    
    .timestamp {
        color: #888888;
        font-size: 0.8em;
    }
    
    .terminal-prompt {
        color: #90EE90;
        font-weight: bold;
    }
    
    .status-loading {
        color: #98FB98;
        animation: blink 1s infinite;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    
    .stButton > button {
        background-color: #006400;
        color: #90EE90;
        border: 1px solid #90EE90;
    }
    
    .stSelectbox > div > div {
        background-color: #004d00;
        color: #90EE90;
    }
    
    .stSidebar {
        background-color: #002200;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state - NO EMOJIS
if "terminal_log" not in st.session_state:
    st.session_state.terminal_log = "EMOTION-AWARE EMPATHETIC ASSISTANT v1.0\n" + "="*50 + "\n\n"
    st.session_state.terminal_log += "System: Empathy Engine initialized\n"
    st.session_state.terminal_log += "System: Ready for compassionate conversation\n\n"

if "conversation_count" not in st.session_state:
    st.session_state.conversation_count = 0

if "current_input_audio" not in st.session_state:
    st.session_state.current_input_audio = None

if "current_input_filename" not in st.session_state:
    st.session_state.current_input_filename = None

if "current_input_text" not in st.session_state:
    st.session_state.current_input_text = ""

if "last_ai_audio" not in st.session_state:
    st.session_state.last_ai_audio = None

if "ai_response_text" not in st.session_state:
    st.session_state.ai_response_text = ""

# ============================================================================
# Model Loading
# ============================================================================

@st.cache_resource
def load_models():
    """Load AI models"""
    try:
        # Load Whisper model
        st.info("Loading Whisper model")
        whisper_model = whisper.load_model("base")
        st.success("Whisper model loaded")
        
        # Load emotion classifier
        st.info("Loading emotion classifier")
        audio_emotion_classifier = None
        try:
            audio_emotion_classifier = pipeline(
                "audio-classification", 
                model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                return_all_scores=True
            )
            st.success("Emotion classifier loaded")
        except Exception as e:
            st.warning(f"Emotion classifier failed to load: {e}")
            st.info("Will use neutral emotion as fallback")
        
        return whisper_model, audio_emotion_classifier
        
    except Exception as e:
        st.error(f"Critical error loading models: {e}")
        return None, None

# Load models and store in session state
if "models_loaded" not in st.session_state:
    with st.spinner("Loading AI models..."):
        whisper_model, audio_emotion_classifier = load_models()
        
        if whisper_model is not None:
            st.session_state.whisper_model = whisper_model
            st.session_state.audio_emotion_classifier = audio_emotion_classifier
            st.session_state.models_loaded = True
            st.session_state.terminal_log += "System: Empathy models loaded successfully\n\n"
        else:
            st.error("Failed to load models. Please check your internet connection and try again.")
            st.stop()

# Get models from session state
whisper_model = st.session_state.get('whisper_model')
audio_emotion_classifier = st.session_state.get('audio_emotion_classifier')

def create_simple_terminal():
    """Create simple terminal display"""
    terminal_content = st.session_state.terminal_log
    
    st.markdown(f"""
    <div style="background: #000000; color: #90EE90; font-family: 'Courier New', monospace; padding: 20px; border: 1px solid #90EE90; border-radius: 5px; height: 400px; overflow-y: auto; white-space: pre-wrap; line-height: 1.4;">
{terminal_content}
    </div>
    """, unsafe_allow_html=True)

def create_main_terminal_with_perfect_sync():
    """Create main terminal with perfect audio-text synchronization"""
    import base64
    import streamlit.components.v1 as components
    
    terminal_content = st.session_state.terminal_log
    
    # Prepare audio data with error handling
    input_audio_b64 = ""
    output_audio_b64 = ""
    
    if st.session_state.get('current_input_audio'):
        try:
            input_audio_b64 = base64.b64encode(st.session_state.current_input_audio).decode()
        except Exception as e:
            pass
    
    if st.session_state.get('last_ai_audio'): 
        try:
            output_audio_b64 = base64.b64encode(st.session_state.last_ai_audio).decode()
        except Exception as e:
            pass
    
    # Get texts for synchronization - clean for robot use with None safety
    input_text = (st.session_state.get('current_input_text') or '').replace('"', '').replace("'", '').replace('\n', ' ')
    output_text = (st.session_state.get('ai_response_text') or '').replace('"', '').replace("'", '').replace('\n', ' ')
    
    html_code = f"""
    <div style="background: #000000; color: #90EE90; font-family: 'Courier New', monospace; padding: 20px; border: 1px solid #90EE90; border-radius: 5px; height: 500px; overflow-y: auto; position: relative;">
        
        <!-- Audio Control Panel -->
        <div style="position: sticky; top: 0; background: #000000; padding: 10px 0; border-bottom: 1px solid #90EE90; margin-bottom: 15px; z-index: 100;">
            <div style="display: flex; gap: 15px; align-items: center; flex-wrap: wrap;">
                <button onclick="playInputWithSync()" style="background: #006400; border: 1px solid #90EE90; color: #90EE90; padding: 8px 16px; cursor: pointer; font-family: monospace; font-size: 12px;">PLAY YOUR VOICE</button>
                <button onclick="playOutputWithSync()" style="background: #006400; border: 1px solid #90EE90; color: #90EE90; padding: 8px 16px; cursor: pointer; font-family: monospace; font-size: 12px;">PLAY AI RESPONSE</button>
                <button onclick="stopAllAudio()" style="background: #640000; border: 1px solid #FF6B6B; color: #FF6B6B; padding: 8px 16px; cursor: pointer; font-family: monospace; font-size: 12px;">STOP</button>
                <span id="syncStatus" style="color: #FFFFE0; font-size: 11px;">Ready for sync</span>
            </div>
        </div>
        
        <!-- Main Terminal Content -->
        <div id="terminalContent" style="white-space: pre-wrap; line-height: 1.5; font-size: 13px;">
{terminal_content}        </div>
        
        <!-- Synchronized Typing Area -->
        <div id="syncTypingArea" style="color: #98FB98; background: #001100; padding: 10px; margin-top: 10px; border: 1px solid #90EE90; border-radius: 3px; white-space: pre-wrap; line-height: 1.4; display: none;">
        </div>
        
        <!-- Hidden Audio Elements -->
        <audio id="inputAudio" preload="auto" style="display: none;">
            <source src="data:audio/wav;base64,{input_audio_b64}" type="audio/wav">
        </audio>
        
        <audio id="outputAudio" preload="auto" style="display: none;">
            <source src="data:audio/mp3;base64,{output_audio_b64}" type="audio/mp3">
        </audio>
    </div>
    
    <script>
        const inputText = "{input_text}";
        const outputText = "{output_text}";
        let syncTypingInterval;
        let currentAudio = null;
        
        function updateStatus(message) {{
            document.getElementById('syncStatus').textContent = message;
        }}
        
        function showSyncTyping(text, label) {{
            const typingArea = document.getElementById('syncTypingArea');
            if (!text) return;
            
            typingArea.style.display = 'block';
            typingArea.textContent = label;
            
            clearInterval(syncTypingInterval);
            
            const typingSpeed = Math.max(30, Math.min(100, 6000 / text.length));
            let index = 0;
            
            syncTypingInterval = setInterval(() => {{
                if (index < text.length) {{
                    typingArea.textContent = label + text.substring(0, index + 1);
                    index++;
                    
                    // Auto scroll to keep typing visible
                    const container = typingArea.parentElement;
                    container.scrollTop = container.scrollHeight;
                }} else {{
                    clearInterval(syncTypingInterval);
                }}
            }}, typingSpeed);
        }}
        
        function hideSyncTyping() {{
            clearInterval(syncTypingInterval);
            const typingArea = document.getElementById('syncTypingArea');
            typingArea.style.display = 'none';
        }}
        
        function playInputWithSync() {{
            const audio = document.getElementById('inputAudio');
            
            if (!audio.src || audio.src.length < 100) {{
                updateStatus('No input audio available');
                return;
            }}
            
            stopAllAudio();
            currentAudio = audio;
            
            audio.play().then(() => {{
                updateStatus('Playing your voice with sync');
                if (inputText) {{
                    showSyncTyping(inputText, 'SYNC YOUR VOICE: ');
                }}
            }}).catch(e => {{
                updateStatus('Audio play failed: ' + e.message);
                console.error('Input audio error:', e);
            }});
        }}
        
        function playOutputWithSync() {{
            const audio = document.getElementById('outputAudio');
            
            if (!audio.src || audio.src.length < 100) {{
                updateStatus('No AI response audio available');
                return;
            }}
            
            stopAllAudio();
            currentAudio = audio;
            
            audio.play().then(() => {{
                updateStatus('Playing AI response with sync');
                if (outputText) {{
                    showSyncTyping(outputText, 'SYNC AI RESPONSE: ');
                }}
            }}).catch(e => {{
                updateStatus('Audio play failed: ' + e.message);
                console.error('Output audio error:', e);
            }});
        }}
        
        function stopAllAudio() {{
            const inputAudio = document.getElementById('inputAudio');
            const outputAudio = document.getElementById('outputAudio');
            
            try {{
                inputAudio.pause();
                inputAudio.currentTime = 0;
            }} catch(e) {{}}
            
            try {{
                outputAudio.pause();
                outputAudio.currentTime = 0;
            }} catch(e) {{}}
            
            hideSyncTyping();
            currentAudio = null;
            updateStatus('Stopped - Ready for sync');
        }}
        
        // Audio event listeners for better sync control
        document.getElementById('inputAudio').addEventListener('ended', function() {{
            updateStatus('Your voice playback finished');
            setTimeout(() => {{
                hideSyncTyping();
                updateStatus('Ready for sync');
            }}, 1000);
        }});
        
        document.getElementById('outputAudio').addEventListener('ended', function() {{
            updateStatus('AI response playback finished');
            setTimeout(() => {{
                hideSyncTyping();
                updateStatus('Ready for sync');
            }}, 1000);
        }});
        
        // Error handling
        document.getElementById('inputAudio').addEventListener('error', function(e) {{
            console.error('Input audio error:', e);
            updateStatus('Input audio error');
            hideSyncTyping();
        }});
        
        document.getElementById('outputAudio').addEventListener('error', function(e) {{
            console.error('Output audio error:', e);
            updateStatus('Output audio error');
            hideSyncTyping();
        }});
        
        // Initialize - scroll to bottom
        setTimeout(() => {{
            const container = document.getElementById('terminalContent').parentElement;
            container.scrollTop = container.scrollHeight;
            updateStatus('Terminal loaded - Ready for sync');
        }}, 100);
        
        // Auto-scroll terminal content to bottom when new content is added
        const terminalContent = document.getElementById('terminalContent');
        if (terminalContent) {{
            const container = terminalContent.parentElement;
            container.scrollTop = container.scrollHeight;
        }}
    </script>
    """
    
    components.html(html_code, height=600)

# ============================================================================
# Text Cleaning Functions - ROBOT COMPATIBLE
# ============================================================================

def clean_text_for_robot(text):
    """Clean text completely for robot use - no punctuation or emojis"""
    if not text:
        return ""
    
    # Remove all emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    
    # Remove ALL punctuation for robot compatibility
    punctuations = '.,!?;:"()[]{}\'`~@#$%^&*+=<>/\\|_-'
    for punct in punctuations:
        text = text.replace(punct, ' ')
    
    # Clean up multiple spaces and trim
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# ============================================================================
# Utility Functions
# ============================================================================

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
    """Convert audio to text and clean for robot use"""
    try:
        whisper_model = st.session_state.get('whisper_model')
        if not whisper_model:
            return "Error: Whisper model not loaded"
        
        result = whisper_model.transcribe(audio_path)
        raw_text = result["text"].strip()
        
        # Clean text for robot compatibility
        cleaned_text = clean_text_for_robot(raw_text)
        
        return cleaned_text
    except Exception as e:
        return f"Error: Speech recognition failed - {e}"

def get_emotion_from_audio(audio_path):
    """Analyze emotion from audio"""
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
    """Generate response using LLM and clean for robot use"""
    
    if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-your-openai-api-key-here":
        return clean_text_for_robot(generate_simple_response(emotion))
    
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""User said: "{text}"
Detected emotion: {emotion} (confidence: {confidence:.1f})

As a compassionate therapist, respond naturally and empathetically. Keep it conversational and 1-2 sentences max. Use only plain words with spaces. No punctuation marks, commas, periods, exclamation marks, question marks, or emojis allowed. Robot compatible response only."""

        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a compassionate therapist and helpful AI assistant. Respond naturally and empathetically using only plain words with spaces. No punctuation marks or emojis allowed. Robot compatible only. Focus on providing therapeutic support and understanding."
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
            ai_text = result["choices"][0]["message"]["content"].strip()
            
            # Clean for robot use
            clean_ai_text = clean_text_for_robot(ai_text)
            return clean_ai_text
            
    except Exception as e:
        pass
    
    return clean_text_for_robot(generate_simple_response(emotion))

def generate_simple_response(emotion):
    """Generate simple fallback responses for robot use - therapist style"""
    responses = {
        "angry": "I can sense your frustration and I want you to know that your feelings are completely valid",
        "sad": "I hear the sadness in your voice and I want you to know that I am here to support you through this",
        "happy": "I can feel your positive energy and it brings me joy to share in this moment with you",
        "fear": "I understand you are feeling anxious and I want you to know that we can work through this together",
        "surprise": "That sounds quite unexpected and I would love to hear more about what you are experiencing",
        "disgust": "I can sense your discomfort and I want you to know that your feelings about this situation are important",
        "neutral": "I am here to listen and provide support in whatever way you need right now"
    }
    return responses.get(emotion, "Thank you for sharing your feelings with me and I want you to know that I am here to help")

def text_to_speech_for_robot(text):
    """Convert clean text to speech for robot use with debugging"""
    try:
        # Extra cleaning for TTS
        super_clean_text = clean_text_for_robot(text)
        super_clean_text = super_clean_text.replace("  ", " ").strip()
        
        if not super_clean_text:
            return None
        
        # Force simple text for testing if too short
        if len(super_clean_text) < 3:
            super_clean_text = "Hello this is a test message"
        
        # Show what we're trying to convert
        st.info(f"🔊 Generating speech for: '{super_clean_text}'")
            
        tts = gTTS(text=super_clean_text, lang="en", slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        audio_data = mp3_fp.getvalue()
        
        # Debug: Check if audio was generated
        if len(audio_data) > 1000:  # Valid MP3 should be at least 1KB
            st.success(f"✅ Speech generated: {len(audio_data)} bytes")
            return audio_data
        else:
            st.error("❌ Generated audio too small")
            return None
        
    except Exception as e:
        st.error(f"❌ TTS failed: {e}")
        # Fallback: Create a simple test audio
        try:
            st.info("🔄 Trying fallback audio...")
            test_tts = gTTS(text="Audio test", lang="en", slow=False)
            test_fp = io.BytesIO()
            test_tts.write_to_fp(test_fp)
            test_fp.seek(0)
            fallback_data = test_fp.getvalue()
            st.warning(f"⚠️ Using fallback audio: {len(fallback_data)} bytes")
            return fallback_data
        except Exception as e2:
            st.error(f"❌ Fallback also failed: {e2}")
            return None

def add_to_terminal(text, role="system", emotion=None, confidence=None):
    """Add text to terminal log with robot-compatible formatting"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if role == "user":
        # Add user text first
        st.session_state.terminal_log += f"[{timestamp}] You: {text}\n"
        # Add emotion info on next line if available - UPPERCASE
        if emotion:
            st.session_state.terminal_log += f"           Detected emotion: {emotion.upper()}\n"
    elif role == "assistant":
        st.session_state.terminal_log += f"[{timestamp}] AI: {text}\n"
    else:
        st.session_state.terminal_log += f"[{timestamp}] {text}\n"
    
    st.session_state.terminal_log += "\n"

# ============================================================================
# Main UI
# ============================================================================

# Title - ROBOT COMPATIBLE
st.markdown("# Emotion-Aware Empathetic Assistant")
st.markdown("**Understanding your emotions - Responding with empathy - Perfect terminal audio sync**")

# Create main layout
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### Main Terminal")
    
    # Use simple terminal without complex sync
    create_simple_terminal()
    
    # Add basic audio players below terminal
    st.markdown("---")
    st.markdown("### Audio Players")
    
    audio_col1, audio_col2 = st.columns(2)
    
    with audio_col1:
        st.markdown("**🎤 Your Voice:**")
        if st.session_state.get('current_input_audio'):
            st.success(f"Loaded: {st.session_state.get('current_input_filename', 'Unknown')}")
            if st.session_state.get('current_input_text'):
                st.caption(f"Text: {st.session_state.current_input_text}")
            
            # Simple streamlit audio player
            st.audio(st.session_state.current_input_audio, format="audio/wav")
        else:
            st.info("No input audio loaded")
    
    with audio_col2:
        st.markdown("**🤖 AI Response:**")
        if st.session_state.get('last_ai_audio'):
            st.success("AI response generated")
            if st.session_state.get('ai_response_text'):
                st.caption(f"Text: {st.session_state.ai_response_text}")
            
            # Simple streamlit audio player
            st.audio(st.session_state.last_ai_audio, format="audio/mp3")
        else:
            st.info("No AI response audio generated")

with col2:
    st.markdown("### Empathy Controls")
    
    # Get audio file list
    audio_files = get_audio_files()
    
    if not audio_files:
        st.warning("No audio files found")
        st.info(f"Add files to {AUDIO_FOLDER} folder")
    else:
        selected_file = st.selectbox(
            "Share Your Voice:",
            audio_files,
            key="file_selector"
        )
        
        # Load selected audio file for main terminal
        if selected_file:
            audio_path = os.path.join(AUDIO_FOLDER, selected_file)
            if os.path.exists(audio_path):
                try:
                    with open(audio_path, 'rb') as f:
                        input_audio_bytes = f.read()
                    
                    st.session_state.current_input_audio = input_audio_bytes
                    st.session_state.current_input_filename = selected_file
                    st.success(f"{selected_file} loaded for sync")
                    
                except Exception as e:
                    st.error(f"Error loading file: {e}")
        
        # Main process button
        if st.button("START EMPATHETIC CONVERSATION", type="primary", use_container_width=True):
            audio_path = os.path.join(AUDIO_FOLDER, selected_file)
            
            if os.path.exists(audio_path):
                add_to_terminal("Processing your voice with empathy", "system")
                
                with st.spinner("Understanding your emotions"):
                    # Step 1: Transcribe and clean for robot
                    clean_transcribed = transcribe_audio(audio_path)
                    st.session_state.current_input_text = clean_transcribed
                    
                    # Step 2: Emotion analysis
                    emotion, confidence = get_emotion_from_audio(audio_path)
                    
                    # Add user input to terminal
                    add_to_terminal(clean_transcribed, "user", emotion, confidence)
                    
                    # Step 3: Generate clean AI response
                    clean_ai_response = generate_llm_response(clean_transcribed, emotion, confidence)
                    
                    # Add AI response to terminal
                    add_to_terminal(clean_ai_response, "assistant")
                    
                    # Step 4: Generate robot-compatible speech
                    speech_audio = text_to_speech_for_robot(clean_ai_response)
                    
                    if speech_audio:
                        st.session_state.last_ai_audio = speech_audio
                        st.session_state.ai_response_text = clean_ai_response
                        # Remove the "Voice response ready for sync" message
                    else:
                        add_to_terminal("Text response ready", "system")
                        
                    st.session_state.conversation_count += 1
                    
                    st.success("Conversation complete - Use main terminal for sync")
                    st.rerun()

# ============================================================================
# Simple Status and Controls
# ============================================================================

st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns(3)

with col_btn1:
    # Clear terminal
    if st.button("CLEAR CONVERSATION", use_container_width=True):
        st.session_state.terminal_log = "EMOTION-AWARE EMPATHETIC ASSISTANT v1.0\n" + "="*50 + "\n\n"
        st.session_state.terminal_log += "System: Conversation cleared\n\n"
        st.session_state.conversation_count = 0
        
        # Clear audio data safely
        st.session_state.last_ai_audio = None
        st.session_state.ai_response_text = ""
        st.session_state.current_input_text = ""
        
        st.rerun()

with col_btn2:
    # Process all files
    if len(audio_files) > 1:
        if st.button("PROCESS ALL VOICES", use_container_width=True):
            for file in audio_files:
                audio_path = os.path.join(AUDIO_FOLDER, file)
                if os.path.exists(audio_path):
                    add_to_terminal(f"Processing {file}", "system")
                    clean_transcribed = transcribe_audio(audio_path)
                    st.session_state.current_input_text = clean_transcribed
                    
                    emotion, confidence = get_emotion_from_audio(audio_path)
                    add_to_terminal(clean_transcribed, "user", emotion, confidence)
                    
                    clean_ai_response = generate_llm_response(clean_transcribed, emotion, confidence)
                    add_to_terminal(clean_ai_response, "assistant")
                    st.session_state.ai_response_text = clean_ai_response
                    st.session_state.conversation_count += 1
            
            st.success(f"Processed {len(audio_files)} voices")
            st.rerun()

with col_btn3:
    # Stats
    st.metric("Conversations", st.session_state.conversation_count)
    api_status = "Advanced" if OPENAI_API_KEY != "sk-your-openai-api-key-here" else "Basic"
    st.metric("Mode", api_status)
    
    # Quick Audio Test
    if st.button("🔊 TEST AUDIO", use_container_width=True):
        st.info("🧪 Testing TTS generation...")
        test_audio = text_to_speech_for_robot("This is an audio test for the robot system")
        if test_audio:
            st.session_state.last_ai_audio = test_audio
            st.session_state.ai_response_text = "This is an audio test for the robot system"
            st.success("✅ Test audio ready! Check AI Response player below")
            st.rerun()
        else:
            st.error("❌ Audio test failed - check internet connection")

# ============================================================================
# Footer - ROBOT INSTRUCTIONS
# ============================================================================

st.markdown("---")
st.markdown("""
**ROBOT OPERATION INSTRUCTIONS:**
1. Add audio files to ./sample-wav-files/ folder
2. Select voice file and click START EMPATHETIC CONVERSATION
3. Use main terminal PLAY buttons for perfect audio-text sync
4. All text is cleaned for robot compatibility (no punctuation or emojis)
5. Green terminal displays all conversations with timestamps

**MAIN TERMINAL SYNC CONTROLS:**
- PLAY YOUR VOICE: Syncs your audio with text highlighting
- PLAY AI RESPONSE: Syncs AI audio with text highlighting  
- STOP: Stops all audio and sync

**Supported formats:** WAV MP3 OGG M4A FLAC
**Robot compatible:** No punctuation, no emojis, clean text only
""")

# Live status - ROBOT READY
st.markdown('<p class="status-loading">EMPATHY ENGINE ACTIVE - MAIN TERMINAL SYNC READY</p>', unsafe_allow_html=True)