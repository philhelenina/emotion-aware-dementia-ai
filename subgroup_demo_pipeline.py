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
    page_title="🧠 Helpful AI Assistant",
    page_icon="💚",
    layout="wide"
)

# Custom CSS for empathetic theme (keeping original styling)
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
        min-height: 400px;
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
    
    #cursor {
        animation: blink 1s infinite;
        color: #90EE90;
        font-weight: bold;
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

# Initialize session state
if "terminal_log" not in st.session_state:
    st.session_state.terminal_log = "EMPATHETIC ASSISTANT v2.0 - Beyond Simple Emotion Recognition 💚\n" + "="*50 + "\n\n"

if "conversation_count" not in st.session_state:
    st.session_state.conversation_count = 0

if "show_typing" not in st.session_state:
    st.session_state.show_typing = False

if "typing_message" not in st.session_state:
    st.session_state.typing_message = ""

if "current_input_audio" not in st.session_state:
    st.session_state.current_input_audio = None

if "current_input_filename" not in st.session_state:
    st.session_state.current_input_filename = None

# ============================================================================
# Model Loading
# ============================================================================

@st.cache_resource
def load_models():
    """Load AI models"""
    try:
        # Load Whisper model
        st.info("Loading Whisper model...")
        whisper_model = whisper.load_model("base")
        st.success("✅ Whisper model loaded")
        
        # Load emotion classifier
        st.info("Loading emotion context analyzer...")
        audio_emotion_classifier = None
        try:
            # Try to load emotion2vec alternative with Ekman-like categories
            # Using the best available model that maps to Ekman's categories
            audio_emotion_classifier = pipeline(
                "audio-classification", 
                model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                return_all_scores=True
            )
            st.success("✅ Emotion classifier loaded")
            st.info("📌 Emotions help understand context, but focus is on helping you")
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
        else:
            st.error("Failed to load models. Please check your internet connection and try again.")
            st.stop()

# Get models from session state
whisper_model = st.session_state.get('whisper_model')
audio_emotion_classifier = st.session_state.get('audio_emotion_classifier')

# ============================================================================
# Ekman's Emotion Mapping
# ============================================================================

def map_to_ekman_categories(emotion):
    """Map various emotion labels to Ekman's 6 basic emotions + neutral"""
    # Ekman's 6 basic emotions: anger, disgust, fear, happiness, sadness, surprise
    # Plus neutral for practical purposes
    
    emotion_lower = emotion.lower()
    
    # Direct mappings to Ekman categories
    ekman_mapping = {
        # Anger mappings
        'angry': 'anger',
        'anger': 'anger',
        'frustrated': 'anger',
        'irritated': 'anger',
        'mad': 'anger',
        
        # Disgust mappings
        'disgust': 'disgust',
        'disgusted': 'disgust',
        'repulsed': 'disgust',
        
        # Fear mappings
        'fear': 'fear',
        'fearful': 'fear',
        'afraid': 'fear',
        'scared': 'fear',
        'anxious': 'fear',
        'worried': 'fear',
        
        # Happiness mappings
        'happy': 'happiness',
        'happiness': 'happiness',
        'joy': 'happiness',
        'joyful': 'happiness',
        'excited': 'happiness',
        'cheerful': 'happiness',
        'elated': 'happiness',
        
        # Sadness mappings
        'sad': 'sadness',
        'sadness': 'sadness',
        'depressed': 'sadness',
        'melancholy': 'sadness',
        'sorrowful': 'sadness',
        
        # Surprise mappings
        'surprise': 'surprise',
        'surprised': 'surprise',
        'astonished': 'surprise',
        'amazed': 'surprise',
        
        # Neutral mappings
        'neutral': 'neutral',
        'calm': 'neutral',
        'indifferent': 'neutral'
    }
    
    return ekman_mapping.get(emotion_lower, 'neutral')

# Ekman emotion emojis
EKMAN_EMOJIS = {
    'anger': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happiness': '😊',
    'sadness': '😢',
    'surprise': '😮',
    'neutral': '😐'
}

# ============================================================================
# Text Processing Functions (Modified for v2.0)
# ============================================================================

def prepare_text_for_tts(text):
    """Prepare text for TTS by handling special characters appropriately"""
    if not text:
        return ""
    
    # Replace emojis with descriptive text for TTS
    emoji_replacements = {
        '😊': 'smiling',
        '😄': 'happy',
        '😢': 'sad',
        '😭': 'crying',
        '😠': 'angry',
        '😡': 'very angry',
        '😨': 'fearful',
        '😮': 'surprised',
        '🤢': 'disgusted',
        '😐': 'neutral',
        '❤️': 'love',
        '💚': 'heart',
        '🎉': 'celebration',
        '👍': 'thumbs up',
        '👎': 'thumbs down',
        '🤔': 'thinking',
        '💭': 'thought',
        '💬': 'speech',
        '🎵': 'music',
        '🎤': 'microphone',
        '🤖': 'robot',
        '💝': 'gift',
        '✨': 'sparkle',
        '🚀': 'rocket',
        '🤗': 'hugging'
    }
    
    # Replace known emojis
    for emoji, replacement in emoji_replacements.items():
        text = text.replace(emoji, f' {replacement} ')
    
    # Remove any remaining emojis that aren't in our mapping
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(' ', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# ============================================================================
# Utility Functions (Modified)
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
    """Convert audio to text (keeping punctuation and structure)"""
    try:
        whisper_model = st.session_state.get('whisper_model')
        if not whisper_model:
            return "Error: Whisper model not loaded"
        
        result = whisper_model.transcribe(audio_path)
        # Keep the original text with punctuation
        return result["text"].strip()
    except Exception as e:
        return f"Error: Speech recognition failed - {e}"

def get_emotion_from_audio(audio_path):
    """Analyze emotion from audio and map to Ekman categories"""
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
            # Map to Ekman categories
            ekman_emotion = map_to_ekman_categories(top_emotion['label'])
            return ekman_emotion, top_emotion['score']
        
    except Exception as e:
        st.error(f"Emotion analysis error: {e}")
    
    return "neutral", 0.5

def generate_llm_response(text, emotion, confidence):
    """Generate response using LLM with deeper empathetic understanding"""
    
    if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-your-openai-api-key-here":
        return generate_simple_response(emotion)
    
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Deeper emotion-aware instructions
        emotion_contexts = {
            'happiness': {
                'focus': 'celebrate their joy and explore what made them happy',
                'approach': 'share their enthusiasm, ask about details, amplify positive moments'
            },
            'sadness': {
                'focus': 'understand the source of pain and offer genuine support',
                'approach': 'validate feelings, offer specific help, avoid toxic positivity'
            },
            'anger': {
                'focus': 'understand what triggered frustration and help find solutions',
                'approach': 'acknowledge the issue, help problem-solve, avoid dismissing feelings'
            },
            'fear': {
                'focus': 'identify specific concerns and provide practical reassurance',
                'approach': 'break down the fear, offer concrete steps, build confidence'
            },
            'surprise': {
                'focus': 'explore what surprised them and share their reaction',
                'approach': 'be curious about details, match their energy level'
            },
            'disgust': {
                'focus': 'understand what bothered them and validate their reaction',
                'approach': 'acknowledge their standards, help process the experience'
            },
            'neutral': {
                'focus': 'engage meaningfully with their topic',
                'approach': 'be helpful, ask clarifying questions, provide value'
            }
        }
        
        context = emotion_contexts.get(emotion, emotion_contexts['neutral'])
        
        prompt = f"""User said: "{text}"
Detected emotion: {emotion} (confidence: {confidence:.1f})

IMPORTANT: Don't just acknowledge their emotion. Instead:
- Focus: {context['focus']}
- Approach: {context['approach']}

Guidelines:
1. Address the CONTENT of what they said, not just the emotion
2. If they mention a problem, offer specific help or ask clarifying questions
3. If they share good news, ask for details and celebrate WITH them
4. If they're struggling, provide actionable support, not just sympathy
5. Use 1-2 emojis naturally, don't overdo it

Respond in 1-2 sentences that actually HELP them, not just mirror their feelings."""

        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant who provides real value in conversations. You understand emotions but focus on being genuinely helpful rather than just acknowledging feelings. You ask good questions, offer specific help, and engage with the actual content of what people say."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": 100,
            "temperature": 0.8
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
    """Generate simple fallback responses that are actually helpful"""
    responses = {
        "anger": "What's frustrating you? Let me help you work through this. 💪",
        "disgust": "That sounds really unpleasant. What specifically bothered you? 🤔",
        "fear": "What's your biggest concern right now? Let's tackle it together. 💚",
        "happiness": "That's fantastic! What made this happen? 🎉",
        "sadness": "What's been weighing on you? I'm here to help. 💚",
        "surprise": "Wow! Was this expected? Tell me more! 😮",
        "neutral": "What can I help you with today? 😊"
    }
    return responses.get(emotion, "What's on your mind? I'm here to help. 💚")

def text_to_speech(text):
    """Convert text to speech, handling emojis appropriately"""
    try:
        # Prepare text for TTS (convert emojis to descriptions)
        tts_text = prepare_text_for_tts(text)
        
        if not tts_text:
            return None
            
        tts = gTTS(text=tts_text, lang="en", slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return mp3_fp.getvalue()
    except Exception as e:
        return None

def add_to_terminal(text, role="system", emotion=None, confidence=None, enable_typing=False):
    """Add text to terminal log with Ekman emotion indicators"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if role == "user":
        st.session_state.terminal_log += f"[{timestamp}] 🎤 You: {text}\n"
        if emotion:
            emotion_emoji = EKMAN_EMOJIS.get(emotion, '💭')
            st.session_state.terminal_log += f"                  Detected emotion: {emotion} {emotion_emoji} (confidence: {confidence:.1f})\n"
    elif role == "assistant":
        if enable_typing:
            st.session_state.typing_message = f"[{timestamp}] 🤖 AI: {text}\n"
            st.session_state.show_typing = True
        else:
            st.session_state.terminal_log += f"[{timestamp}] 🤖 AI: {text}\n"
    else:
        st.session_state.terminal_log += f"[{timestamp}] ⚙️ {text}\n"
    
    if not enable_typing:
        st.session_state.terminal_log += "\n"

# ============================================================================
# Main UI
# ============================================================================

# Title with empathetic style
st.markdown("# 🧠 Emotion-Aware Empathetic Assistant v2.0")
st.markdown("**Beyond emotion recognition • Practical support • Real conversations 💚**")

# Create two columns - main terminal and controls
col1, col2 = st.columns([3, 1])

with col1:
    # Single terminal display area
    st.markdown("### 💬 Empathetic Conversation")
    
    # Display terminal with markdown (preserving emojis)
    terminal_container = st.container()
    with terminal_container:
        st.code(st.session_state.terminal_log, language="")

with col2:
    st.markdown("### 🤝 Empathy Controls")
    
    # Show Ekman emotions legend
    with st.expander("📊 Detected Emotions", expanded=False):
        for emotion, emoji in EKMAN_EMOJIS.items():
            st.write(f"{emoji} **{emotion.capitalize()}**")
    
    # Get audio file list
    audio_files = get_audio_files()
    
    if not audio_files:
        st.warning("⚠️ No audio files found")
        st.info(f"Add files to `{AUDIO_FOLDER}` folder")
    else:
        selected_file = st.selectbox(
            "🎤 Share Your Voice:",
            audio_files,
            key="file_selector"
        )
        
        # Load selected audio file into session state for INPUT player
        if selected_file:
            audio_path = os.path.join(AUDIO_FOLDER, selected_file)
            if os.path.exists(audio_path):
                # Read the INPUT audio file
                try:
                    with open(audio_path, 'rb') as f:
                        input_audio_bytes = f.read()
                    
                    # Store INPUT audio in session state
                    st.session_state.current_input_audio = input_audio_bytes
                    st.session_state.current_input_filename = selected_file
                    st.success(f"✅ {selected_file} ready!")
                    
                except Exception as e:
                    st.error(f"Error loading file: {e}")
        
        # Process button
        if st.button("💝 START EMPATHETIC CONVERSATION", type="primary", use_container_width=True):
            audio_path = os.path.join(AUDIO_FOLDER, selected_file)
            
            if os.path.exists(audio_path):
                # Process audio
                with st.spinner("Understanding what you need... 💭"):
                    # Step 1: Transcribe (with punctuation)
                    transcribed_text = transcribe_audio(audio_path)
                    
                    # Step 2: Emotion analysis (Ekman categories)
                    emotion, confidence = get_emotion_from_audio(audio_path)
                    
                    # Add user input to terminal
                    add_to_terminal(transcribed_text, "user", emotion, confidence)
                    
                    # Step 3: Generate response (with emotion awareness)
                    ai_response = generate_llm_response(transcribed_text, emotion, confidence)
                    
                    # Add AI response to terminal
                    add_to_terminal(ai_response, "assistant")
                    
                    # Step 4: Generate speech
                    speech_audio = text_to_speech(ai_response)
                    
                    # Store AI audio in session state
                    if speech_audio:
                        st.session_state.last_ai_audio = speech_audio
                        st.session_state.ai_response_text = ai_response
                    
                    st.session_state.conversation_count += 1
                    
                    st.success("💚 Empathetic conversation complete!")
                    st.rerun()

# ============================================================================
# AUDIO PLAYERS SECTION
# ============================================================================

st.markdown("---")
st.markdown("## 🎧 Voice Conversation")

# INPUT Audio Player
col_input, col_output = st.columns(2)

with col_input:
    st.markdown("### 🎤 Your Voice")
    
    if st.session_state.current_input_audio:
        st.markdown(f"**File:** {st.session_state.current_input_filename}")
        
        st.audio(st.session_state.current_input_audio, format="audio/wav")
        
        st.download_button(
            label="💾 Download Your Voice",
            data=st.session_state.current_input_audio,
            file_name=st.session_state.current_input_filename or "input.wav",
            mime="audio/wav",
            use_container_width=True
        )
    else:
        st.info("📁 Select your voice recording above")

# AI OUTPUT Audio Player
with col_output:
    st.markdown("### 🤖 Empathetic Response")
    
    if "last_ai_audio" in st.session_state and st.session_state.last_ai_audio:
        if "ai_response_text" in st.session_state:
            st.markdown(f"**Text:** {st.session_state.ai_response_text}")
        
        st.audio(st.session_state.last_ai_audio, format="audio/mp3")
        
        st.download_button(
            label="💾 Download Response",
            data=st.session_state.last_ai_audio,
            file_name=f"empathetic_response_{datetime.now().strftime('%H%M%S')}.mp3",
            mime="audio/mp3",
            use_container_width=True
        )
    else:
        st.info("🤖 Share your voice to receive a response")

# ============================================================================
# CONTROL BUTTONS
# ============================================================================

st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns(3)

with col_btn1:
    # Clear terminal
    if st.button("🧹 CLEAR CONVERSATION", use_container_width=True):
        st.session_state.terminal_log = "EMPATHETIC ASSISTANT v2.0 - Beyond Simple Emotion Recognition 💚\n" + "="*50 + "\n\n"
        st.session_state.conversation_count = 0
        if "last_ai_audio" in st.session_state:
            del st.session_state.last_ai_audio
        if "ai_response_text" in st.session_state:
            del st.session_state.ai_response_text
        st.rerun()

with col_btn2:
    # Quick process all files
    if len(audio_files) > 1:
        if st.button("💝 PROCESS ALL FILES", use_container_width=True):
            for file in audio_files:
                audio_path = os.path.join(AUDIO_FOLDER, file)
                if os.path.exists(audio_path):
                    transcribed = transcribe_audio(audio_path)
                    emotion, confidence = get_emotion_from_audio(audio_path)
                    add_to_terminal(transcribed, "user", emotion, confidence)
                    ai_response = generate_llm_response(transcribed, emotion, confidence)
                    add_to_terminal(ai_response, "assistant")
                    st.session_state.conversation_count += 1
            
            st.success(f"💚 Completed {len(audio_files)} conversations!")
            st.rerun()

with col_btn3:
    # Stats
    st.metric("Conversations", f"{st.session_state.conversation_count} 💬")
    
    # API Status with emoji
    if OPENAI_API_KEY and OPENAI_API_KEY != "sk-your-openai-api-key-here":
        api_status = "💚 Advanced"
    else:
        api_status = "💛 Basic"
    st.metric("AI Mode", api_status)

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown("""
**🤝 How It Works:**
1. 📁 Add audio files to `./sample-wav-files/` folder
2. 🎤 Select your voice recording from dropdown
3. 💝 Click 'START EMPATHETIC CONVERSATION' to begin
4. 💬 Get practical, helpful responses - not just emotion mirroring
5. 🔊 Listen to AI's response

**What makes this different:**
- ✅ Focuses on YOUR CONTENT, not just emotions
- ✅ Asks clarifying questions to better help
- ✅ Offers specific suggestions and solutions
- ✅ Celebrates your wins and supports your struggles
- ✅ Has real conversations, not just "I see you're sad"

**Supported formats:** WAV, MP3, OGG, M4A, FLAC  

**📊 Emotion Detection (Ekman's Model):**
😊 Happiness • 😢 Sadness • 😠 Anger • 😨 Fear • 😮 Surprise • 🤢 Disgust • 😐 Neutral
""")

# Live status indicator
st.markdown('<p class="status-loading">💚 READY FOR MEANINGFUL CONVERSATION</p>', unsafe_allow_html=True)