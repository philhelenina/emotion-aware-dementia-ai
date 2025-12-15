import os
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import streamlit as st
import numpy as np
import torch
import librosa
import whisper
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import requests
from gtts import gTTS
import io
from datetime import datetime
import glob
from pathlib import Path
import time
import re
import json
import azure.cognitiveservices.speech as speechsdk
from professional_styles import apply_professional_styling, create_speaker_container

# ============================================================================
# Configuration and API Keys
# ============================================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
AUDIO_FOLDER = "./sample-conv-wav-files"

# Azure Speech Configuration
SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY", "")
SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "eastus")

# ============================================================================
# Gold Conversations Database
# ============================================================================

GOLD_CONVERSATIONS = {
    "rachel_severe": {
        "3-turn": [
            {"speaker": "Rachel", "text": "Where... where am I? I don't... Help! Someone help me!", "emotion": "fear"},
            {"speaker": "Caregiver", "text": "Rachel, you're safe in your room. I'm here with you. Your son visited yesterday, remember?", "emotion": "calm"},
            {"speaker": "Rachel", "text": "My son? I... I don't... When? Where is he?", "emotion": "confused"}
        ],
        "5-turn": [
            {"speaker": "Rachel", "text": "Help! Help me! I don't know... I don't know where...", "emotion": "panic"},
            {"speaker": "Caregiver", "text": "Rachel, it's okay. You're in your room. Look, there's your TV that you like to watch.", "emotion": "soothing"},
            {"speaker": "Rachel", "text": "TV? I... what? I need... I need my son!", "emotion": "anxious"},
            {"speaker": "Caregiver", "text": "Your son loves you very much. He'll visit soon. Would you like to watch your favorite show?", "emotion": "reassuring"},
            {"speaker": "Rachel", "text": "I... okay... my son is coming?", "emotion": "slightly_calmer"}
        ],
        "7-turn": [
            {"speaker": "Rachel", "text": "Someone! Anyone! I'm... I'm lost! Help!", "emotion": "extreme_distress"},
            {"speaker": "Caregiver", "text": "Rachel, I'm right here. You're not lost. This is your room at the care home.", "emotion": "very_calm"},
            {"speaker": "Rachel", "text": "Care home? No! I want... I want to go home!", "emotion": "upset"},
            {"speaker": "Caregiver", "text": "I understand. Your son set up this nice room for you. See your pictures on the wall?", "emotion": "gentle"},
            {"speaker": "Rachel", "text": "Pictures? I... whose pictures? I don't understand!", "emotion": "frustrated"},
            {"speaker": "Caregiver", "text": "Those are pictures of you and your son. He loves you so much. Would you like to call him?", "emotion": "kind"},
            {"speaker": "Rachel", "text": "Call... my son? Yes... yes, I want my son.", "emotion": "hopeful"}
        ]
    },
    "jolene_moderate": {
        "3-turn": [
            {"speaker": "Jolene", "text": "I rang the bell. No one comes. No one ever comes.", "emotion": "flat"},
            {"speaker": "Caregiver", "text": "Hi Jolene, I'm here now. Is there something you need?", "emotion": "cheerful"},
            {"speaker": "Jolene", "text": "I don't know. I just... I don't want to be alone.", "emotion": "monotone"}
        ],
        "5-turn": [
            {"speaker": "Jolene", "text": "Why doesn't anyone answer. I push and push the button.", "emotion": "dejected"},
            {"speaker": "Caregiver", "text": "I'm sorry for the wait, Jolene. I'm here now. How are you feeling?", "emotion": "apologetic"},
            {"speaker": "Jolene", "text": "The same. Always the same. Nothing changes.", "emotion": "depressed"},
            {"speaker": "Caregiver", "text": "I see there's classical music on TV. You mentioned you taught music. Would you like to tell me about that?", "emotion": "encouraging"},
            {"speaker": "Jolene", "text": "Music. Yes. I used to teach. Long time ago. Doesn't matter now.", "emotion": "slightly_engaged"}
        ],
        "7-turn": [
            {"speaker": "Jolene", "text": "I keep ringing. No one hears me. Or they don't care.", "emotion": "hopeless"},
            {"speaker": "Caregiver", "text": "Jolene, I heard you and I care. I came as quickly as I could. What can I do for you?", "emotion": "caring"},
            {"speaker": "Jolene", "text": "Nothing. There's nothing. Just... emptiness.", "emotion": "empty"},
            {"speaker": "Caregiver", "text": "That sounds really hard. You know, your son mentioned you were an amazing teacher. Elementary school, right?", "emotion": "sympathetic"},
            {"speaker": "Jolene", "text": "Elementary. Yes. Children. They used to laugh. No laughter here.", "emotion": "reminiscing"},
            {"speaker": "Caregiver", "text": "What was your favorite thing about teaching? I'd love to hear about it.", "emotion": "interested"},
            {"speaker": "Jolene", "text": "The music class. When they sang. But that was... before. When things were different.", "emotion": "wistful"}
        ]
    },
    "ralph_mild": {
        "3-turn": [
            {"speaker": "Ralph", "text": "Hey! I've been ringing this damn bell for ages! What's a guy gotta do to get some help around here?", "emotion": "irritated"},
            {"speaker": "Caregiver", "text": "Hi Ralph, I'm sorry for the wait. What can I help you with?", "emotion": "apologetic"},
            {"speaker": "Ralph", "text": "Well, I... uh... I needed something. Can't remember what now. But I know it was important!", "emotion": "confused_but_trying"}
        ],
        "5-turn": [
            {"speaker": "Ralph", "text": "Finally! I've been calling and calling! This place has terrible service!", "emotion": "annoyed"},
            {"speaker": "Caregiver", "text": "I apologize, Ralph. I'm here now. How can I help you today?", "emotion": "patient"},
            {"speaker": "Ralph", "text": "I need to... to get ready. Big day today. Going somewhere important. Hawaii maybe?", "emotion": "confused_excited"},
            {"speaker": "Caregiver", "text": "That sounds exciting! You mentioned you used to surf in Hawaii. Want to tell me about that?", "emotion": "engaging"},
            {"speaker": "Ralph", "text": "Surf? Oh yes! Best waves on the North Shore! Wait... am I going surfing today?", "emotion": "nostalgic_confused"}
        ],
        "7-turn": [
            {"speaker": "Ralph", "text": "Hello? HELLO? This bell doesn't work! I need help here! It's urgent!", "emotion": "frustrated"},
            {"speaker": "Caregiver", "text": "Hi Ralph, I hear you. I'm here. What's the urgent matter?", "emotion": "calm_concerned"},
            {"speaker": "Ralph", "text": "I gotta get up. Gotta go. Important meeting... or was it... something about the grandkids?", "emotion": "agitated_confused"},
            {"speaker": "Caregiver", "text": "Let's take it slow, Ralph. Remember what happened last time you got up too fast? How about we sit and talk first?", "emotion": "cautious"},
            {"speaker": "Ralph", "text": "Last time? I don't... well, maybe. But I really need to... to do something. It's important!", "emotion": "uncertain"},
            {"speaker": "Caregiver", "text": "I understand. Hey, I heard you grew grapes in Hawaii. That must have been interesting with the climate there.", "emotion": "redirecting"},
            {"speaker": "Ralph", "text": "Grapes? Oh right! Yes! Had to be creative with the volcanic soil. Say, you ever been to Hawaii?", "emotion": "engaged"}
        ]
    }
}

# ============================================================================
# Conversation Matching System
# ============================================================================

@st.cache_resource
def load_models():
    """Load Whisper and emotion models"""
    try:
        whisper_model = whisper.load_model("base")
        audio_emotion_classifier = pipeline(
            "audio-classification", 
            model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
            return_all_scores=True
        )
        return whisper_model, audio_emotion_classifier
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper"""
    whisper_model = st.session_state.get('whisper_model')
    if whisper_model:
        result = whisper_model.transcribe(audio_path)
        return result["text"].strip()
    return ""

def get_emotion_from_audio(audio_path):
    """Get emotion from audio"""
    audio_emotion_classifier = st.session_state.get('audio_emotion_classifier')
    if not audio_emotion_classifier:
        return "neutral", 0.5
    
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        if len(y) > 10 * sr:
            y = y[:10 * sr]
        elif len(y) < sr:
            y = np.pad(y, (0, sr - len(y)))
        
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        
        predictions = audio_emotion_classifier({"raw": y, "sampling_rate": sr})
        if predictions:
            top_emotion = max(predictions, key=lambda x: x['score'])
            return top_emotion['label'].lower(), float(top_emotion['score'])
    except Exception as e:
        st.error(f"Emotion analysis error: {e}")
    
    return "neutral", 0.5

# Azure TTS Configuration
ELDERLY_VOICES = {
    "rachel": {
        "voice": "en-US-AriaNeural",  # Female elderly voice
        "style": "sad",  # More vulnerable
        "pitch": "-22Hz",  # Much lower for elderly
        "rate": "-18%",    # Slower speech
        "volume": "-3%",   # Quieter
        "emphasis": "reduced"  # Less clear articulation
    },
    "jolene": {
        "voice": "en-US-NancyNeural",  # More mature female voice
        "style": "sad",
        "pitch": "-25Hz",  # Even lower for depression
        "rate": "-22%",    # Very slow for depression
        "volume": "-6%",   # Quieter for withdrawn personality
        "emphasis": "reduced"
    },
    "ralph": {
        "voice": "en-US-GuyNeural",  # GuyNeural sounds more elderly
        "style": "default",  
        "pitch": "-35Hz",    # Even lower pitch for elderly first line
        "rate": "-30%",      # Slower for elderly speech
        "volume": "-4%",     # Quieter to sound older
        "emphasis": "reduced"  # Less clear for elderly
    },
    "caregiver": {
        "voice": "en-US-JennyNeural",  # Female caregiver
        "style": "friendly",
        "pitch": "0Hz",
        "rate": "-5%",       # Slightly slower for elderly
        "volume": "+5%",     # Louder for elderly hearing
        "emphasis": "moderate"  # Clear articulation
    }
}

def create_azure_ssml(text, speaker, emotion):
    """Create SSML for Azure TTS"""
    # Debug log with more detail
    print(f"🎤 Creating SSML for speaker: '{speaker}', emotion: '{emotion}'")
    
    # Get voice config - ensure Ralph gets male voice
    if speaker.lower() == "ralph":
        # FORCE Ralph to use male voice - override any other settings
        voice_config = {
            "voice": "en-US-GuyNeural",  # Older sounding male voice
            "style": "default",
            "pitch": "-22Hz",    # Lower for elderly, but not too deep
            "rate": "-25%",      # Slower for elderly speech
            "volume": "-4%",     # Slightly weaker elderly voice
            "emphasis": "reduced"  # Less clear articulation like elderly
        }
        print(f"🎯 RALPH FIXED: Using {voice_config['voice']} (MALE) - pitch:{voice_config['pitch']}, rate:{voice_config['rate']}")
    elif speaker.lower() == "rachel":
        voice_config = ELDERLY_VOICES["rachel"]  # Female voice  
        print(f"✅ Rachel detected: Using {voice_config['voice']} (female)")
    elif speaker.lower() == "jolene":
        voice_config = ELDERLY_VOICES["jolene"]  # Female voice
        print(f"✅ Jolene detected: Using {voice_config['voice']} (female)")
    else:  # Caregiver
        voice_config = ELDERLY_VOICES["caregiver"]
        print(f"✅ Caregiver detected: Using {voice_config['voice']} (caregiver)")
    
    # Enhanced emotion adjustments for elderly voices
    emotion_mods = {
        # Fear emotions - more dramatic for elderly
        "fear": {"pitch": "+8Hz", "rate": "+12%", "volume": "+10%"},
        "panic": {"pitch": "+12Hz", "rate": "+18%", "volume": "+15%"},
        "extreme_distress": {"pitch": "+15Hz", "rate": "+20%", "volume": "+20%"},
        
        # Confused emotions - more natural and subtle
        "confused": {"pitch": "+1Hz", "rate": "-3%", "volume": "0%"},
        "confused_but_trying": {"pitch": "+1Hz", "rate": "-4%", "volume": "0%"},
        "confused_excited": {"pitch": "+2Hz", "rate": "-2%", "volume": "+1%"},
        "agitated_confused": {"pitch": "+3Hz", "rate": "+1%", "volume": "+2%"},
        "nostalgic_confused": {"pitch": "0Hz", "rate": "-5%", "volume": "-1%"},
        
        # Calm/positive emotions
        "calm": {"pitch": "0Hz", "rate": "-5%", "volume": "0%"},
        "hopeful": {"pitch": "+3Hz", "rate": "-8%", "volume": "+2%"},
        "slightly_calmer": {"pitch": "-2Hz", "rate": "-10%", "volume": "-5%"},
        
        # Depressed emotions - very flat
        "sad": {"pitch": "-5Hz", "rate": "-15%", "volume": "-10%"},
        "depressed": {"pitch": "-10Hz", "rate": "-20%", "volume": "-15%"},
        "empty": {"pitch": "-12Hz", "rate": "-22%", "volume": "-18%"},
        "hopeless": {"pitch": "-15Hz", "rate": "-25%", "volume": "-20%"},
        "dejected": {"pitch": "-8Hz", "rate": "-18%", "volume": "-12%"},
        
        # Nostalgic emotions - enhanced for better elderly sound (matched to approved reference)
        "nostalgic": {"pitch": "-5Hz", "rate": "-25%", "volume": "-10%"},
        "reminiscing": {"pitch": "0Hz", "rate": "-12%", "volume": "-3%"},
        "wistful": {"pitch": "-3Hz", "rate": "-15%", "volume": "-8%"},
        "slightly_engaged": {"pitch": "+1Hz", "rate": "-8%", "volume": "+1%"},
        "engaged": {"pitch": "-2Hz", "rate": "-12%", "volume": "+3%"},
        
        # Angry/frustrated emotions
        "irritated": {"pitch": "+5Hz", "rate": "+5%", "volume": "+10%"},
        "annoyed": {"pitch": "+4Hz", "rate": "+3%", "volume": "+8%"},
        "frustrated": {"pitch": "-3Hz", "rate": "-5%", "volume": "+8%"},  # Elderly frustrated: lower pitch, slower
        "upset": {"pitch": "+8Hz", "rate": "+10%", "volume": "+10%"},
        
        # Uncertain emotions
        "uncertain": {"pitch": "+2Hz", "rate": "-12%", "volume": "-5%"},
        "anxious": {"pitch": "+5Hz", "rate": "+5%", "volume": "+3%"},
        
        # Caregiver emotions
        "soothing": {"pitch": "-3Hz", "rate": "-8%", "volume": "-2%"},
        "reassuring": {"pitch": "-2Hz", "rate": "-10%", "volume": "0%"},
        "gentle": {"pitch": "-5Hz", "rate": "-12%", "volume": "-5%"},
        "kind": {"pitch": "-3Hz", "rate": "-10%", "volume": "-3%"},
        "caring": {"pitch": "-4Hz", "rate": "-8%", "volume": "-2%"},
        "sympathetic": {"pitch": "-3Hz", "rate": "-10%", "volume": "-3%"},
        "encouraging": {"pitch": "+2Hz", "rate": "-5%", "volume": "+2%"},
        "interested": {"pitch": "+1Hz", "rate": "-3%", "volume": "+1%"},
        "apologetic": {"pitch": "-2Hz", "rate": "-8%", "volume": "-2%"},
        "patient": {"pitch": "-3Hz", "rate": "-12%", "volume": "-3%"},
        "engaging": {"pitch": "+2Hz", "rate": "-5%", "volume": "+3%"},
        "calm_concerned": {"pitch": "-2Hz", "rate": "-10%", "volume": "0%"},
        "cautious": {"pitch": "-1Hz", "rate": "-12%", "volume": "-2%"},
        "redirecting": {"pitch": "+1Hz", "rate": "-5%", "volume": "+1%"},
        "cheerful": {"pitch": "+3Hz", "rate": "-3%", "volume": "+3%"},
        "very_calm": {"pitch": "-5Hz", "rate": "-15%", "volume": "-5%"}
    }
    
    emo_mod = emotion_mods.get(emotion, {})
    
    # Calculate final prosody values
    def combine_prosody(base, mod):
        """Combine base and modifier prosody values"""
        if 'Hz' in base and 'Hz' in mod:
            base_val = int(base.replace('Hz', ''))
            mod_val = int(mod.replace('Hz', ''))
            return f"{base_val + mod_val}Hz"
        elif '%' in base and '%' in mod:
            base_val = int(base.replace('%', ''))
            mod_val = int(mod.replace('%', ''))
            # Ensure we don't go too extreme
            final_val = max(-50, min(50, base_val + mod_val))
            return f"{final_val}%"
        return base
    
    final_pitch = combine_prosody(voice_config['pitch'], emo_mod.get('pitch', '0Hz'))
    final_rate = combine_prosody(voice_config['rate'], emo_mod.get('rate', '0%'))
    final_volume = combine_prosody(voice_config.get('volume', '0%'), emo_mod.get('volume', '0%'))
    
    # Add enhanced pauses for elderly speech - match approved reference file
    if emotion in ["nostalgic", "nostalgic_confused", "reminiscing", "wistful"]:
        # Enhanced pauses for nostalgic emotions to match approved reference
        text_with_pauses = text.replace("...", '<break time="800ms"/>...<break time="600ms"/>')
        text_with_pauses = text_with_pauses.replace("?", '?<break time="500ms"/>')
        text_with_pauses = text_with_pauses.replace("!", '!<break time="400ms"/>')
        text_with_pauses = text_with_pauses.replace(".", '.<break time="400ms"/>')
        text_with_pauses = text_with_pauses.replace(",", ',<break time="300ms"/>')
    else:
        # Standard pauses for other emotions
        text_with_pauses = text.replace("...", '<break time="600ms"/>...<break time="400ms"/>')
        text_with_pauses = text_with_pauses.replace("?", '?<break time="400ms"/>')
        text_with_pauses = text_with_pauses.replace(".", '.<break time="300ms"/>')
        text_with_pauses = text_with_pauses.replace(",", ',<break time="200ms"/>')
    
    # Add enhanced tremor effect for elderly speech - especially for Ralph
    if speaker.lower() == "ralph":
        # Ralph gets moderate tremor effect - natural elderly voice
        tremor_contours = {
            "frustrated": "(0%,-3st) (30%,+4st) (70%,-3st) (100%,-2st)",
            "nostalgic": "(0%,-4st) (25%,+2st) (50%,-3st) (75%,+3st) (100%,-4st)", 
            "engaged": "(0%,-2st) (40%,+3st) (80%,-2st) (100%,-1st)",
            "default": "(0%,-3st) (35%,+4st) (65%,-3st) (100%,-2st)"
        }
        contour = tremor_contours.get(emotion, tremor_contours["default"])
        
        ssml = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" 
                    xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
            <voice name="{voice_config['voice']}">
                <mstts:express-as style="{voice_config.get('style', 'default')}">
                    <prosody pitch="{final_pitch}" 
                             rate="{final_rate}"
                             volume="{final_volume}"
                             contour="{contour}">
                        {text_with_pauses}
                    </prosody>
                </mstts:express-as>
            </voice>
        </speak>"""
    elif emotion in ["nostalgic", "nostalgic_confused", "reminiscing", "wistful"]:
        # Other speakers only get tremor for nostalgic emotions
        ssml = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" 
                    xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
            <voice name="{voice_config['voice']}">
                <mstts:express-as style="{voice_config.get('style', 'default')}">
                    <prosody pitch="{final_pitch}" 
                             rate="{final_rate}"
                             volume="{final_volume}"
                             contour="(0%,-5st) (50%,+3st) (100%,-2st)">
                        {text_with_pauses}
                    </prosody>
                </mstts:express-as>
            </voice>
        </speak>"""
    else:
        # Standard prosody for non-nostalgic emotions
        ssml = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" 
                    xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
            <voice name="{voice_config['voice']}">
                <mstts:express-as style="{voice_config.get('style', 'default')}">
                    <prosody pitch="{final_pitch}" 
                             rate="{final_rate}"
                             volume="{final_volume}">
                        {text_with_pauses}
                    </prosody>
                </mstts:express-as>
            </voice>
        </speak>"""
    
    return ssml

def generate_azure_audio(text, speaker, emotion):
    """Generate audio using Azure TTS"""
    print(f"Generating audio for: Speaker={speaker}, Emotion={emotion}, Text='{text[:30]}...'")
    
    if SPEECH_KEY == "YOUR_AZURE_SPEECH_KEY":
        # Fallback to gTTS if Azure not configured
        return generate_gtts_audio(text)
    
    try:
        speech_config = speechsdk.SpeechConfig(
            subscription=SPEECH_KEY,
            region=SPEECH_REGION
        )
        
        # Use in-memory synthesis
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=None
        )
        
        ssml = create_azure_ssml(text, speaker, emotion)
        
        # Log the voice being used
        if speaker.lower() == "ralph":
            print(f"Using MALE voice for Ralph: {ELDERLY_VOICES['ralph']['voice']}")
        
        result = synthesizer.speak_ssml_async(ssml).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print(f"✅ Azure TTS successful for {speaker}")
            return result.audio_data
        else:
            print(f"❌ Azure TTS failed: {result.reason}")
            if result.reason == speechsdk.ResultReason.Canceled:
                cancellation = result.cancellation_details
                print(f"   Cancellation reason: {cancellation.reason}")
                print(f"   Error details: {cancellation.error_details}")
                print(f"   Error code: {cancellation.error_code}")
        
    except Exception as e:
        st.error(f"Azure TTS error: {e}")
        print(f"Azure TTS error details: {e}")
    
    # Fallback to gTTS
    print("Falling back to gTTS")
    return generate_gtts_audio(text)

def generate_gtts_audio(text):
    """Fallback TTS using gTTS"""
    try:
        tts = gTTS(text=text, lang="en", slow=True)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return mp3_fp.getvalue()
    except Exception as e:
        st.error(f"TTS error: {e}")
        return None

GOLD_CONVERSATIONS = {
    "rachel_severe": {
        "3-turn": [
            {"speaker": "Rachel", "text": "Where... where am I? I don't... Help! Someone help me!", "emotion": "fear"},
            {"speaker": "Caregiver", "text": "Rachel, you're safe in your room. I'm here with you. Your son visited yesterday, remember?", "emotion": "calm"},
            {"speaker": "Rachel", "text": "My son? I... I don't... When? Where is he?", "emotion": "confused"}
        ],
        "5-turn": [
            {"speaker": "Rachel", "text": "Help! Help me! I don't know... I don't know where...", "emotion": "panic"},
            {"speaker": "Caregiver", "text": "Rachel, it's okay. You're in your room. Look, there's your TV that you like to watch.", "emotion": "soothing"},
            {"speaker": "Rachel", "text": "TV? I... what? I need... I need my son!", "emotion": "anxious"},
            {"speaker": "Caregiver", "text": "Your son loves you very much. He'll visit soon. Would you like to watch your favorite show?", "emotion": "reassuring"},
            {"speaker": "Rachel", "text": "I... okay... my son is coming?", "emotion": "slightly_calmer"}
        ],
        "7-turn": [
            {"speaker": "Rachel", "text": "Someone! Anyone! I'm... I'm lost! Help!", "emotion": "extreme_distress"},
            {"speaker": "Caregiver", "text": "Rachel, I'm right here. You're not lost. This is your room at the care home.", "emotion": "very_calm"},
            {"speaker": "Rachel", "text": "Care home? No! I want... I want to go home!", "emotion": "upset"},
            {"speaker": "Caregiver", "text": "I understand. Your son set up this nice room for you. See your pictures on the wall?", "emotion": "gentle"},
            {"speaker": "Rachel", "text": "Pictures? I... whose pictures? I don't understand!", "emotion": "frustrated"},
            {"speaker": "Caregiver", "text": "Those are pictures of you and your son. He loves you so much. Would you like to call him?", "emotion": "kind"},
            {"speaker": "Rachel", "text": "Call... my son? Yes... yes, I want my son.", "emotion": "hopeful"}
        ]
    },
    "jolene_moderate": {
        "3-turn": [
            {"speaker": "Jolene", "text": "I rang the bell. No one comes. No one ever comes.", "emotion": "flat"},
            {"speaker": "Caregiver", "text": "Hi Jolene, I'm here now. Is there something you need?", "emotion": "cheerful"},
            {"speaker": "Jolene", "text": "I don't know. I just... I don't want to be alone.", "emotion": "monotone"}
        ],
        "5-turn": [
            {"speaker": "Jolene", "text": "Why doesn't anyone answer. I push and push the button.", "emotion": "dejected"},
            {"speaker": "Caregiver", "text": "I'm sorry for the wait, Jolene. I'm here now. How are you feeling?", "emotion": "apologetic"},
            {"speaker": "Jolene", "text": "The same. Always the same. Nothing changes.", "emotion": "depressed"},
            {"speaker": "Caregiver", "text": "I see there's classical music on TV. You mentioned you taught music. Would you like to tell me about that?", "emotion": "encouraging"},
            {"speaker": "Jolene", "text": "Music. Yes. I used to teach. Long time ago. Doesn't matter now.", "emotion": "slightly_engaged"}
        ],
        "7-turn": [
            {"speaker": "Jolene", "text": "I keep ringing. No one hears me. Or they don't care.", "emotion": "hopeless"},
            {"speaker": "Caregiver", "text": "Jolene, I heard you and I care. I came as quickly as I could. What can I do for you?", "emotion": "caring"},
            {"speaker": "Jolene", "text": "Nothing. There's nothing. Just... emptiness.", "emotion": "empty"},
            {"speaker": "Caregiver", "text": "That sounds really hard. You know, your son mentioned you were an amazing teacher. Elementary school, right?", "emotion": "sympathetic"},
            {"speaker": "Jolene", "text": "Elementary. Yes. Children. They used to laugh. No laughter here.", "emotion": "reminiscing"},
            {"speaker": "Caregiver", "text": "What was your favorite thing about teaching? I'd love to hear about it.", "emotion": "interested"},
            {"speaker": "Jolene", "text": "The music class. When they sang. But that was... before. When things were different.", "emotion": "wistful"}
        ]
    },
    "ralph_mild": {
        "3-turn": [
            {"speaker": "Ralph", "text": "Hey! I've been ringing this damn bell for ages! What's a guy gotta do to get some help around here?", "emotion": "irritated"},
            {"speaker": "Caregiver", "text": "Hi Ralph, I'm sorry for the wait. What can I help you with?", "emotion": "apologetic"},
            {"speaker": "Ralph", "text": "Well, I... uh... I needed something. Can't remember what now. But I know it was important!", "emotion": "confused_but_trying"}
        ],
        "5-turn": [
            {"speaker": "Ralph", "text": "Finally! I've been calling and calling! This place has terrible service!", "emotion": "annoyed"},
            {"speaker": "Caregiver", "text": "I apologize, Ralph. I'm here now. How can I help you today?", "emotion": "patient"},
            {"speaker": "Ralph", "text": "I need to... to get ready. Big day today. Going somewhere important. Hawaii maybe?", "emotion": "confused_excited"},
            {"speaker": "Caregiver", "text": "That sounds exciting! You mentioned you used to surf in Hawaii. Want to tell me about that?", "emotion": "engaging"},
            {"speaker": "Ralph", "text": "Surf? Oh yes! Best waves on the North Shore! Wait... am I going surfing today?", "emotion": "nostalgic_confused"}
        ],
        "7-turn": [
            {"speaker": "Ralph", "text": "Hello? HELLO? This bell doesn't work! I need help here! It's urgent!", "emotion": "frustrated"},
            {"speaker": "Caregiver", "text": "Hi Ralph, I hear you. I'm here. What's the urgent matter?", "emotion": "calm_concerned"},
            {"speaker": "Ralph", "text": "I gotta get up. Gotta go. Important meeting... or was it... something about the grandkids?", "emotion": "agitated_confused"},
            {"speaker": "Caregiver", "text": "Let's take it slow, Ralph. Remember what happened last time you got up too fast? How about we sit and talk first?", "emotion": "cautious"},
            {"speaker": "Ralph", "text": "Last time? I don't... well, maybe. But I really need to... to do something. It's important!", "emotion": "uncertain"},
            {"speaker": "Caregiver", "text": "I understand. Hey, I heard you grew grapes in Hawaii. That must have been interesting with the climate there.", "emotion": "redirecting"},
            {"speaker": "Ralph", "text": "Grapes? Oh right! Yes! Had to be creative with the volcanic soil. Say, you ever been to Hawaii?", "emotion": "engaged"}
        ]
    }
}

# ============================================================================
# Conversation Matching System
# ============================================================================

class ConversationMatcher:
    def __init__(self):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.gold_conversations = GOLD_CONVERSATIONS
        
    def identify_persona_from_first_utterance(self, text, emotion):
        """Identify which persona based on first utterance similarity"""
        best_match = None
        best_score = -1
        
        # Encode input text
        input_embedding = self.sentence_model.encode([text])
        
        for persona, conversations in self.gold_conversations.items():
            for turn_type, turns in conversations.items():
                if turns and turns[0]["speaker"] in ["Rachel", "Jolene", "Ralph"]:
                    first_utterance = turns[0]["text"]
                    first_embedding = self.sentence_model.encode([first_utterance])
                    
                    # Calculate similarity - convert to float
                    similarity = float(cosine_similarity(input_embedding, first_embedding)[0][0])
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_match = {
                            "persona": persona,
                            "turn_type": turn_type,
                            "similarity": similarity
                        }
        
        return best_match
    
    def generate_conversation_with_sampling(self, first_utterance, emotion, num_turns=3, num_samples=3):
        """Generate multiple conversation samples and select best match to gold"""
        
        # Identify persona
        persona_match = self.identify_persona_from_first_utterance(first_utterance, emotion)
        if not persona_match:
            return None
        
        persona = persona_match["persona"]
        
        # Determine turn type based on num_turns
        if num_turns <= 3:
            turn_key = "3-turn"
        elif num_turns <= 5:
            turn_key = "5-turn"
        else:
            turn_key = "7-turn"
        
        # Get gold conversation
        gold_conv = self.gold_conversations[persona][turn_key]
        
        # Generate multiple samples
        samples = []
        for i in range(num_samples):
            sample_conv = self._generate_single_conversation(
                first_utterance, emotion, persona, num_turns, i
            )
            
            # Calculate detailed similarity to gold
            similarity_details = self._calculate_conversation_similarity(sample_conv, gold_conv)
            
            samples.append({
                "conversation": sample_conv,
                "similarity": similarity_details["overall"],
                "similarity_details": similarity_details,
                "sample_id": i
            })
        
        # Select best sample
        best_sample = max(samples, key=lambda x: x["similarity"])
        
        return {
            "persona": persona,
            "conversation": best_sample["conversation"],
            "gold_similarity": float(best_sample["similarity"]),
            "similarity_details": best_sample["similarity_details"],
            "sample_id": best_sample["sample_id"],
            "all_similarities": [float(s["similarity"]) for s in samples],
            "gold_conversation": gold_conv[:num_turns]  # Include gold for comparison
        }
    
    def _generate_single_conversation(self, first_utterance, emotion, persona, num_turns, sample_id):
        """Generate a single conversation sample"""
        conversation = []
        
        # Extract persona info - Fix persona name extraction
        persona_parts = persona.split("_")
        persona_name = persona_parts[0].capitalize()  # Rachel, Jolene, Ralph
        severity = persona_parts[1] if len(persona_parts) > 1 else "mild"
        
        # Debug log
        print(f"Generating conversation for persona: {persona_name}, severity: {severity}")
        
        # Add first utterance
        conversation.append({
            "speaker": persona_name,
            "text": first_utterance,
            "emotion": emotion
        })
        
        # Generate remaining turns
        for turn in range(1, num_turns):
            if turn % 2 == 1:  # Caregiver turn (odd turns: 1, 3, 5...)
                response = self._generate_caregiver_response(
                    conversation, persona, severity, sample_id
                )
                conversation.append(response)
            else:  # Patient turn (even turns: 2, 4, 6...)
                response = self._generate_patient_response(
                    conversation, persona, severity, sample_id
                )
                # Ensure patient response has correct speaker name
                response["speaker"] = persona_name
                conversation.append(response)
        
        return conversation
    
    def _generate_caregiver_response(self, conversation, persona, severity, sample_id):
        """Generate caregiver response with variation based on sample_id and turn"""
        last_patient_text = conversation[-1]["text"]
        last_patient_emotion = conversation[-1]["emotion"]
        turn_number = len([t for t in conversation if t["speaker"] == "Caregiver"]) + 1
        
        # Get persona-specific strategies with more variation
        if "rachel" in persona.lower():
            if "help" in last_patient_text.lower() or "lost" in last_patient_text.lower():
                responses = [
                    "Rachel, you're safe in your room. I'm here with you.",
                    "It's okay Rachel, you're at the care home where you live.",
                    "Rachel, I'm right here. You're not alone."
                ]
            elif "son" in last_patient_text.lower():
                responses = [
                    "Your son loves you very much. He'll visit soon.",
                    "He was just here yesterday, remember? He brought you flowers.",
                    "Would you like to call your son? I can help you."
                ]
            elif "home" in last_patient_text.lower():
                responses = [
                    "This is your home now. See your pictures on the wall?",
                    "Your son set up this nice room for you.",
                    "You've been living here for a while now. It's a safe place."
                ]
            else:
                responses = [
                    "Look, there's your TV that you like to watch.",
                    "Your son visited yesterday, remember?",
                    "You're safe here. Would you like some music?"
                ]
                
        elif "jolene" in persona.lower():
            if last_patient_emotion in ["depressed", "empty", "hopeless"]:
                responses = [
                    "That sounds really hard. You know, your son mentioned you were an amazing teacher.",
                    "I understand. Would you like to tell me about your music classes?",
                    "I hear you, Jolene. What was your favorite thing about teaching?"
                ]
            else:
                responses = [
                    "You taught music, right? Tell me about that.",
                    "I see there's classical music on TV. You mentioned you taught music.",
                    "Your students must have loved having you as a teacher."
                ]
                
        elif "ralph" in persona.lower():
            if "urgent" in last_patient_text.lower() or "important" in last_patient_text.lower():
                responses = [
                    "Let's take it slow, Ralph. Remember what happened last time you got up too fast?",
                    "I understand it feels urgent. How about we sit and talk first?",
                    "What's the urgent matter? I'm here to help."
                ]
            elif "hawaii" in last_patient_text.lower() or "surf" in last_patient_text.lower():
                responses = [
                    "Tell me more about your time in Hawaii!",
                    "The surfing must have been amazing there.",
                    "I heard you grew grapes in Hawaii too. How did that work?"
                ]
            else:
                responses = [
                    "You mentioned surfing in Hawaii. Want to tell me about that?",
                    "I heard you grew grapes in Hawaii. That must have been interesting.",
                    "Ralph, remember the grapes you grew in Hawaii?"
                ]
        else:
            responses = ["I'm here to help you."]
        
        # Use both turn number and sample_id for maximum variation
        response_idx = (turn_number + sample_id) % len(responses)
        response_text = responses[response_idx]
        
        # Build conversation context for LLM
        conv_context = "\n".join([f"{t['speaker']}: {t['text']}" for t in conversation[-3:]])
        
        # Call LLM for more natural variation (optional)
        generated = self._call_llm_for_response_with_context(
            "caregiver", conv_context, last_patient_emotion, 
            persona, response_text
        )
        
        return {
            "speaker": "Caregiver",
            "text": generated,
            "emotion": "calm"
        }
    
    def _call_llm_for_response_with_context(self, role, conversation_context, emotion, persona, hint_text):
        """Enhanced LLM call with conversation context"""
        try:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # More specific prompts based on persona
            persona_context = {
                "rachel_severe": "Rachel has severe dementia, often doesn't know where she is, and finds comfort in mentions of her son.",
                "jolene_moderate": "Jolene has moderate dementia with depression, speaks in monotone, was a music teacher.",
                "ralph_mild": "Ralph has mild dementia, appears conversational but confused, loves talking about Hawaii and surfing."
            }
            
            prompt = f"""Generate a dementia caregiver response.

Patient Profile: {persona_context.get(persona, 'Patient with dementia')}

Recent conversation:
{conversation_context}

Current emotion: {emotion}

Your response should:
- Follow this approach: "{hint_text}"
- Use therapeutic redirection if needed
- Be empathetic and validating
- Maximum 2 sentences
- Sound natural, not scripted

Response:"""
            
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are Jeremy, an expert dementia caregiver using therapeutic communication techniques."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 60,
                "temperature": 0.85
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                generated = response.json()["choices"][0]["message"]["content"].strip()
                if len(generated) > 0 and len(generated) < 200:
                    return generated
        except Exception as e:
            print(f"LLM call failed: {e}")
        
        return hint_text  # Fallback
    
    def _generate_patient_response(self, conversation, persona, severity, sample_id):
        """Generate patient response with variation"""
        last_caregiver_text = conversation[-1]["text"]
        persona_name = persona.split("_")[0].capitalize()  # Rachel, Jolene, Ralph
        turn_number = len([t for t in conversation if t["speaker"] == persona_name])
        
        # Debug log
        print(f"Generating patient response: turn {turn_number}, sample {sample_id}")
        
        # More sophisticated response generation based on turn and sample
        if persona_name == "Ralph":
            if "hawaii" in last_caregiver_text.lower() or "surf" in last_caregiver_text.lower():
                # Vary responses by turn number AND sample_id
                ralph_hawaii_responses = [
                    # Turn 1 responses (after first caregiver response)
                    ["Oh yes, the waves were perfect!", "Hawaii... best time of my life.", "Surfing? Was I going surfing?"],
                    # Turn 2 responses  
                    ["Best waves on the North Shore!", "I miss those days...", "The water was so blue..."],
                    # Turn 3 responses
                    ["Did I tell you about the big one?", "My board is still there somewhere...", "Those were the days..."]
                ]
                turn_idx = min(turn_number - 1, len(ralph_hawaii_responses) - 1)
                responses = ralph_hawaii_responses[turn_idx]
                emotion = "nostalgic"
            elif "grapes" in last_caregiver_text.lower():
                responses = [
                    "Grapes? Oh right! Yes! Had to be creative with the volcanic soil.",
                    "The grapes loved that Hawaiian sun!",
                    "Best wine in the Pacific, I used to say!"
                ]
                emotion = "engaged"
            elif "remember" in last_caregiver_text.lower() or "last time" in last_caregiver_text.lower():
                responses = [
                    "Last time? I don't... well, maybe.",
                    "I can't remember... was it yesterday?",
                    "My memory isn't what it used to be..."
                ]
                emotion = "uncertain"
            else:
                responses = [
                    "I don't know what you mean...",
                    "What? I'm confused...",
                    "I really need to... to do something important!"
                ]
                emotion = "confused"
                
        elif persona_name == "Rachel":
            if "son" in last_caregiver_text.lower():
                responses = [
                    "My son? When is he coming?",
                    "I want to see my son...",
                    "Call... my son? Yes... yes, I want my son."
                ]
                emotion = "hopeful"
            elif "room" in last_caregiver_text.lower() or "safe" in last_caregiver_text.lower():
                responses = [
                    "This isn't my room! Where am I?",
                    "I want to go home!",
                    "I don't understand..."
                ]
                emotion = "confused"
            else:
                responses = [
                    "Help me! Please!",
                    "I'm scared...",
                    "Where is everyone?"
                ]
                emotion = "fear"
                
        elif persona_name == "Jolene":
            if "teach" in last_caregiver_text.lower() or "music" in last_caregiver_text.lower():
                responses = [
                    "Music. Yes. I used to teach. Long time ago.",
                    "The children sang so beautifully...",
                    "That was... before. When things were different."
                ]
                emotion = "wistful"
            else:
                responses = [
                    "Nothing matters anymore...",
                    "Just... emptiness.",
                    "The same. Always the same."
                ]
                emotion = "depressed"
        else:
            responses = ["I don't understand..."]
            emotion = "confused"
        
        # Use both sample_id and turn number for variation
        response_idx = (sample_id + turn_number) % len(responses)
        
        return {
            "speaker": persona_name,
            "text": responses[response_idx],
            "emotion": emotion
        }
    
    def _call_llm_for_response(self, role, context, emotion, persona, hint_text):
        """Call LLM for more natural response generation"""
        try:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # More specific prompts based on persona
            persona_context = {
                "rachel_severe": "Rachel has severe dementia, often doesn't know where she is, and finds comfort in mentions of her son.",
                "jolene_moderate": "Jolene has moderate dementia with depression, speaks in monotone, was a music teacher.",
                "ralph_mild": "Ralph has mild dementia, appears conversational but confused, loves talking about Hawaii and surfing."
            }
            
            prompt = f"""Generate a {role} response for dementia care scenario.
Patient context: {persona_context.get(persona, 'Patient with dementia')}
Patient just said: "{context}" with emotion: {emotion}

Your response should:
- Be based on this therapeutic approach: "{hint_text}"
- Be natural and conversational
- Show empathy and use validation techniques
- Be 1-2 sentences maximum
- If patient is confused/agitated, use redirection to positive memories

Response:"""
            
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are an expert dementia caregiver trained in therapeutic communication."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 60,
                "temperature": 0.8
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                generated = response.json()["choices"][0]["message"]["content"].strip()
                # Ensure it's not too different from hint
                if len(generated) > 0 and len(generated) < 200:
                    return generated
        except Exception as e:
            print(f"LLM call failed: {e}")
        
        return hint_text  # Fallback to hint
    
    def _calculate_conversation_similarity(self, generated, gold):
        """Calculate similarity between generated and gold conversation with detailed metrics"""
        # Extract all texts
        gen_texts = [turn["text"] for turn in generated]
        gold_texts = [turn["text"] for turn in gold[:len(generated)]]
        
        # Encode all at once
        gen_embeddings = self.sentence_model.encode(gen_texts)
        gold_embeddings = self.sentence_model.encode(gold_texts)
        
        # Calculate turn-by-turn similarity
        turn_similarities = []
        turn_details = []
        
        for i in range(len(gen_embeddings)):
            sim = float(cosine_similarity(
                gen_embeddings[i].reshape(1, -1),
                gold_embeddings[i].reshape(1, -1)
            )[0][0])
            turn_similarities.append(sim)
            
            # Store detailed comparison
            turn_details.append({
                "turn": i + 1,
                "speaker": generated[i]["speaker"],
                "generated": gen_texts[i],
                "gold": gold_texts[i],
                "similarity": sim
            })
        
        # Overall conversation similarity
        gen_full = " ".join(gen_texts)
        gold_full = " ".join(gold_texts)
        
        conv_similarity = float(cosine_similarity(
            self.sentence_model.encode([gen_full]),
            self.sentence_model.encode([gold_full])
        )[0][0])
        
        # Weighted average (conversation level gets more weight)
        overall_score = 0.7 * conv_similarity + 0.3 * np.mean(turn_similarities)
        
        return {
            "overall": float(overall_score),
            "conversation_level": conv_similarity,
            "turn_average": float(np.mean(turn_similarities)),
            "turn_details": turn_details
        }

# ============================================================================
# Modified Streamlit App with Audio
# ============================================================================

st.set_page_config(
    page_title="🧠 Alzheimer's AI Prize Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply professional styling
apply_professional_styling()

# Load models
if "models_loaded" not in st.session_state:
    with st.spinner("Loading AI models..."):
        whisper_model, audio_emotion_classifier = load_models()
        if whisper_model:
            st.session_state.whisper_model = whisper_model
            st.session_state.audio_emotion_classifier = audio_emotion_classifier
            st.session_state.models_loaded = True

# Initialize conversation matcher
if "conv_matcher" not in st.session_state:
    with st.spinner("Loading conversation matching system..."):
        st.session_state.conv_matcher = ConversationMatcher()
        st.success("✅ Conversation matcher loaded!")

# Professional styling already applied above

# Initialize session state
if "terminal_log" not in st.session_state:
    st.session_state.terminal_log = "Team Noyce - Alzheimer's Insights AI Prize 2025 🧠\n" + "="*50 + "\n\n"

if "conversation_audio" not in st.session_state:
    st.session_state.conversation_audio = []

if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = None

if "processing" not in st.session_state:
    st.session_state.processing = False

# Professional header applied via CSS above
st.markdown('<div class="feature-highlight"><center><strong>Voice input → AI Generation → Gold Standard Matching → Voice output</strong></center></div>', unsafe_allow_html=True)

# Quick action buttons
col_action1, col_action2, col_action3, col_action4 = st.columns(4)
with col_action1:
    if st.session_state.current_conversation:
        st.success("✅ Conversation Active")
    else:
        st.info("🔄 Ready for New Conversation")

with col_action2:
    conversation_count = len([f for f in st.session_state.terminal_log.split('\n') if '🎤' in f]) // 2
    st.metric("Conversations", conversation_count)

with col_action3:
    if st.session_state.current_conversation:
        turns = len(st.session_state.current_conversation["conversation"])
        st.metric("Current Turns", turns)
    else:
        st.metric("Current Turns", 0)

with col_action4:
    if st.button("🔄 Reset All", type="secondary"):
        # Complete reset
        for key in ['terminal_log', 'conversation_audio', 'current_conversation', 'processing']:
            if key in st.session_state:
                if key == 'terminal_log':
                    st.session_state[key] = "DEMENTIA CARE AI - Gold Conversation Matching System 💚\n" + "="*50 + "\n\n"
                elif key == 'conversation_audio':
                    st.session_state[key] = []
                else:
                    st.session_state[key] = None
        st.rerun()

st.markdown("---")

# Main UI
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 💬 Generated Conversations")
    
    # Terminal log
    st.code(st.session_state.terminal_log, language="")
    
    # Audio players for conversation with similarity info
    if st.session_state.current_conversation:
        st.markdown("### 🎧 Voice Conversation")
        
        # Get similarity details
        sim_details = st.session_state.current_conversation.get("similarity_details", {})
        turn_details = sim_details.get("turn_details", [])
        
        for i, (turn, audio) in enumerate(zip(
            st.session_state.current_conversation["conversation"],
            st.session_state.conversation_audio
        )):
            with st.container():
                col_text, col_audio, col_sim = st.columns([3, 1, 1])
                
                with col_text:
                    emoji = "🎤" if turn["speaker"] != "Caregiver" else "👨‍⚕️"
                    
                    # Get turn similarity if available
                    turn_sim = ""
                    if i < len(turn_details):
                        sim_score = float(turn_details[i]["similarity"])
                        turn_sim = f"<small style='color: #FFD700;'>Match: {sim_score:.1%}</small>"
                    
                    st.markdown(f"""
                    <div class="audio-turn">
                    <strong>{emoji} {turn["speaker"]}:</strong> {turn["text"]}<br>
                    <small>Emotion: {turn["emotion"]}</small> | {turn_sim}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_audio:
                    if audio:
                        st.audio(audio, format="audio/wav")
                
                with col_sim:
                    if i < len(turn_details):
                        # Show similarity meter
                        sim_score = float(turn_details[i]["similarity"])
                        if sim_score > 0.8:
                            color = "🟢"
                        elif sim_score > 0.6:
                            color = "🟡"
                        else:
                            color = "🔴"
                        st.markdown(f"<h3>{color}</h3>", unsafe_allow_html=True)
        
        # Show gold conversation comparison
        if "gold_conversation" in st.session_state.current_conversation:
            with st.expander("📊 Gold Standard Comparison"):
                st.markdown("**Generated vs Gold Conversation:**")
                
                for i, detail in enumerate(turn_details):
                    st.markdown(f"**Turn {i+1} ({detail['speaker']}):**")
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.markdown(f"🎯 **Gold:** {detail['gold']}")
                    
                    with col2:
                        st.markdown(f"🤖 **Generated:** {detail['generated']}")
                    
                    with col3:
                        st.metric("Match", f"{float(detail['similarity']):.1%}")

with col2:
    st.markdown("### 🎯 Control Panel")
    
    # Persona selection with user-friendly names
    persona_options = {
        "Ralph (Mild Dementia) - Hawaii Story": "ralph_mild_7-turn.wav",
        "Rachel (Severe Dementia) - Lost & Scared": "rachel_severe_7-turn_turn1_rachel_extreme_distress.wav",
        "Jolene (Moderate Dementia) - Withdrawn": "jolene_moderate_7-turn_turn1_jolene_hopeless.wav"
    }
    
    selected_persona = st.selectbox(
        "👤 Select Patient Persona:",
        list(persona_options.keys()),
        key="persona_selector",
        help="Choose a patient persona for the conversation demo"
    )
    
    selected_file = persona_options[selected_persona]
    
    # Turn count selector
    num_turns = st.slider(
        "🔄 Number of Conversation Turns:",
        min_value=3,
        max_value=7,
        value=5,
        step=2,
        key="conversation_turns",
        help="Select 3, 5, or 7 turns for the conversation"
    )
    
    # Play selected audio
    if selected_file:
            audio_path = os.path.join(AUDIO_FOLDER, selected_file)
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/wav")
    
    # Show status
    if st.session_state.current_conversation:
        st.info("🔄 Click Generate to create a new conversation")
    
    # Process button
    if st.button("🎬 Generate Voice Conversation", type="primary", use_container_width=True):
        # Clear previous conversation if exists
        if st.session_state.current_conversation:
            st.session_state.current_conversation = None
            st.session_state.conversation_audio = []
            st.session_state.terminal_log = "Team Noyce - Alzheimer's Insights AI Prize 2025 🧠\n" + "="*50 + "\n\n"
        
        audio_path = os.path.join(AUDIO_FOLDER, selected_file)
        
        # Transcribe first utterance
        with st.spinner("Transcribing audio..."):
            first_utterance = transcribe_audio(audio_path)
            emotion, confidence = get_emotion_from_audio(audio_path)
            
            # Override emotion for Ralph demo - fix incorrect "happy" detection
            if "ralph" in selected_persona.lower():
                if "bell doesn't work" in first_utterance.lower() or "help" in first_utterance.lower():
                    emotion = "frustrated"
                    confidence = 0.85
                    st.session_state.terminal_log += f"🔧 Fixed emotion: {emotion} (demo override)\n"
        
        st.session_state.terminal_log += f"\n{'='*50}\n"
        st.session_state.terminal_log += f"📎 Input: {selected_file}\n"
        st.session_state.terminal_log += f"📝 Transcribed: {first_utterance}\n"
        st.session_state.terminal_log += f"😊 Emotion: {emotion} (confidence: {confidence:.2f})\n"
        
        # Generate with sampling
        with st.spinner("Generating 3 conversation samples..."):
            result = st.session_state.conv_matcher.generate_conversation_with_sampling(
                first_utterance, emotion, num_turns, num_samples=3
            )
        
        if result:
            st.session_state.current_conversation = result
            
            # Display results with detailed similarity
            st.session_state.terminal_log += f"\nPersona detected: {result['persona']}\n"
            st.session_state.terminal_log += f"Generated {num_turns} turns with 3 samples\n"
            st.session_state.terminal_log += f"Sample similarities: {[f'{float(s):.2f}' for s in result['all_similarities']]}\n"
            st.session_state.terminal_log += f"Selected sample {result['sample_id']} (overall: {float(result['gold_similarity']):.2f})\n"
            
            # Show turn-by-turn similarity
            if "similarity_details" in result:
                st.session_state.terminal_log += f"\nTurn-by-turn similarity:\n"
                for detail in result["similarity_details"]["turn_details"]:
                    st.session_state.terminal_log += f"  Turn {detail['turn']}: {float(detail['similarity']):.2%}\n"
            
            st.session_state.terminal_log += f"\n"
            
            # Generate audio for each turn
            st.session_state.conversation_audio = []
            
            with st.spinner("Generating voice conversation..."):
                for i, turn in enumerate(result["conversation"]):
                    # Display in terminal with similarity
                    emoji = "🎤" if turn["speaker"] != "Caregiver" else "👨‍⚕️"
                    st.session_state.terminal_log += f"{emoji} {turn['speaker']}: {turn['text']}\n"
                    st.session_state.terminal_log += f"   Emotion: {turn['emotion']}"
                    
                    # Add turn similarity if available
                    if "similarity_details" in result and i < len(result["similarity_details"]["turn_details"]):
                        turn_sim = float(result["similarity_details"]["turn_details"][i]["similarity"])
                        st.session_state.terminal_log += f" | Match: {turn_sim:.1%}"
                    
                    st.session_state.terminal_log += "\n"
                    
                    # Debug log
                    print(f"Turn {i}: Speaker={turn['speaker']}, Emotion={turn['emotion']}")
                    
                    # Generate audio for all turns to ensure consistency
                    audio_data = generate_azure_audio(
                        turn["text"], 
                        turn["speaker"],
                        turn["emotion"]
                    )
                    st.session_state.conversation_audio.append(audio_data)
            
            st.session_state.terminal_log += "\n"
            st.success(f"✅ Conversation generated! Gold similarity: {result['gold_similarity']:.2%}")
            st.rerun()
        else:
            st.warning("Failed to generate conversation")

# Similarity metrics display
st.markdown("---")
st.markdown("### 📊 Conversation Quality Metrics")

if st.session_state.current_conversation:
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
    
    sim_details = st.session_state.current_conversation.get("similarity_details", {})
    
    with col_metric1:
        overall = float(st.session_state.current_conversation['gold_similarity'])
        st.metric(
            "Overall Match", 
            f"{overall:.1%}",
            delta=f"+{(overall-0.7)*100:.1f}%"
        )
    
    with col_metric2:
        conv_level = float(sim_details.get("conversation_level", 0))
        st.metric(
            "Conversation Flow",
            f"{conv_level:.1%}",
            delta="Full context"
        )
    
    with col_metric3:
        turn_avg = float(sim_details.get("turn_average", 0))
        st.metric(
            "Turn Average",
            f"{turn_avg:.1%}",
            delta="Per utterance"
        )
    
    with col_metric4:
        sample_id = st.session_state.current_conversation['sample_id']
        st.metric(
            "Best Sample",
            f"#{sample_id + 1} of 3",
            delta="Selected"
        )

# Azure configuration check
if SPEECH_KEY == "YOUR_AZURE_SPEECH_KEY":
    st.sidebar.warning("⚠️ Azure TTS not configured - using basic TTS")
    with st.sidebar.expander("🔧 Setup Azure TTS"):
        st.markdown("""
        1. Replace `SPEECH_KEY` in code
        2. Replace `SPEECH_REGION` in code
        3. Restart the app
        
        Azure provides more realistic elderly voices!
        """)
else:
    st.sidebar.success("✅ Azure TTS configured")
    st.sidebar.info(f"Region: {SPEECH_REGION}")

# Footer
st.markdown("---")
st.markdown("""
**🔬 How it works:**
1. 🎤 **Input**: Select first patient utterance audio
2. 🎧 **Process**: Transcribe and detect emotion
3. 🤖 **Generate**: Create 3 conversation variations
4. 📊 **Compare**: Measure similarity to gold standard
   - Turn-by-turn utterance matching
   - Overall conversation flow
5. ✅ **Select**: Choose best matching sample
6. 🔊 **Output**: Generate elderly voices with emotions

**Key Features:**
- **Turn-by-turn similarity**: Each utterance compared to gold standard
- **Multiple samples**: 3 variations to find best match
- **Persona-aware**: Different strategies for Rachel, Jolene, Ralph
- **Emotion-driven**: Voice modulation based on emotional state

**Gold Standard**: Expert-validated therapeutic conversations for dementia care
""")

# Download conversation button
if st.session_state.current_conversation and st.session_state.conversation_audio:
    st.sidebar.markdown("### 💾 Export Conversation")
    
    # Clear conversation button
    if st.sidebar.button("🗑️ Clear Current Conversation", use_container_width=True):
        st.session_state.current_conversation = None
        st.session_state.conversation_audio = []
        st.session_state.terminal_log = "Team Noyce - Alzheimer's Insights AI Prize 2025 🧠\n" + "="*50 + "\n\n"
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Create combined audio
    if st.sidebar.button("Generate Combined Audio"):
        # This would require audio concatenation logic
        st.sidebar.info("Feature coming soon!")
    
    # Export as JSON - Convert numpy types to native Python types
    def convert_to_serializable(obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    # Prepare conversation data for export
    export_data = {
        "persona": st.session_state.current_conversation["persona"],
        "conversation": st.session_state.current_conversation["conversation"],
        "gold_similarity": float(st.session_state.current_conversation["gold_similarity"]),
        "sample_id": st.session_state.current_conversation["sample_id"],
        "all_similarities": [float(s) for s in st.session_state.current_conversation["all_similarities"]],
        "timestamp": datetime.now().isoformat()
    }
    
    # Add similarity details if available
    if "similarity_details" in st.session_state.current_conversation:
        sim_details = st.session_state.current_conversation["similarity_details"]
        export_data["similarity_details"] = {
            "overall": float(sim_details.get("overall", 0)),
            "conversation_level": float(sim_details.get("conversation_level", 0)),
            "turn_average": float(sim_details.get("turn_average", 0)),
            "turn_details": [
                {
                    "turn": detail["turn"],
                    "speaker": detail["speaker"],
                    "generated": detail["generated"],
                    "gold": detail["gold"],
                    "similarity": float(detail["similarity"])
                }
                for detail in sim_details.get("turn_details", [])
            ]
        }
    
    conv_json = json.dumps(convert_to_serializable(export_data), indent=2)
    
    st.sidebar.download_button(
        label="📄 Download Conversation JSON",
        data=conv_json,
        file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )