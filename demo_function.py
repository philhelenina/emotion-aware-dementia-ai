#!/usr/bin/env python3
"""
Robot Audio Processor - Command Line Interface
Usage: python demo.py <audio_file_path>
Returns: Empathetic response with emotion detection
"""

import argparse
import sys
import json
import numpy as np
import torch
import librosa
import whisper
from transformers import pipeline
import os
import requests
from datetime import datetime
import re
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# Configuration
# ============================================================================

# OpenAI API Key (Replace with your actual key or use environment variable)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-your-openai-api-key-here")

# Model cache directory (to avoid reloading models every time)
MODEL_CACHE_DIR = "./model_cache"

# ============================================================================
# Text Cleaning Functions
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
# Model Loading
# ============================================================================

def load_whisper_model():
    """Load Whisper model"""
    try:
        model = whisper.load_model("base")
        return model
    except Exception as e:
        print(f"ERROR: Failed to load Whisper model: {e}", file=sys.stderr)
        return None

def load_emotion_classifier():
    """Load emotion classifier"""
    try:
        classifier = pipeline(
            "audio-classification", 
            model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
            return_all_scores=True
        )
        return classifier
    except Exception as e:
        print(f"WARNING: Failed to load emotion classifier: {e}", file=sys.stderr)
        return None

# ============================================================================
# Audio Processing Functions
# ============================================================================

def transcribe_audio(audio_path, whisper_model):
    """Convert audio to text and clean for robot use"""
    try:
        result = whisper_model.transcribe(audio_path)
        raw_text = result["text"].strip()
        cleaned_text = clean_text_for_robot(raw_text)
        return cleaned_text
    except Exception as e:
        print(f"ERROR: Speech recognition failed: {e}", file=sys.stderr)
        return None

def get_emotion_from_audio(audio_path, emotion_classifier):
    """Analyze emotion from audio"""
    if not emotion_classifier:
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
        predictions = emotion_classifier({"raw": y, "sampling_rate": sr})
        
        if predictions and len(predictions) > 0:
            top_emotion = max(predictions, key=lambda x: x['score'])
            return top_emotion['label'], top_emotion['score']
        
    except Exception as e:
        print(f"WARNING: Emotion analysis failed: {e}", file=sys.stderr)
    
    return "neutral", 0.5

# ============================================================================
# Response Generation
# ============================================================================

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
            clean_ai_text = clean_text_for_robot(ai_text)
            return clean_ai_text
            
    except Exception as e:
        print(f"WARNING: LLM response failed: {e}", file=sys.stderr)
    
    return clean_text_for_robot(generate_simple_response(emotion))

# ============================================================================
# Main Processing Function
# ============================================================================

def process_audio_file(audio_path, verbose=False):
    """
    Process a single audio file and return the empathetic response
    
    This function performs:
    1. Speech-to-Text (Whisper)
    2. Emotion Recognition (wav2vec2-based emotion classifier)
    3. Empathetic Response Generation (based on text + emotion)
    
    Args:
        audio_path: Path to the audio file
        verbose: If True, print debug information
    
    Returns:
        dict with transcribed_text, emotion, confidence, response_text
    """
    
    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found: {audio_path}", file=sys.stderr)
        return None
    
    if os.path.isdir(audio_path):
        print(f"ERROR: '{audio_path}' is a directory, not an audio file!", file=sys.stderr)
        print(f"Usage: python {sys.argv[0]} <audio_file_path>", file=sys.stderr)
        print(f"Example: python {sys.argv[0]} {audio_path}/sample.wav", file=sys.stderr)
        return None
    
    if verbose:
        print(f"Processing: {audio_path}", file=sys.stderr)
    
    if verbose:
        print("Loading models...", file=sys.stderr)
    
    whisper_model = load_whisper_model()
    if not whisper_model:
        return None
    
    emotion_classifier = load_emotion_classifier()
    
    # Step 1: Transcribe audio (Speech-to-Text)
    if verbose:
        print("Step 1: Transcribing audio (Speech-to-Text)...", file=sys.stderr)
    
    transcribed_text = transcribe_audio(audio_path, whisper_model)
    if not transcribed_text:
        return None
    
    if verbose:
        print(f"   Transcribed: {transcribed_text}", file=sys.stderr)
    
    # Step 2: Detect emotion from audio (Emotion Recognition)
    if verbose:
        print("Step 2: Detecting emotion from audio...", file=sys.stderr)
    
    emotion, confidence = get_emotion_from_audio(audio_path, emotion_classifier)
    
    if verbose:
        print(f"   Detected emotion: {emotion} (confidence: {confidence:.2f})", file=sys.stderr)
    
    # Step 3: Generate empathetic response based on text + emotion
    if verbose:
        print("Step 3: Generating empathetic response...", file=sys.stderr)
    
    response_text = generate_llm_response(transcribed_text, emotion, confidence)
    
    if verbose:
        print(f"   Response: {response_text}", file=sys.stderr)
    
    # Return complete result
    result = {
        "transcribed_text": transcribed_text,
        "emotion": emotion,
        "confidence": confidence,
        "response_text": response_text
    }
    
    return result

# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Process robot audio and generate empathetic response"
    )
    
    parser.add_argument(
        "audio_file",
        help="Path to the audio file to process"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print verbose debug information to stderr"
    )
    
    parser.add_argument(
        "-j", "--json",
        action="store_true",
        help="Output result as JSON"
    )
    
    parser.add_argument(
        "-s", "--simple",
        action="store_true",
        help="Output only the response text (for server integration)"
    )
    
    args = parser.parse_args()
    
    # Process the audio file
    result = process_audio_file(args.audio_file, verbose=args.verbose)
    
    if not result:
        sys.exit(1)
    
    # Output the result
    if args.json:
        print(json.dumps(result))
    elif args.simple:
        print(result["response_text"])
    else:
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Input text
        print(f"[{timestamp}] You: {result['transcribed_text']}")
        print(f"           Detected emotion: {result['emotion'].upper()}")
        print()
        
        print(f"[{timestamp}] AI: {result['response_text']}")

if __name__ == "__main__":
    main()