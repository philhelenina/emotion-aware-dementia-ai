#!/usr/bin/env python3
"""
Improved elderly voice settings for Azure TTS
"""
import azure.cognitiveservices.speech as speechsdk
import os

SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY", "")
SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "eastus")

def create_improved_elderly_ssml(text, speaker, emotion):
    """Improved SSML for more convincing elderly voices"""
    
    # Better elderly voice configurations
    ELDERLY_VOICES = {
        "ralph": {
            "voice": "en-US-BrandonNeural",  # More mature male voice
            "style": "default",  
            "pitch": "-25Hz",    # Lower pitch for elderly
            "rate": "-15%",      # Slower speech 
            "volume": "-5%",     # Slightly quieter
        },
        "ralph_alt": {
            "voice": "en-US-RyanNeural",  # Alternative mature voice
            "style": "default",
            "pitch": "-30Hz",
            "rate": "-20%", 
            "volume": "-3%"
        }
    }
    
    # More dramatic emotion adjustments for elderly
    emotion_adjustments = {
        "frustrated": {
            "pitch": "+15Hz", 
            "rate": "+10%", 
            "volume": "+15%",
            "extra_pauses": True
        },
        "nostalgic": {
            "pitch": "-5Hz", 
            "rate": "-25%", 
            "volume": "-10%",
            "extra_pauses": True
        },
        "engaged": {
            "pitch": "+5Hz", 
            "rate": "-10%", 
            "volume": "+5%",
            "extra_pauses": False
        }
    }
    
    voice_config = ELDERLY_VOICES.get(speaker, ELDERLY_VOICES["ralph"])
    emo_config = emotion_adjustments.get(emotion, {})
    
    def combine_values(base, modifier):
        if 'Hz' in base and 'Hz' in modifier:
            base_val = int(base.replace('Hz', ''))
            mod_val = int(modifier.replace('Hz', ''))
            return f"{base_val + mod_val}Hz"
        elif '%' in base and '%' in modifier:
            base_val = int(base.replace('%', ''))
            mod_val = int(modifier.replace('%', ''))
            final_val = max(-50, min(50, base_val + mod_val))
            return f"{final_val}%"
        return base
    
    final_pitch = combine_values(voice_config['pitch'], emo_config.get('pitch', '0Hz'))
    final_rate = combine_values(voice_config['rate'], emo_config.get('rate', '0%'))
    final_volume = combine_values(voice_config['volume'], emo_config.get('volume', '0%'))
    
    # Enhanced pauses for elderly speech patterns
    processed_text = text
    
    if emo_config.get('extra_pauses', False):
        processed_text = processed_text.replace("...", '<break time="800ms"/>...<break time="600ms"/>')
        processed_text = processed_text.replace("?", '?<break time="500ms"/>')
        processed_text = processed_text.replace("!", '!<break time="400ms"/>')
        processed_text = processed_text.replace(".", '.<break time="400ms"/>')
        processed_text = processed_text.replace(",", ',<break time="300ms"/>')
    else:
        processed_text = processed_text.replace("...", '<break time="400ms"/>...<break time="300ms"/>')
        processed_text = processed_text.replace("?", '?<break time="300ms"/>')
        processed_text = processed_text.replace("!", '!<break time="250ms"/>')
        processed_text = processed_text.replace(".", '.<break time="250ms"/>')
    
    # Add slight tremor effect for very elderly speech
    if emotion in ["nostalgic", "sad"]:
        processed_text = f'<prosody rate="{final_rate}" pitch="{final_pitch}" volume="{final_volume}" contour="(0%,-5st) (50%,+3st) (100%,-2st)">{processed_text}</prosody>'
        ssml = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" 
                    xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
            <voice name="{voice_config['voice']}">
                <mstts:express-as style="{voice_config.get('style', 'default')}">
                    {processed_text}
                </mstts:express-as>
            </voice>
        </speak>"""
    else:
        ssml = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" 
                    xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
            <voice name="{voice_config['voice']}">
                <mstts:express-as style="{voice_config.get('style', 'default')}">
                    <prosody pitch="{final_pitch}" 
                             rate="{final_rate}"
                             volume="{final_volume}">
                        {processed_text}
                    </prosody>
                </mstts:express-as>
            </voice>
        </speak>"""
    
    return ssml

def test_improved_voices():
    """Test improved elderly voice settings"""
    
    test_cases = [
        {
            "text": "Hello? HELLO? This bell doesn't work! I need help here! It's urgent!",
            "speaker": "ralph",
            "emotion": "frustrated",
            "label": "Ralph Frustrated (Original)"
        },
        {
            "text": "Hello? HELLO? This bell doesn't work! I need help here! It's urgent!",
            "speaker": "ralph_alt",
            "emotion": "frustrated", 
            "label": "Ralph Frustrated (Alt Voice)"
        },
        {
            "text": "Grapes? Oh right! Yes... Had to be creative with the volcanic soil. Say, you ever been to Hawaii?",
            "speaker": "ralph",
            "emotion": "nostalgic",
            "label": "Ralph Nostalgic (Original)"
        },
        {
            "text": "Grapes? Oh right! Yes... Had to be creative with the volcanic soil. Say, you ever been to Hawaii?",
            "speaker": "ralph_alt", 
            "emotion": "nostalgic",
            "label": "Ralph Nostalgic (Alt Voice)"
        }
    ]
    
    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION
    )
    
    for i, case in enumerate(test_cases):
        print(f"\n🎤 Testing: {case['label']}")
        
        try:
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config,
                audio_config=None
            )
            
            ssml = create_improved_elderly_ssml(
                case['text'], 
                case['speaker'], 
                case['emotion']
            )
            
            result = synthesizer.speak_ssml_async(ssml).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                filename = f"test_voice_{i+1}_{case['speaker']}_{case['emotion']}.wav"
                with open(filename, 'wb') as f:
                    f.write(result.audio_data)
                print(f"✅ Success! Saved: {filename}")
                print(f"   Audio size: {len(result.audio_data)} bytes")
            else:
                print(f"❌ Failed: {result.reason}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_improved_voices()