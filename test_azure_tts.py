#!/usr/bin/env python3
"""
Quick test script to check Azure TTS elderly voice quality
"""
import azure.cognitiveservices.speech as speechsdk
import io
import os

SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY", "")
SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "eastus")

def create_azure_ssml(text, speaker, emotion):
    """Create SSML for Azure TTS - simplified version"""
    
    # Ralph voice config from original code
    voice_config = {
        "voice": "en-US-GuyNeural",
        "style": "default",  
        "pitch": "-15Hz",
        "rate": "-5%",
        "volume": "0%"
    }
    
    # Add emotional adjustments
    emotion_mods = {
        "frustrated": {"pitch": "+6Hz", "rate": "+8%", "volume": "+12%"},
        "nostalgic": {"pitch": "+1Hz", "rate": "-10%", "volume": "-2%"},
        "engaged": {"pitch": "+2Hz", "rate": "-5%", "volume": "+3%"}
    }
    
    emo_mod = emotion_mods.get(emotion, {})
    
    def combine_prosody(base, mod):
        if 'Hz' in base and 'Hz' in mod:
            base_val = int(base.replace('Hz', ''))
            mod_val = int(mod.replace('Hz', ''))
            return f"{base_val + mod_val}Hz"
        elif '%' in base and '%' in mod:
            base_val = int(base.replace('%', ''))
            mod_val = int(mod.replace('%', ''))
            final_val = max(-50, min(50, base_val + mod_val))
            return f"{final_val}%"
        return base
    
    final_pitch = combine_prosody(voice_config['pitch'], emo_mod.get('pitch', '0Hz'))
    final_rate = combine_prosody(voice_config['rate'], emo_mod.get('rate', '0%'))
    final_volume = combine_prosody(voice_config.get('volume', '0%'), emo_mod.get('volume', '0%'))
    
    # Add pauses for elderly speech
    text_with_pauses = text.replace("...", '<break time="600ms"/>...<break time="400ms"/>')
    text_with_pauses = text_with_pauses.replace("?", '?<break time="400ms"/>')
    text_with_pauses = text_with_pauses.replace(".", '.<break time="300ms"/>')
    
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

def test_azure_tts(text, speaker, emotion):
    """Test Azure TTS with elderly voice"""
    print(f"🧪 Testing Azure TTS...")
    print(f"   Speaker: {speaker}, Emotion: {emotion}")
    print(f"   Text: '{text[:50]}...'")
    
    try:
        speech_config = speechsdk.SpeechConfig(
            subscription=SPEECH_KEY,
            region=SPEECH_REGION
        )
        
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=None
        )
        
        ssml = create_azure_ssml(text, speaker, emotion)
        print(f"📝 SSML generated: {len(ssml)} characters")
        
        result = synthesizer.speak_ssml_async(ssml).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print(f"✅ Azure TTS successful!")
            print(f"   Audio data size: {len(result.audio_data)} bytes")
            
            # Save to file for testing
            with open('test_output.wav', 'wb') as f:
                f.write(result.audio_data)
            print(f"   Saved to: test_output.wav")
            return result.audio_data
            
        else:
            print(f"❌ Azure TTS failed: {result.reason}")
            if result.reason == speechsdk.ResultReason.Canceled:
                cancellation = result.cancellation_details
                print(f"   Cancellation reason: {cancellation.reason}")
                print(f"   Error details: {cancellation.error_details}")
                print(f"   Error code: {cancellation.error_code}")
        
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None
    
    return None

if __name__ == "__main__":
    # Test with Ralph's frustrated voice
    test_text = "Hello? HELLO? This bell doesn't work! I need help here! It's urgent!"
    test_azure_tts(test_text, "Ralph", "frustrated")
    
    # Test with Ralph's nostalgic voice
    test_text2 = "Grapes? Oh right! Yes! Had to be creative with the volcanic soil. Say, you ever been to Hawaii?"
    test_azure_tts(test_text2, "Ralph", "nostalgic")