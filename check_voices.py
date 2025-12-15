#!/usr/bin/env python3
"""
Check available Azure voices and test elderly-sounding ones
"""
import azure.cognitiveservices.speech as speechsdk
import os

SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY", "")
SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "eastus")

def test_voice_options():
    """Test different male voices that might sound more elderly"""
    
    # Test these male voices for elderly sound
    male_voices_to_test = [
        "en-US-GuyNeural",      # Original
        "en-US-DavisNeural",    # Mature male
        "en-US-JasonNeural",    # Adult male
        "en-US-TonyNeural",     # Adult male
        "en-US-ChristopherNeural", # Adult male
    ]
    
    test_text = "Grapes? Oh right! Yes... Had to be creative with the volcanic soil. Those were the days."
    
    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION
    )
    
    for voice_name in male_voices_to_test:
        print(f"\n🎤 Testing voice: {voice_name}")
        
        try:
            # Create SSML with elderly adjustments
            ssml = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" 
                        xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
                <voice name="{voice_name}">
                    <prosody pitch="-25Hz" rate="-20%" volume="-5%">
                        {test_text}<break time="400ms"/>
                    </prosody>
                </voice>
            </speak>"""
            
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config,
                audio_config=None
            )
            
            result = synthesizer.speak_ssml_async(ssml).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                filename = f"voice_test_{voice_name.replace('-', '_').replace('Neural', '')}.wav"
                with open(filename, 'wb') as f:
                    f.write(result.audio_data)
                print(f"✅ Success! Saved: {filename}")
                print(f"   Audio size: {len(result.audio_data)} bytes")
            else:
                print(f"❌ Failed: {result.reason}")
                if result.reason == speechsdk.ResultReason.Canceled:
                    cancellation = result.cancellation_details
                    print(f"   Error: {cancellation.error_details}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")

def create_optimized_ralph_ssml(text, emotion):
    """Create the best elderly voice SSML based on testing"""
    
    # Use the most mature-sounding voice available
    voice_name = "en-US-DavisNeural"  # Try this first, fallback to Guy
    
    emotion_configs = {
        "frustrated": {
            "pitch": "-20Hz",  # Not too extreme
            "rate": "-10%",    # Slightly slower when agitated  
            "volume": "+8%",   # Louder when frustrated
            "style": "angry"   # If supported
        },
        "nostalgic": {
            "pitch": "-30Hz",  # Lower for nostalgic/sad tone
            "rate": "-25%",    # Much slower, reminiscing 
            "volume": "-8%",   # Quieter, more intimate
            "style": "sad"     # If supported
        },
        "engaged": {
            "pitch": "-20Hz",  # Still elderly but more energetic
            "rate": "-15%",    # Moderately slow
            "volume": "0%",    # Normal volume
            "style": "friendly"
        }
    }
    
    config = emotion_configs.get(emotion, emotion_configs["engaged"])
    
    # Add realistic elderly speech patterns
    processed_text = text
    if emotion == "nostalgic":
        # Add more pauses and hesitation
        processed_text = processed_text.replace("...", '<break time="800ms"/>...<break time="600ms"/>')
        processed_text = processed_text.replace("?", '?<break time="500ms"/>')
        processed_text = processed_text.replace(".", '.<break time="400ms"/>')
        processed_text = processed_text.replace(",", ',<break time="300ms"/>')
    elif emotion == "frustrated":
        # Shorter, more urgent pauses
        processed_text = processed_text.replace("!", '!<break time="200ms"/>')
        processed_text = processed_text.replace("?", '?<break time="300ms"/>')
    
    ssml = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" 
                xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
        <voice name="{voice_name}">
            <mstts:express-as style="{config.get('style', 'default')}">
                <prosody pitch="{config['pitch']}" 
                         rate="{config['rate']}"
                         volume="{config['volume']}">
                    {processed_text}
                </prosody>
            </mstts:express-as>
        </voice>
    </speak>"""
    
    return ssml

if __name__ == "__main__":
    test_voice_options()
    
    # Test optimized version
    print("\n" + "="*50)
    print("🎯 Testing OPTIMIZED elderly voice:")
    
    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION
    )
    
    test_cases = [
        ("Hello? HELLO? This bell doesn't work! I need help here!", "frustrated"),
        ("Grapes? Oh right! Yes... Had to be creative with the volcanic soil. Those were the days.", "nostalgic")
    ]
    
    for text, emotion in test_cases:
        try:
            ssml = create_optimized_ralph_ssml(text, emotion)
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
            result = synthesizer.speak_ssml_async(ssml).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                filename = f"optimized_ralph_{emotion}.wav"
                with open(filename, 'wb') as f:
                    f.write(result.audio_data)
                print(f"✅ {emotion.title()}: {filename}")
            else:
                print(f"❌ {emotion.title()}: Failed")
        except Exception as e:
            print(f"❌ {emotion.title()}: Error - {e}")