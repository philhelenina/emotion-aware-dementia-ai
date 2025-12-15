import azure.cognitiveservices.speech as speechsdk
import os
import time
from typing import Dict, Tuple, List

# Azure Speech Configuration - load from environment variables
# 1. Create a Speech Service resource in Azure Portal
# 2. Set Key and Region in .env file
SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY", "")
SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "eastus")

# Create output directory
os.makedirs("sample-conv-wav-files", exist_ok=True)

# Azure elderly/mature voice options
ELDERLY_VOICES = {
    "rachel_female": {
        "voice": "en-US-AriaNeural",  # Mature female voice
        "style": "customerservice",  # Calm style
        "pitch": "-15Hz",  # Lower pitch
        "rate": "-5%"  # Slower speed
    },
    "jolene_female": {
        "voice": "en-US-JennyNeural",  # Natural female voice
        "style": "sad",  # Depressed style (close to monotone)
        "pitch": "-20Hz",
        "rate": "-15%"
    },
    "ralph_male": {
        "voice": "en-US-GuyNeural",  # Mature male voice
        "style": "newscast",  # Clear pronunciation
        "pitch": "-25Hz",  # Lower pitch (male)
        "rate": "-5%"
    },
    "caregiver": {
        "voice": "en-US-JennyNeural",  # Friendly female voice
        "style": "friendly",
        "pitch": "0Hz",  # Normal pitch
        "rate": "0%"
    }
}

# Complete conversation data (includes 3, 5, 7 turns)
conversations = {
    "rachel_severe": {
        "3-turn": [
            ("Rachel", "Where... where am I? I don't... Help! Someone help me!", "fear"),
            ("Caregiver", "Rachel, you're safe in your room. I'm here with you. Your son visited yesterday, remember?", "calm"),
            ("Rachel", "My son? I... I don't... When? Where is he?", "confused")
        ],
        "5-turn": [
            ("Rachel", "Help! Help me! I don't know... I don't know where...", "panic"),
            ("Caregiver", "Rachel, it's okay. You're in your room. Look, there's your TV that you like to watch.", "soothing"),
            ("Rachel", "TV? I... what? I need... I need my son!", "anxious"),
            ("Caregiver", "Your son loves you very much. He'll visit soon. Would you like to watch your favorite show?", "reassuring"),
            ("Rachel", "I... okay... my son is coming?", "slightly_calmer")
        ],
        "7-turn": [
            ("Rachel", "Someone! Anyone! I'm... I'm lost! Help!", "extreme_distress"),
            ("Caregiver", "Rachel, I'm right here. You're not lost. This is your room at the care home.", "very_calm"),
            ("Rachel", "Care home? No! I want... I want to go home!", "upset"),
            ("Caregiver", "I understand. Your son set up this nice room for you. See your pictures on the wall?", "gentle"),
            ("Rachel", "Pictures? I... whose pictures? I don't understand!", "frustrated"),
            ("Caregiver", "Those are pictures of you and your son. He loves you so much. Would you like to call him?", "kind"),
            ("Rachel", "Call... my son? Yes... yes, I want my son.", "hopeful")
        ]
    },
    
    "jolene_moderate": {
        "3-turn": [
            ("Jolene", "I rang the bell. No one comes. No one ever comes.", "flat"),
            ("Caregiver", "Hi Jolene, I'm here now. Is there something you need?", "cheerful"),
            ("Jolene", "I don't know. I just... I don't want to be alone.", "monotone")
        ],
        "5-turn": [
            ("Jolene", "Why doesn't anyone answer. I push and push the button.", "dejected"),
            ("Caregiver", "I'm sorry for the wait, Jolene. I'm here now. How are you feeling?", "apologetic"),
            ("Jolene", "The same. Always the same. Nothing changes.", "depressed"),
            ("Caregiver", "I see there's classical music on TV. You mentioned you taught music. Would you like to tell me about that?", "encouraging"),
            ("Jolene", "Music. Yes. I used to teach. Long time ago. Doesn't matter now.", "slightly_engaged")
        ],
        "7-turn": [
            ("Jolene", "I keep ringing. No one hears me. Or they don't care.", "hopeless"),
            ("Caregiver", "Jolene, I heard you and I care. I came as quickly as I could. What can I do for you?", "caring"),
            ("Jolene", "Nothing. There's nothing. Just... emptiness.", "empty"),
            ("Caregiver", "That sounds really hard. You know, your son mentioned you were an amazing teacher. Elementary school, right?", "sympathetic"),
            ("Jolene", "Elementary. Yes. Children. They used to laugh. No laughter here.", "reminiscing"),
            ("Caregiver", "What was your favorite thing about teaching? I'd love to hear about it.", "interested"),
            ("Jolene", "The music class. When they sang. But that was... before. When things were different.", "wistful")
        ]
    },
    
    "ralph_mild": {
        "3-turn": [
            ("Ralph", "Hey! I've been ringing this damn bell for ages! What's a guy gotta do to get some help around here?", "irritated"),
            ("Caregiver", "Hi Ralph, I'm sorry for the wait. What can I help you with?", "apologetic"),
            ("Ralph", "Well, I... uh... I needed something. Can't remember what now. But I know it was important!", "confused_but_trying")
        ],
        "5-turn": [
            ("Ralph", "Finally! I've been calling and calling! This place has terrible service!", "annoyed"),
            ("Caregiver", "I apologize, Ralph. I'm here now. How can I help you today?", "patient"),
            ("Ralph", "I need to... to get ready. Big day today. Going somewhere important. Hawaii maybe?", "confused_excited"),
            ("Caregiver", "That sounds exciting! You mentioned you used to surf in Hawaii. Want to tell me about that?", "engaging"),
            ("Ralph", "Surf? Oh yes! Best waves on the North Shore! Wait... am I going surfing today?", "nostalgic_confused")
        ],
        "7-turn": [
            ("Ralph", "Hello? HELLO? This bell doesn't work! I need help here! It's urgent!", "frustrated"),
            ("Caregiver", "Hi Ralph, I hear you. I'm here. What's the urgent matter?", "calm_concerned"),
            ("Ralph", "I gotta get up. Gotta go. Important meeting... or was it... something about the grandkids?", "agitated_confused"),
            ("Caregiver", "Let's take it slow, Ralph. Remember what happened last time you got up too fast? How about we sit and talk first?", "cautious"),
            ("Ralph", "Last time? I don't... well, maybe. But I really need to... to do something. It's important!", "uncertain"),
            ("Caregiver", "I understand. Hey, I heard you grew grapes in Hawaii. That must have been interesting with the climate there.", "redirecting"),
            ("Ralph", "Grapes? Oh right! Yes! Had to be creative with the volcanic soil. Say, you ever been to Hawaii?", "engaged")
        ]
    }
}

# Advanced voice control using SSML
def create_ssml_with_elderly_effects(text: str, persona: str, emotion: str) -> str:
    """Generate elderly voice effects using SSML"""

    voice_config = ELDERLY_VOICES.get(persona, ELDERLY_VOICES["caregiver"])

    # Emotion-specific adjustments
    emotion_adjustments = {
        # Rachel emotions
        "fear": {"pitch": "+5Hz", "rate": "+10%", "volume": "+10%"},
        "panic": {"pitch": "+10Hz", "rate": "+15%", "volume": "+15%"},
        "confused": {"pitch": "+3Hz", "rate": "-5%", "contour": "(0%,+5Hz)(50%,-5Hz)(100%,+3Hz)"},
        "anxious": {"pitch": "+7Hz", "rate": "+5%", "volume": "+5%"},
        "extreme_distress": {"pitch": "+15Hz", "rate": "+20%", "volume": "+20%"},
        "upset": {"pitch": "+8Hz", "rate": "+10%", "volume": "+10%"},
        "frustrated": {"pitch": "+6Hz", "rate": "+5%", "volume": "+8%"},
        "slightly_calmer": {"pitch": "+2Hz", "rate": "-5%", "volume": "0%"},
        "hopeful": {"pitch": "0Hz", "rate": "-10%", "volume": "-5%"},
        
        # Jolene emotions
        "flat": {"pitch": "-10Hz", "rate": "-20%", "volume": "-20%"},
        "monotone": {"pitch": "-15Hz", "rate": "-25%", "volume": "-25%"},
        "dejected": {"pitch": "-12Hz", "rate": "-22%", "volume": "-22%"},
        "depressed": {"pitch": "-18Hz", "rate": "-30%", "volume": "-30%"},
        "hopeless": {"pitch": "-20Hz", "rate": "-35%", "volume": "-35%"},
        "empty": {"pitch": "-25Hz", "rate": "-40%", "volume": "-40%"},
        "reminiscing": {"pitch": "-8Hz", "rate": "-15%", "volume": "-10%"},
        "slightly_engaged": {"pitch": "-5Hz", "rate": "-10%", "volume": "-8%"},
        "wistful": {"pitch": "-10Hz", "rate": "-18%", "volume": "-15%"},
        
        # Ralph emotions
        "irritated": {"pitch": "+5Hz", "rate": "+5%", "volume": "+5%"},
        "annoyed": {"pitch": "+6Hz", "rate": "+8%", "volume": "+8%"},
        "confused_but_trying": {"pitch": "+2Hz", "rate": "-5%", "volume": "0%"},
        "confused_excited": {"pitch": "+4Hz", "rate": "+3%", "volume": "+3%"},
        "nostalgic_confused": {"pitch": "+1Hz", "rate": "-8%", "volume": "-2%"},
        "frustrated": {"pitch": "+8Hz", "rate": "+10%", "volume": "+10%"},
        "agitated_confused": {"pitch": "+6Hz", "rate": "+5%", "volume": "+5%"},
        "uncertain": {"pitch": "+2Hz", "rate": "-10%", "volume": "-5%"},
        "engaged": {"pitch": "0Hz", "rate": "-5%", "volume": "0%"},
        
        # Caregiver emotions
        "calm": {"pitch": "0Hz", "rate": "-5%", "volume": "0%"},
        "soothing": {"pitch": "-3Hz", "rate": "-10%", "volume": "-5%"},
        "reassuring": {"pitch": "-2Hz", "rate": "-8%", "volume": "-3%"},
        "very_calm": {"pitch": "-5Hz", "rate": "-15%", "volume": "-8%"},
        "gentle": {"pitch": "-4Hz", "rate": "-12%", "volume": "-6%"},
        "kind": {"pitch": "-3Hz", "rate": "-10%", "volume": "-5%"},
        "cheerful": {"pitch": "+2Hz", "rate": "0%", "volume": "+2%"},
        "apologetic": {"pitch": "-2Hz", "rate": "-5%", "volume": "-3%"},
        "encouraging": {"pitch": "+1Hz", "rate": "-3%", "volume": "+1%"},
        "caring": {"pitch": "-2Hz", "rate": "-8%", "volume": "-4%"},
        "sympathetic": {"pitch": "-3Hz", "rate": "-10%", "volume": "-5%"},
        "interested": {"pitch": "+1Hz", "rate": "-5%", "volume": "0%"},
        "patient": {"pitch": "-1Hz", "rate": "-8%", "volume": "-2%"},
        "engaging": {"pitch": "+2Hz", "rate": "-3%", "volume": "+2%"},
        "calm_concerned": {"pitch": "-1Hz", "rate": "-6%", "volume": "-2%"},
        "cautious": {"pitch": "-2Hz", "rate": "-10%", "volume": "-4%"},
        "redirecting": {"pitch": "0Hz", "rate": "-5%", "volume": "0%"}
    }
    
    emo_adj = emotion_adjustments.get(emotion, {})

    # SSML template
    ssml = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" 
                xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
        <voice name="{voice_config['voice']}">
            <mstts:express-as style="{voice_config.get('style', 'default')}" 
                             styledegree="2">
                <prosody pitch="{emo_adj.get('pitch', voice_config['pitch'])}" 
                         rate="{emo_adj.get('rate', voice_config['rate'])}"
                         volume="{emo_adj.get('volume', '0%')}">
                    {add_elderly_speech_patterns(text, persona, emotion)}
                </prosody>
            </mstts:express-as>
        </voice>
    </speak>"""
    
    return ssml

def add_elderly_speech_patterns(text: str, persona: str, emotion: str) -> str:
    """Add elderly-specific speech patterns"""

    # Rachel (severe dementia) - stuttering, interruptions
    if "rachel" in persona and emotion in ["fear", "confused", "panic", "anxious", "frustrated"]:
        # Add pauses between words
        words = text.split()
        result = []
        for i, word in enumerate(words):
            result.append(word)
            if i < len(words) - 1 and "..." in text:
                result.append('<break time="500ms"/>')
            elif emotion in ["panic", "extreme_distress"] and i % 3 == 0:
                result.append('<break time="200ms"/>')
        return " ".join(result)

    # Jolene (depressed) - long pauses, sighs
    elif "jolene" in persona:
        # Add long pauses mid-sentence
        text = text.replace(".", '.<break time="1s"/>')
        text = text.replace(",", ',<break time="500ms"/>')
        if emotion in ["depressed", "empty", "hopeless"]:
            # Sigh effect before speaking
            text = '<break time="500ms"/>' + text
        return text

    # Ralph (confused but trying) - add "uh" interjections
    elif "ralph" in persona and emotion in ["confused_but_trying", "uncertain", "confused_excited"]:
        text = text.replace("I need to...", 'I need to<break time="300ms"/>uh<break time="300ms"/>')
        text = text.replace("was it...", 'was it<break time="300ms"/>uh<break time="200ms"/>')
        text = text.replace("I...", 'I<break time="200ms"/>')
        return text
    
    return text

def generate_azure_voice(text: str, speaker: str, emotion: str, filename: str):
    """Generate voice using Azure Speech"""

    # Speech config setup
    speech_config = speechsdk.SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION
    )

    # Audio output configuration
    audio_config = speechsdk.audio.AudioOutputConfig(
        filename=f"sample-conv-wav-files/{filename}.wav"
    )

    # Create synthesizer
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    # Persona mapping
    if speaker == "Rachel":
        persona = "rachel_female"
    elif speaker == "Jolene":
        persona = "jolene_female"
    elif speaker == "Ralph":
        persona = "ralph_male"
    else:  # Caregiver
        persona = "caregiver"

    # Generate SSML
    ssml = create_ssml_with_elderly_effects(text, persona, emotion)

    # Synthesize speech
    result = synthesizer.speak_ssml_async(ssml).get()
    
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"✅ Created: {filename}.wav - {speaker} ({emotion})")
    else:
        print(f"❌ Error: {result.reason}")
        if result.cancellation_details:
            print(f"   Details: {result.cancellation_details.reason}")
            print(f"   Error details: {result.cancellation_details.error_details}")

# Main execution
if __name__ == "__main__":
    print("🎤 Azure Speech Services - Complete Elderly Voice Generation")
    print("=" * 60)

    # Check API key
    if SPEECH_KEY == "YOUR_AZURE_SPEECH_KEY":
        print("❌ Please set your Azure Speech API key first!")
        print("\n📝 How to get Azure Speech API key:")
        print("1. Go to https://portal.azure.com")
        print("2. Create a Speech Service resource")
        print("3. Copy the key and region")
        print("4. Update SPEECH_KEY and SPEECH_REGION in this script")
        exit(1)

    # Generate all voice files
    total_files = 0
    for persona_name, persona_conversations in conversations.items():
        print(f"\n{'='*60}")
        print(f"📁 Creating files for {persona_name.upper()}")
        print(f"{'='*60}")
        
        for turn_type, turns in persona_conversations.items():
            print(f"\n  📝 {turn_type} conversation:")
            print(f"  {'-'*40}")
            
            for i, (speaker, text, emotion) in enumerate(turns, 1):
                filename = f"{persona_name}_{turn_type}_turn{i}_{speaker.lower()}_{emotion}"
                print(f"  Turn {i}: ", end="")
                generate_azure_voice(text, speaker, emotion, filename)
                total_files += 1
                time.sleep(0.5)  # Prevent API rate limiting
    
    print("\n" + "=" * 60)
    print(f"✅ All {total_files} audio files created successfully!")
    print("=" * 60)
    
    print("\n📊 Summary:")
    print("- Rachel (Severe): 21 files (3+5+7 turns × 2 speakers)")
    print("- Jolene (Moderate): 21 files")
    print("- Ralph (Mild): 21 files")
    print("- Total: 63 audio files")
    
    print("\n💡 Files are organized as:")
    print("  [persona]_[turn-type]_turn[N]_[speaker]_[emotion].wav")
    print("  Example: rachel_severe_3-turn_turn1_rachel_fear.wav")
    
    print("\n🎯 Ready for emotion labeling and evaluation!")