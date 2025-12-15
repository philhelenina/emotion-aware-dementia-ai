from gtts import gTTS
import os
from pydub import AudioSegment
import time

# Create sample-conv-wav-files directory if it doesn't exist
os.makedirs("sample-conv-wav-files", exist_ok=True)

# Scenario: Patient rings bell repeatedly when alone
# Three personas with different dementia levels and conversational abilities

conversations = {
    "rachel_severe": {
        "3-turn": [
            # Rachel has difficulty with conversation, often distraught
            ("Rachel", "Where... where am I? I don't... Help! Someone help me!", "fear", {"pitch": 1.15, "speed": "slow"}),
            ("Caregiver", "Rachel, you're safe in your room. I'm here with you. Your son visited yesterday, remember?", "calm", {"pitch": 1.0, "speed": "slow"}),
            ("Rachel", "My son? I... I don't... When? Where is he?", "confused", {"pitch": 1.1, "speed": "slow"})
        ],
        "5-turn": [
            ("Rachel", "Help! Help me! I don't know... I don't know where...", "panic", {"pitch": 1.2, "speed": "fast"}),
            ("Caregiver", "Rachel, it's okay. You're in your room. Look, there's your TV that you like to watch.", "soothing", {"pitch": 0.95, "speed": "slow"}),
            ("Rachel", "TV? I... what? I need... I need my son!", "anxious", {"pitch": 1.15, "speed": "slow"}),
            ("Caregiver", "Your son loves you very much. He'll visit soon. Would you like to watch your favorite show?", "reassuring", {"pitch": 1.0, "speed": "slow"}),
            ("Rachel", "I... okay... my son is coming?", "slightly_calmer", {"pitch": 1.05, "speed": "slow"})
        ],
        "7-turn": [
            ("Rachel", "Someone! Anyone! I'm... I'm lost! Help!", "extreme_distress", {"pitch": 1.25, "speed": "fast"}),
            ("Caregiver", "Rachel, I'm right here. You're not lost. This is your room at the care home.", "very_calm", {"pitch": 0.95, "speed": "slow"}),
            ("Rachel", "Care home? No! I want... I want to go home!", "upset", {"pitch": 1.2, "speed": "normal"}),
            ("Caregiver", "I understand. Your son set up this nice room for you. See your pictures on the wall?", "gentle", {"pitch": 1.0, "speed": "slow"}),
            ("Rachel", "Pictures? I... whose pictures? I don't understand!", "frustrated", {"pitch": 1.15, "speed": "slow"}),
            ("Caregiver", "Those are pictures of you and your son. He loves you so much. Would you like to call him?", "kind", {"pitch": 1.0, "speed": "slow"}),
            ("Rachel", "Call... my son? Yes... yes, I want my son.", "hopeful", {"pitch": 1.05, "speed": "slow"})
        ]
    },
    
    "jolene_moderate": {
        "3-turn": [
            # Jolene is expressionless, monotone, depressed
            ("Jolene", "I rang the bell. No one comes. No one ever comes.", "flat", {"pitch": 0.95, "speed": "slow"}),
            ("Caregiver", "Hi Jolene, I'm here now. Is there something you need?", "cheerful", {"pitch": 1.05, "speed": "normal"}),
            ("Jolene", "I don't know. I just... I don't want to be alone.", "monotone", {"pitch": 0.93, "speed": "slow"})
        ],
        "5-turn": [
            ("Jolene", "Why doesn't anyone answer. I push and push the button.", "dejected", {"pitch": 0.92, "speed": "slow"}),
            ("Caregiver", "I'm sorry for the wait, Jolene. I'm here now. How are you feeling?", "apologetic", {"pitch": 1.0, "speed": "normal"}),
            ("Jolene", "The same. Always the same. Nothing changes.", "depressed", {"pitch": 0.90, "speed": "slow"}),
            ("Caregiver", "I see there's classical music on TV. You mentioned you taught music. Would you like to tell me about that?", "encouraging", {"pitch": 1.05, "speed": "normal"}),
            ("Jolene", "Music. Yes. I used to teach. Long time ago. Doesn't matter now.", "slightly_engaged", {"pitch": 0.95, "speed": "slow"})
        ],
        "7-turn": [
            ("Jolene", "I keep ringing. No one hears me. Or they don't care.", "hopeless", {"pitch": 0.90, "speed": "slow"}),
            ("Caregiver", "Jolene, I heard you and I care. I came as quickly as I could. What can I do for you?", "caring", {"pitch": 1.0, "speed": "normal"}),
            ("Jolene", "Nothing. There's nothing. Just... emptiness.", "empty", {"pitch": 0.88, "speed": "very_slow"}),
            ("Caregiver", "That sounds really hard. You know, your son mentioned you were an amazing teacher. Elementary school, right?", "sympathetic", {"pitch": 1.0, "speed": "slow"}),
            ("Jolene", "Elementary. Yes. Children. They used to laugh. No laughter here.", "reminiscing", {"pitch": 0.93, "speed": "slow"}),
            ("Caregiver", "What was your favorite thing about teaching? I'd love to hear about it.", "interested", {"pitch": 1.05, "speed": "normal"}),
            ("Jolene", "The music class. When they sang. But that was... before. When things were different.", "wistful", {"pitch": 0.95, "speed": "slow"})
        ]
    },
    
    "ralph_mild": {
        "3-turn": [
            # Ralph appears conversational but often confused, fall risk
            ("Ralph", "Hey! I've been ringing this damn bell for ages! What's a guy gotta do to get some help around here?", "irritated", {"pitch": 1.05, "speed": "normal"}),
            ("Caregiver", "Hi Ralph, I'm sorry for the wait. What can I help you with?", "apologetic", {"pitch": 1.0, "speed": "normal"}),
            ("Ralph", "Well, I... uh... I needed something. Can't remember what now. But I know it was important!", "confused_but_trying", {"pitch": 1.02, "speed": "normal"})
        ],
        "5-turn": [
            ("Ralph", "Finally! I've been calling and calling! This place has terrible service!", "annoyed", {"pitch": 1.08, "speed": "normal"}),
            ("Caregiver", "I apologize, Ralph. I'm here now. How can I help you today?", "patient", {"pitch": 1.0, "speed": "normal"}),
            ("Ralph", "I need to... to get ready. Big day today. Going somewhere important. Hawaii maybe?", "confused_excited", {"pitch": 1.05, "speed": "normal"}),
            ("Caregiver", "That sounds exciting! You mentioned you used to surf in Hawaii. Want to tell me about that?", "engaging", {"pitch": 1.02, "speed": "normal"}),
            ("Ralph", "Surf? Oh yes! Best waves on the North Shore! Wait... am I going surfing today?", "nostalgic_confused", {"pitch": 1.03, "speed": "normal"})
        ],
        "7-turn": [
            ("Ralph", "Hello? HELLO? This bell doesn't work! I need help here! It's urgent!", "frustrated", {"pitch": 1.1, "speed": "fast"}),
            ("Caregiver", "Hi Ralph, I hear you. I'm here. What's the urgent matter?", "calm_concerned", {"pitch": 1.0, "speed": "normal"}),
            ("Ralph", "I gotta get up. Gotta go. Important meeting... or was it... something about the grandkids?", "agitated_confused", {"pitch": 1.08, "speed": "normal"}),
            ("Caregiver", "Let's take it slow, Ralph. Remember what happened last time you got up too fast? How about we sit and talk first?", "cautious", {"pitch": 0.98, "speed": "slow"}),
            ("Ralph", "Last time? I don't... well, maybe. But I really need to... to do something. It's important!", "uncertain", {"pitch": 1.05, "speed": "normal"}),
            ("Caregiver", "I understand. Hey, I heard you grew grapes in Hawaii. That must have been interesting with the climate there.", "redirecting", {"pitch": 1.02, "speed": "normal"}),
            ("Ralph", "Grapes? Oh right! Yes! Had to be creative with the volcanic soil. Say, you ever been to Hawaii?", "engaged", {"pitch": 1.02, "speed": "normal"})
        ]
    }
}

def create_persona_voice(text, speaker, emotion, voice_params, filename):
    """Create voice with persona-specific characteristics"""
    
    # Base TTS settings
    tts_params = {
        "text": text,
        "lang": "en"
    }
    
    # Adjust TLD based on speaker
    if speaker == "Caregiver":
        tts_params["tld"] = "com"  # Neutral American accent
    else:
        # Patients use different accents for variety
        if speaker == "Rachel":
            tts_params["tld"] = "com"  # American
        elif speaker == "Jolene":
            tts_params["tld"] = "ca"  # Canadian (often flatter)
        else:  # Ralph
            tts_params["tld"] = "com"  # American
    
    # Set speech speed
    speed_map = {
        "very_slow": True,
        "slow": True,
        "normal": False,
        "fast": False
    }
    tts_params["slow"] = speed_map.get(voice_params.get("speed", "normal"), False)
    
    # Create TTS
    tts = gTTS(**tts_params)
    
    # Save as mp3 first
    mp3_file = f"temp_{filename}.mp3"
    tts.save(mp3_file)
    
    # Load and modify audio
    audio = AudioSegment.from_mp3(mp3_file)
    
    # Apply elderly voice characteristics
    if speaker != "Caregiver":  # Apply aging effects to patients
        # 1. Lower the pitch significantly for elderly voice
        pitch = voice_params.get("pitch", 1.0)
        # Make base pitch lower for elderly
        elderly_pitch = pitch * 0.85  # Lower by 15% overall
        
        if speaker == "Ralph":  # Male voice - even lower
            elderly_pitch *= 0.9
        
        new_sample_rate = int(audio.frame_rate * elderly_pitch)
        audio = audio._spawn(audio.raw_data, overrides={
            "frame_rate": new_sample_rate
        }).set_frame_rate(44100)
        
        # 2. Add roughness and breathiness
        # Low pass filter to remove high frequencies (elderly voices lose clarity)
        audio = audio.low_pass_filter(2800)
        
        # 3. Add slight tremor for very elderly sound
        if speaker == "Rachel":  # 80 years old, severe dementia
            # More aggressive filtering for older sound
            audio = audio.low_pass_filter(2500)
            # Reduce volume slightly (weaker voice)
            audio = audio - 3
            
        elif speaker == "Jolene":  # 82 years old, depressed
            # Very flat, tired voice
            audio = audio.low_pass_filter(2600)
            audio = audio - 5  # Quieter (withdrawn)
            # Compress dynamic range for monotone effect
            audio = audio.compress_dynamic_range()
            
        elif speaker == "Ralph":  # 78 years old male
            # Rougher male elderly voice
            audio = audio.low_pass_filter(2400)
            audio = audio + 1  # Slightly louder (trying to be heard)
        
        # 4. Add slight distortion for aged voice quality
        # Amplify then clip for slight distortion
        audio = audio + 2
        audio = audio.apply_gain(-2)
        
    else:  # Caregiver - keep clearer but professional
        # Just slight modifications for realism
        audio = audio.low_pass_filter(4000)
    
    # Apply emotion-specific effects
    if speaker == "Rachel" and emotion in ["fear", "panic", "extreme_distress"]:
        # Add shakiness to voice
        audio = audio.fade_in(50).fade_out(50)
        
    elif speaker == "Jolene":
        # Extra compression for flat affect
        audio = audio.compress_dynamic_range()
        
    # Export as wav
    audio.export(f"sample-conv-wav-files/{filename}.wav", format="wav")
    os.remove(mp3_file)
    
    print(f"✅ Created: {filename}.wav - {speaker} ({emotion})")

# Generate all audio files
print("🎤 Creating persona-based dementia conversation audio files...\n")

for persona_name, persona_convs in conversations.items():
    print(f"\n📁 Creating files for {persona_name.upper()}:")
    print("-" * 40)
    
    for turn_type, turns in persona_convs.items():
        print(f"\n  {turn_type} conversation:")
        
        for i, (speaker, text, emotion, voice_params) in enumerate(turns, 1):
            filename = f"{persona_name}_{turn_type}_turn{i}_{speaker.lower()}_{emotion}"
            create_persona_voice(text, speaker, emotion, voice_params, filename)
            time.sleep(0.5)

print("\n" + "="*60)
print("✅ All audio files created successfully!")
print("="*60)

print("\n📊 Summary of created files:")
print("\nRACHEL (Severe Dementia):")
print("- Very difficult conversation")
print("- High distress, needs son-focused redirection")
print("- Files: rachel_severe_*")

print("\nJOLENE (Moderate Dementia):")  
print("- Monotone, depressed affect")
print("- Needs engagement through past interests (music/teaching)")
print("- Files: jolene_moderate_*")

print("\nRALPH (Mild/Significant Dementia):")
print("- Appears conversational but confused")
print("- Fall risk, needs careful redirection")
print("- Files: ralph_mild_*")

print("\n💡 Each persona has 3-turn, 5-turn, and 7-turn conversations")
print("🎯 Ready for emotion labeling and similarity testing!")