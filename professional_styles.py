"""
Professional CSS styles for Alzheimer's AI Prize demo
"""

PROFESSIONAL_CSS = """
<style>
    /* Overall app styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #FFFFFF;
    }
    
    /* Remove default padding */
    .main > div {
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Header styling */
    h1 {
        color: #FFFFFF !important;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-weight: 300;
        font-size: 2.5rem !important;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    h2 {
        color: #F0F8FF !important;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-weight: 300;
        font-size: 1.8rem !important;
        margin-top: 2rem;
        border-bottom: 2px solid #E6F3FF;
        padding-bottom: 0.5rem;
    }
    
    h3 {
        color: #E6F3FF !important;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-weight: 400;
        font-size: 1.4rem !important;
        margin-top: 1.5rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #E6F3FF;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    /* Control panel styling */
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.15) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 8px !important;
        backdrop-filter: blur(10px);
    }
    
    .stSelectbox > div > div > select {
        color: #FFFFFF !important;
        background-color: transparent !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #4CAF50, #45a049) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3) !important;
        transition: all 0.3s ease !important;
        text-transform: none !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4) !important;
        background: linear-gradient(45deg, #45a049, #4CAF50) !important;
    }
    
    /* Secondary buttons */
    .stButton[data-testid="baseButton-secondary"] > button {
        background: linear-gradient(45deg, #FF6B6B, #FF5252) !important;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3) !important;
    }
    
    .stButton[data-testid="baseButton-secondary"] > button:hover {
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4) !important;
    }
    
    /* Metrics styling */
    .metric-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    [data-testid="metric-container"] > div {
        color: #FFFFFF !important;
    }
    
    /* Code/terminal styling */
    .stCode {
        background: rgba(0, 0, 0, 0.4) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(5px) !important;
    }
    
    /* Audio conversation styling */
    .audio-turn {
        background: rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin: 0.75rem 0 !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Speaker identification */
    .speaker-patient {
        border-left: 4px solid #FFD700 !important;
    }
    
    .speaker-caregiver {
        border-left: 4px solid #4CAF50 !important;
    }
    
    /* Similarity indicators */
    .similarity-high { color: #4CAF50 !important; font-weight: bold; }
    .similarity-medium { color: #FFC107 !important; font-weight: bold; }
    .similarity-low { color: #FF5722 !important; font-weight: bold; }
    
    /* Progress indicators */
    .stProgress > div > div {
        background: linear-gradient(90deg, #4CAF50, #8BC34A) !important;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #4CAF50 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 0 0 8px 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Success/info/warning styling */
    .stSuccess {
        background: rgba(76, 175, 80, 0.2) !important;
        border: 1px solid rgba(76, 175, 80, 0.5) !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
    }
    
    .stInfo {
        background: rgba(33, 150, 243, 0.2) !important;
        border: 1px solid rgba(33, 150, 243, 0.5) !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
    }
    
    .stWarning {
        background: rgba(255, 193, 7, 0.2) !important;
        border: 1px solid rgba(255, 193, 7, 0.5) !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(0, 0, 0, 0.2) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Audio player styling */
    audio {
        width: 100% !important;
        border-radius: 8px !important;
        background: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Loading animation */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .processing {
        animation: pulse 2s infinite;
    }
    
    /* Demo title styling */
    .demo-title {
        text-align: center;
        padding: 2rem 0;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Feature highlight */
    .feature-highlight {
        background: linear-gradient(135deg, rgba(255, 215, 0, 0.2), rgba(255, 165, 0, 0.2));
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 215, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Hide Streamlit elements */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.5);
    }
</style>
"""

DEMO_HEADER = """
<div class="demo-title">
    <h1>🧠 A Comprehensive Conversational Dataset for Dementia Care</h1>
    <div class="subtitle">
        Integrating Expert-Designed Personas, Clinical Scenarios, and Episodic Discourse Analysis<br>
        <strong>Team Noyce</strong> • Alzheimer's Insights AI Prize 2025
    </div>
</div>
"""

def apply_professional_styling():
    """Apply professional CSS styling to Streamlit app"""
    import streamlit as st
    st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)
    st.markdown(DEMO_HEADER, unsafe_allow_html=True)

def create_speaker_container(speaker, text, emotion, similarity=None):
    """Create a styled container for speaker text"""
    speaker_class = "speaker-patient" if speaker != "Caregiver" else "speaker-caregiver"
    emoji = "🎤" if speaker != "Caregiver" else "👨‍⚕️"
    
    sim_text = ""
    if similarity is not None:
        if similarity > 0.8:
            sim_class = "similarity-high"
            sim_icon = "🟢"
        elif similarity > 0.6:
            sim_class = "similarity-medium" 
            sim_icon = "🟡"
        else:
            sim_class = "similarity-low"
            sim_icon = "🔴"
        
        sim_text = f'<span class="{sim_class}">{sim_icon} {similarity:.1%}</span>'
    
    return f"""
    <div class="audio-turn {speaker_class}">
        <strong>{emoji} {speaker}:</strong> {text}<br>
        <small>Emotion: {emotion}</small> {sim_text}
    </div>
    """