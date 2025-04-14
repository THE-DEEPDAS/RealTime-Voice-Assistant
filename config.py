# API keys mate environment variables load karo
import os
from dotenv import load_dotenv
import pyaudio
from enum import Enum

# Environment variables load karo
load_dotenv()

# Different API keys get karo
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")  # Voice generation mate
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")          # OpenAI services mate
GROQ_API_KEY = os.getenv("GROQ_API_KEY")             # Groq AI services mate

# API keys check karo ke available chhe ke nahi
if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY environment variable set nathi")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable set nathi")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable set nathi")

# Different voices na IDs store karva mate enum
class Voices(Enum):
    APHRODITE = "fQuiOHUGZu5WDKWT80Wz"  # Female voice
    ADAM = "pNInz6obpgDQGcFmaJgB"       # Male voice
    CJ_MURPH = "876MHA6EtWKaHTEGzjy5"   # Another voice option

# Default voice set karo
VOICE_ID = Voices.ADAM

# Audio recording na parameters
FORMAT = pyaudio.paInt16        # Audio format (16-bit)
CHANNELS = 1                    # Mono audio
RATE = 44100                   # Sample rate (Hz)
CHUNK = 1024                   # Audio chunk size

# Voice detection na parameters
SILENCE_THRESHOLD = 200        # Silence detect karva mate threshold
SILENCE_DURATION = 2           # Ketla seconds maun pachi recording band karvi
PRE_SPEECH_BUFFER_DURATION = 0.5  # Speech detect thai te pela ketla seconds no audio rakhvo
