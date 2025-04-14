import wave
from time import time
import numpy as np
from io import BytesIO
import pyttsx3
from groq import Groq
import pygame
import tempfile
import os
from scipy.io.wavfile import write
from config import (
    SILENCE_THRESHOLD,
    SILENCE_DURATION,
    PRE_SPEECH_BUFFER_DURATION,
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class VoiceAssistant:
    def __init__(self):
        self.g_client = Groq(api_key=GROQ_API_KEY)
        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 180)  # Speed
        self.tts_engine.setProperty('volume', 0.9)  # Volume
        # Initialize pygame mixer
        pygame.mixer.init()
        # Store conversation history
        self.conversation_history = []

    def is_silence(self, data):
        """
        Detect if the provided audio data is silence.

        Args:
            data (numpy.ndarray): Audio data.

        Returns:
            bool: True if the data is considered silence, False otherwise.
        """
        if len(data) == 0:  # Ensure audio data is not empty
            return True
        rms = np.sqrt(np.mean(data**2))
        return rms < SILENCE_THRESHOLD

    def process_uploaded_audio(self, uploaded_file):
        """
        Process an uploaded audio file.

        Args:
            uploaded_file: The uploaded audio file.

        Returns:
            BytesIO: The audio bytes.
        """
        audio_bytes = BytesIO(uploaded_file.read())
        audio_bytes.seek(0)
        return audio_bytes

    def speech_to_text(self, audio_bytes):
        """
        Transcribe speech to text using Groq's whisper model.

        Args:
            audio_bytes (BytesIO): The audio bytes to transcribe.

        Returns:
            str: The transcribed text.
        """
        start = time()
        audio_bytes.seek(0)
        try:
            transcription = self.g_client.audio.transcriptions.create(
                file=("temp.wav", audio_bytes.read()),
                model="whisper-large-v3",
            )
            end = time()
            text = transcription.text
            print(f"Recognized text: {text}")
            print(f"Transcription time: {end - start:.2f} seconds")
            return text
        except Exception as e:
            print(f"Error transcribing audio with Groq: {e}")
            return ""

    def text_to_speech(self, text):
        """
        Convert text to speech using pyttsx3 (offline TTS).

        Args:
            text (str): The text to convert to speech.

        Returns:
            BytesIO: The audio stream.
        """
        temp_file_path = tempfile.mktemp(suffix='.wav')
        self.tts_engine.save_to_file(text, temp_file_path)
        self.tts_engine.runAndWait()

        audio_stream = BytesIO()
        with open(temp_file_path, 'rb') as f:
            audio_stream.write(f.read())
        os.remove(temp_file_path)
        audio_stream.seek(0)
        return audio_stream

    def stream_audio(self, audio_stream):
        """
        Play audio using pygame mixer.

        Args:
            audio_stream (BytesIO): The audio stream to play.
        """
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file.write(audio_stream.getvalue())
                temp_file_path = temp_file.name

            pygame.mixer.quit()
            pygame.mixer.init(frequency=22050, channels=2)
            pygame.mixer.music.load(temp_file_path)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        finally:
            if 'temp_file_path' in locals():
                os.remove(temp_file_path)
            pygame.mixer.quit()
            pygame.mixer.init()

    def chat(self, query: str) -> str:
        """
        Get response directly from Groq LLM.

        Args:
            query (str): The user's query.

        Returns:
            str: Groq LLM response.
        """
        start = time()
        self.conversation_history.append({"role": "user", "content": query})

        system_message = {
            "role": "system",
            "content": "You are a personal assistant that is helpful. Respond naturally."
        }

        messages = [system_message] + self.conversation_history

        try:
            chat_completion = self.g_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                temperature=0.2,
                max_tokens=800,
            )
            response = chat_completion.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": response})
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            end = time()
            print(f"Response: {response}\nResponse Time: {end - start:.2f} seconds")
            return response
        except Exception as e:
            print(f"Error getting response from Groq: {e}")
            return "I'm sorry, I couldn't process your request."

    def run(self, uploaded_audio):
        """
        Main function to run the voice assistant.
        """
        try:
            audio_bytes = self.process_uploaded_audio(uploaded_audio)
            text = self.speech_to_text(audio_bytes)
            if not text:
                print("No speech detected, listening again...")
                return

            response_text = self.chat(text)
            audio_stream = self.text_to_speech(response_text)
            self.stream_audio(audio_stream)
        except Exception as e:
            print(f"An error occurred: {e}")            # API keys mate environment variables load karo
            import os
            from dotenv import load_dotenv
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
            FORMAT = "int16"              # Audio format (16-bit)
            CHANNELS = 1                  # Mono audio
            RATE = 44100                  # Sample rate (Hz)
            CHUNK = 1024                  # Audio chunk size
            
            # Voice detection na parameters
            SILENCE_THRESHOLD = 200       # Silence detect karva mate threshold
            SILENCE_DURATION = 2          # Ketla seconds maun pachi recording band karvi
            PRE_SPEECH_BUFFER_DURATION = 0.5  # Speech detect thai te pela ketla seconds no audio rakhvo