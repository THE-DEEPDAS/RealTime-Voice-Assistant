import wave
from time import time
import numpy as np
from io import BytesIO
from groq import Groq
import pygame
import tempfile
import os
from scipy.io.wavfile import write
import toml
from gtts import gTTS
from config import (
    SILENCE_THRESHOLD,
    SILENCE_DURATION,
    PRE_SPEECH_BUFFER_DURATION,
)

# Load environment variables from TOML file
try:
    config = toml.load(".streamlit/config.template.toml")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY") or config["api_keys"]["groq"]
except Exception as e:
    print(f"Error loading config: {e}")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class VoiceAssistant:
    def __init__(self):
        self.g_client = Groq(api_key=GROQ_API_KEY)
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
        Convert text to speech using gTTS.

        Args:
            text (str): The text to convert to speech.

        Returns:
            BytesIO: The audio stream.
        """
        try:
            tts = gTTS(text=text, lang="en")
            temp_file_path = tempfile.mktemp(suffix=".mp3")
            tts.save(temp_file_path)

            # Convert MP3 to WAV for playback
            audio_stream = BytesIO()
            with open(temp_file_path, "rb") as f:
                audio_stream.write(f.read())
            os.remove(temp_file_path)
            audio_stream.seek(0)
            return audio_stream
        except Exception as e:
            print(f"Error generating speech: {e}")
            return None

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
            print(f"An error occurred: {e}")