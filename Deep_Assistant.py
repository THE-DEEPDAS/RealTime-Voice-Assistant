import wave
from time import time
import pyaudio
import numpy as np
from io import BytesIO
import pyttsx3
from groq import Groq
import pygame
import tempfile
import os
from config import (
    FORMAT,
    CHANNELS,
    RATE,
    CHUNK,
    SILENCE_THRESHOLD,
    SILENCE_DURATION,
    PRE_SPEECH_BUFFER_DURATION,
)
from dotenv import load_dotenv
import sounddevice as sd
from scipy.io.wavfile import write

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class VoiceAssistant:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.g_client = Groq(api_key=GROQ_API_KEY)
        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        # Set properties (optional)
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
            data (bytes): Audio data.

        Returns:
            bool: True if the data is considered silence, False otherwise.
        """
        audio_data = np.frombuffer(data, dtype=np.int16)
        if len(audio_data) == 0:  # Ensure audio data is not empty
            return True
        rms = np.sqrt(np.mean(audio_data**2))
        return rms < SILENCE_THRESHOLD

    def listen_for_speech(self):
        """
        Continuously detect silence and start recording when speech is detected.

        Returns:
            BytesIO: The recorded audio bytes.
        """
        print("Listening for speech...")
        duration = 10  # Set a maximum duration for recording
        sample_rate = RATE
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=CHANNELS, dtype='int16')
        sd.wait()  # Wait until recording is finished
        print("Speech detected, processing...")

        # Convert audio data to BytesIO
        audio_bytes = BytesIO()
        write(audio_bytes, sample_rate, audio_data)
        audio_bytes.seek(0)
        return audio_bytes

    def record_audio(self, pre_speech_buffer):
        """
        Record audio until silence is detected.

        Args:
            pre_speech_buffer (list): Buffer containing pre-speech audio data.

        Returns:
            BytesIO: The recorded audio bytes.
        """
        frames = pre_speech_buffer.copy()
        silent_chunks = 0

        print("Recording audio...")
        while True:
            # Record a chunk of audio
            data = sd.rec(CHUNK, samplerate=RATE, channels=CHANNELS, dtype='int16')
            sd.wait()
            frames.append(data)

            # Check for silence
            if self.is_silence(data.tobytes()):
                silent_chunks += 1
            else:
                silent_chunks = 0

            # Stop recording if silence duration exceeds threshold
            if silent_chunks > int(RATE / CHUNK * SILENCE_DURATION):
                break

        # Convert recorded frames to BytesIO
        audio_bytes = BytesIO()
        write(audio_bytes, RATE, np.concatenate(frames))
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
        # Create a temporary file to save the speech
        temp_file_path = tempfile.mktemp(suffix='.wav')  # Changed from .mp3 to .wav
        
        # Generate speech and save to file
        self.tts_engine.save_to_file(text, temp_file_path)
        self.tts_engine.runAndWait()
        
        # Read the file into a BytesIO object
        audio_stream = BytesIO()
        with open(temp_file_path, 'rb') as f:
            audio_stream.write(f.read())
        
        # Clean up the temporary file
        os.remove(temp_file_path)
        
        audio_stream.seek(0)
        return audio_stream

    def audio_stream_to_iterator(self, audio_stream, format='wav'):
        """
        Convert audio stream to an iterator of raw PCM audio bytes using wave module.

        Args:
            audio_stream (BytesIO): The audio stream.
            format (str): The format of the audio stream (must be 'wav').

        Returns:
            bytes: The raw PCM audio bytes.
        """
        if format != 'wav':
            raise ValueError("Only 'wav' format is supported with this implementation.")

        audio_stream.seek(0)
        with wave.open(audio_stream, 'rb') as wf:
            chunk_size = 1024  # Adjust as necessary
            while True:
                data = wf.readframes(chunk_size)
                if not data:
                    break
                yield data

    def stream_audio(self, audio_stream, rate=22050, channels=2, format=pyaudio.paInt16):
        """
        Play audio using pygame mixer.

        Args:
            audio_stream (BytesIO): The audio stream to play.
        """
        try:
            # Create a temporary file to store the audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:  # Changed from .mp3 to .wav
                temp_file.write(audio_stream.getvalue())
                temp_file_path = temp_file.name

            # Initialize pygame mixer with the correct frequency and channels
            pygame.mixer.quit()  # Reset the mixer
            pygame.mixer.init(frequency=22050, channels=2)  # Initialize with specific settings
            
            # Load and play the audio
            pygame.mixer.music.load(temp_file_path)
            pygame.mixer.music.play()
            
            # Wait for the audio to finish playing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
        finally:
            # Clean up the temporary file
            if 'temp_file_path' in locals():
                os.remove(temp_file_path)
            
            # Reset the mixer to default settings
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
        
        # Add the user's query to the conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        system_message = {
            "role": "system", 
            "content": "You are a personal assistant that is helpful. You are part of a realtime voice to voice interaction with the human. Make your responses sound natural, like a human. Respond with fill words like 'hmm', 'ohh', and similar wherever relevant to make your responses sound natural. Remember you cannot give emotions in text and like *laughs* as that won't be recognised to convert into text and will be converted as it is, so only give output that is to be spoken out."
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
            
            # Add the assistant's response to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Keep conversation history manageable (last 10 messages)
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
                
            end = time()
            print(f"Response: {response}\nResponse Time: {end - start:.2f} seconds")
            return response
        except Exception as e:
            print(f"Error getting response from Groq: {e}")
            return "I'm sorry, I couldn't process your request."

    def run(self):
        """
        Main function to run the voice assistant.
        """
        while True:
            try:
                # STT using Groq
                audio_bytes = self.listen_for_speech()
                text = self.speech_to_text(audio_bytes)
                
                if not text:
                    print("No speech detected, listening again...")
                    continue

                # Get response from Groq LLM directly
                response_text = self.chat(text)

                # TTS with pyttsx3 (fast offline)
                audio_stream = self.text_to_speech(response_text)
                self.stream_audio(audio_stream)
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()