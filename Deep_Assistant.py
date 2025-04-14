import wave
from time import time
import numpy as np
from io import BytesIO
import pyttsx3
from groq import Groq
import pygame
import tempfile
import os
import sounddevice as sd
from scipy.io.wavfile import write
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

    def listen_for_speech(self):
        """
        Continuously detect silence and start recording when speech is detected.

        Returns:
            BytesIO: The recorded audio bytes.
        """
        print("Listening for speech...")
        pre_speech_buffer = []
        silent_chunks = 0
        recording = False
        frames = []

        def callback(indata, frame_count, time, status):
            nonlocal recording, silent_chunks, pre_speech_buffer, frames

            if status:
                print(f"Sounddevice error: {status}")

            audio_data = indata[:, 0]  # Use the first channel
            if recording:
                frames.append(audio_data)
                if self.is_silence(audio_data):
                    silent_chunks += 1
                else:
                    silent_chunks = 0

                # Stop recording if silence duration exceeds threshold
                if silent_chunks > int(RATE / CHUNK * SILENCE_DURATION):
                    raise sd.CallbackStop()
            else:
                if not self.is_silence(audio_data):
                    recording = True
                    frames = pre_speech_buffer.copy()
                    frames.append(audio_data)
                else:
                    pre_speech_buffer.append(audio_data)
                    if len(pre_speech_buffer) > int(RATE * PRE_SPEECH_BUFFER_DURATION / CHUNK):
                        pre_speech_buffer.pop(0)

        with sd.InputStream(
            samplerate=RATE,
            channels=CHANNELS,
            dtype='float32',
            blocksize=CHUNK,
            callback=callback
        ):
            try:
                sd.sleep(int(10 * 1000))  # Listen for up to 10 seconds
            except sd.CallbackStop:
                pass

        # Convert recorded frames to BytesIO
        audio_bytes = BytesIO()
        write(audio_bytes, RATE, np.concatenate(frames).astype(np.int16))
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

    def run(self):
        """
        Main function to run the voice assistant.
        """
        while True:
            try:
                audio_bytes = self.listen_for_speech()
                text = self.speech_to_text(audio_bytes)
                if not text:
                    print("No speech detected, listening again...")
                    continue

                response_text = self.chat(text)
                audio_stream = self.text_to_speech(response_text)
                self.stream_audio(audio_stream)
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()