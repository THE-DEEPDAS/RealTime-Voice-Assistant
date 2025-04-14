import streamlit as st
import time
from Deep_Assistant import VoiceAssistant
import streamlit.components.v1 as components

st.set_page_config(page_title="Magical Voice Assistant", layout="wide")

# Custom CSS for black-hole magical animations
def load_css():
    st.markdown("""
        <style>
        body {
            margin: 0;
            padding: 0;
            background: radial-gradient(circle, #000000, #1a1a1a, #000000, #ffd700);
            height: 100vh;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .black-hole {
            position: absolute;
            top: 50%;
            left: 50%;
            width: 300px;
            height: 300px;
            background: radial-gradient(circle, #000000, #1a1a1a, #000000);
            border-radius: 50%;
            box-shadow: 0 0 50px #ffd700, inset 0 0 50px #ffffff;
            transform: translate(-50%, -50%);
        }

        .black-hole::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 500px;
            height: 500px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0));
            box-shadow: 0 0 60px 30px rgba(255, 255, 255, 0.3);
            transform: translate(-50%, -50%);
            z-index: -1;
        }

        @keyframes circular-motion {
            0% { transform: translate(-50%, -50%) rotate(0deg) translateX(200px) rotate(0deg); }
            100% { transform: translate(-50%, -50%) rotate(360deg) translateX(200px) rotate(-360deg); }
        }

        .magical-ball {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: radial-gradient(circle at 30% 30%, #ffd700, #ffffff);
            box-shadow: 0 0 20px #ffd700, inset 0 0 20px #ffffff;
            position: absolute;
            top: 50%;
            left: 50%;
            animation: circular-motion 5s linear infinite;
        }

        .container {
            position: relative;
            z-index: 10;
            text-align: center;
            color: #ffffff;
            padding: 2rem;
        }

        .status-text {
            color: #ffd700;
            text-align: center;
            font-size: 1.5rem;
            margin-top: 2rem;
            text-shadow: 0 0 10px #ffffff;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    load_css()

    # Add the black hole and magical ball
    st.markdown('<div class="black-hole"></div>', unsafe_allow_html=True)
    st.markdown('<div class="magical-ball"></div>', unsafe_allow_html=True)

    # Main container
    with st.container():
        st.markdown('<div class="container">', unsafe_allow_html=True)

        st.title("✨ Magical Black-Hole Voice Assistant ✨")

        # Initialize voice assistant
        if 'assistant' not in st.session_state:
            st.session_state.assistant = VoiceAssistant()
            st.session_state.status = "Waiting..."
            st.session_state.conversation = []

        # Status display
        status_placeholder = st.empty()
        conversation_placeholder = st.empty()

        # Start/Stop button
        if st.button("🎙️ Start Listening", key="start_button"):
            try:
                status_placeholder.markdown('<p class="status-text">Listening...</p>', unsafe_allow_html=True)

                # Get audio input
                audio_bytes = st.session_state.assistant.listen_for_speech()
                status_placeholder.markdown('<p class="status-text">Processing speech...</p>', unsafe_allow_html=True)

                # Convert speech to text
                text = st.session_state.assistant.speech_to_text(audio_bytes)

                if text:
                    # Get AI response
                    status_placeholder.markdown('<p class="status-text">Getting response...</p>', unsafe_allow_html=True)
                    response_text = st.session_state.assistant.chat(text)

                    # Add to conversation history
                    st.session_state.conversation.append({"user": text, "assistant": response_text})

                    # Convert response to speech
                    status_placeholder.markdown('<p class="status-text">Speaking...</p>', unsafe_allow_html=True)
                    audio_stream = st.session_state.assistant.text_to_speech(response_text)
                    st.session_state.assistant.stream_audio(audio_stream)

                    # Display conversation
                    with conversation_placeholder.container():
                        for conv in st.session_state.conversation:
                            st.markdown(f"**You:** {conv['user']}")
                            st.markdown(f"**Assistant:** {conv['assistant']}")
                            st.markdown("---")

                status_placeholder.markdown('<p class="status-text">Ready for next input!</p>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                status_placeholder.markdown('<p class="status-text">Error occurred!</p>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()