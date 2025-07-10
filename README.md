# Voice Assistant Project

## Overview
The Voice Assistant project is a voice-activated assistant that utilizes speech recognition and text-to-speech conversion to interact with users. It leverages the Groq API for generating responses and provides a user-friendly interface through a Streamlit web application.

## Files
- **Deep_Assistant.py**: Contains the `VoiceAssistant` class, which handles audio input, speech recognition, text-to-speech conversion, and interaction with the Groq API for chat responses.
- **app.py**: Sets up a Streamlit web application that provides a user interface for the voice assistant. It includes custom CSS for styling and manages the conversation flow between the user and the assistant.
- **requirements.txt**: Lists the necessary Python packages required to run the project.
- **README.md**: Documentation for the project, explaining how to set it up and use it.

## Installation
To set up the project, follow these steps:

1. Clone the repository or download the project files.
2. Navigate to the project directory.
3. Install the required packages using pip:

```
pip install -r requirements.txt
```

## Usage
To run the voice assistant application, execute the following command:

```
streamlit run app.py
```

Once the application is running, you can interact with the voice assistant by clicking the "Start Listening" button and speaking your queries.
