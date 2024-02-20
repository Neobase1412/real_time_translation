# Real-Time Audio Recording and Transcription

This project provides a Python script that continuously records audio, transcribes the audio in real-time using OpenAI's Whisper API, and then processes the transcription with GPT-4 to generate meeting notes or summaries.

## Features

- Continuous audio recording in predefined segments.
- Real-time audio transcription using OpenAI's Whisper API.
- Processing of transcriptions with GPT-4 for summary or notes generation.
- Two-stage GPT-4 processing for enhanced summary quality.

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/Neobase1412/real_time_translation.git
   ```
2. Navigate into the project directory:
   ```
   cd real_time_translation
   ```
3. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the root of your project directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=<your-api-key-here>
   ```
   Replace `<your-api-key-here>` with your actual OpenAI API key.

## Usage

To start recording and transcription, run the script with:
```
python main.py
```
Press `Enter` to start recording. Press `Enter` again at any time to stop the recording and begin transcription processing.

## Requirements

- Python 3.11+
- A valid OpenAI API key

## Note

This project is for demonstration purposes and requires an actual OpenAI API key for transcription and GPT-4 processing functionalities.
