import os
from dotenv import load_dotenv
import sounddevice as sd
import numpy as np
import scipy.io.wavfile
from datetime import datetime
from openai import OpenAI

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def record_audio(duration, fs):
    """
    Record audio from the microphone and save it to a file with a timestamp.
    :param duration: Recording duration in seconds
    :param fs: Sampling rate
    """
    # Create the data directory if it doesn't exist
    if not os.path.exists('./data'):
        os.makedirs('./data')
    
    # Generate a timestamped filename
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'./data/{timestamp}.wav'
    
    print("Recording...")
    # Adjust the `channels` parameter to 1 for mono recording
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording stopped.")
    
    # Save the recording to a WAV file
    scipy.io.wavfile.write(filename, fs, recording)
    print(f"File saved as {filename}")
    
    return filename

def transcribe_audio(audio_path):
    """
    Transcribe the given audio file using Whisper API.
    :param audio_path: Path to the audio file to transcribe
    """
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1", 
            )
            transcription_text = transcript.text
        print(f"Transcription: {transcription_text}")
        return transcription_text
    except Exception as e:
        print(f"Error in transcription: {e}")
        return None

if __name__ == "__main__":
    duration = 5  # seconds
    fs = 44100  # Sample rate
    
    # Step 1: Record audio and get the saved filename
    filename = record_audio(duration, fs)
    
    # Step 2: Transcribe recorded audio
    transcribe_audio(filename)

