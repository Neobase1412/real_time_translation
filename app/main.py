import os
import threading
import time
from datetime import datetime
import sounddevice as sd
import scipy.io.wavfile
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

stop_recording = threading.Event()

def record_and_transcribe_continuous(duration, fs, stop_event):
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    transcription_file = './content/transcription.txt'
    os.makedirs(os.path.dirname(transcription_file), exist_ok=True)

    while not stop_event.is_set():
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = os.path.join(data_dir, f'{timestamp}.wav')
        
        print("Recording...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        print("Recording stopped.")
        
        scipy.io.wavfile.write(filename, fs, recording)
        print(f"File saved as {filename}")
        
        # Transcribe the audio file
        transcription_text = transcribe_audio(filename)
        if transcription_text:
            with open(transcription_file, 'a') as file:
                file.write(transcription_text + '\n')

        if stop_event.is_set():
            break

        time.sleep(0.1)

def transcribe_audio(audio_path):
    try:
        with open(audio_path, "rb") as audio_file:
            transcript_response = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
            )
            transcription_text = transcript_response.text
        return transcription_text
    except Exception as e:
        return f"Error in transcription: {e}"


def send_openai_request(system_message, user_message):
    try:
        response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content if response.choices else "No response content"
    except Exception as e:
        print(f"Error sending request: {e}")
        return ""

def process_final_transcription():
    system_message_path = './content/prompt_clean.txt'
    transcription_file = './content/transcription.txt'
    meeting_notes_file = './content/cleaned_transcription.txt'
    final_system_message_path = './content/prompt_notes.txt'
    final_output_path = './content/final.md'

    # First GPT-4 call
    try:
        with open(system_message_path, 'r') as file:
            system_message = file.read()

        with open(transcription_file, 'r') as file:
            user_message = file.read()

        response = send_openai_request(system_message, user_message)
        print("GPT-4 Response saved to meeting_transcription.txt")
        with open(meeting_notes_file, 'w') as file:
            file.write(response)
    except Exception as e:
        print(f"Error in processing initial transcription: {e}")
        return

    # Second GPT-4 call using the response
    try:
        with open(final_system_message_path, 'r') as file:
            new_system_message = file.read()

        # Use the response from the first GPT-4 call as user message
        new_user_message = response

        final_response = send_openai_request(new_system_message, new_user_message)
        print("Final GPT-4 Response saved to final.txt")
        with open(final_output_path, 'w') as file:
            file.write(final_response)
    except Exception as e:
        print(f"Error in processing final GPT-4 call: {e}")


def main():
    duration = 5  # Duration of each recording segment in seconds
    fs = 44100  # Sample rate
    
    print("Press Enter to start recording. Press Enter again at any time to stop.")
    input()  # Wait for the user to press Enter to start

    recording_thread = threading.Thread(target=record_and_transcribe_continuous, args=(duration, fs, stop_recording))
    
    recording_thread.start()
    
    input()
    stop_recording.set()
    
    recording_thread.join()
    
    print("Recording and transcription stopped by user.")
    process_final_transcription()

if __name__ == "__main__":
    main()
