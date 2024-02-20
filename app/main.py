import os
import threading
import time
from datetime import datetime
import sounddevice as sd
import scipy.io.wavfile
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from a .env file for secure API key management
load_dotenv()

# Initialize the OpenAI client with API key from environment variables
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# A threading event to signal when to stop recording
stop_recording = threading.Event()

def record_and_transcribe_continuous(duration, fs, stop_event):
    """
    Continuously records audio and transcribes each segment using the Whisper API.
    Each transcription is appended to a file in real-time.
    """
    # Ensure the data directory exists for storing recordings
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Ensure the transcription file's directory exists
    transcription_file = './content/transcription.txt'
    os.makedirs(os.path.dirname(transcription_file), exist_ok=True)

    # Loop until the stop event is signaled
    while not stop_event.is_set():
        # Generate a timestamped filename for the recording
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = os.path.join(data_dir, f'{timestamp}.wav')
        
        print("Recording...")
        # Record audio for the given duration and sample rate
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()  # Wait for the recording to finish
        print("Recording stopped.")
        
        # Save the recorded audio to a WAV file
        scipy.io.wavfile.write(filename, fs, recording)
        print(f"File saved as {filename}")
        
        # Transcribe the audio file using the Whisper API
        transcription_text = transcribe_audio(filename)
        # Append the transcription to the transcription file
        if transcription_text:
            with open(transcription_file, 'a') as file:
                file.write(transcription_text + '\n')

        # Pause briefly to reduce CPU usage
        time.sleep(0.1)

def transcribe_audio(audio_path):
    """
    Transcribes the given audio file using the Whisper API provided by OpenAI.
    """
    try:
        # Open and read the audio file, and send it for transcription
        with open(audio_path, "rb") as audio_file:
            transcript_response = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
            )
            # Extract the transcription text from the response
            transcription_text = transcript_response.text
        return transcription_text
    except Exception as e:
        # Return an error message if transcription fails
        return f"Error in transcription: {e}"

def send_openai_request(system_message, user_message):
    """
    Sends a request to OpenAI GPT-4 using the provided system and user messages.
    """
    try:
        # Send the request to GPT-4 and return the response content
        response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content if response.choices else "No response content"
    except Exception as e:
        # Print and return an error message if the request fails
        print(f"Error sending request: {e}")
        return ""

def process_final_transcription():
    """
    Processes the final transcription: uses the initial GPT-4 response as input for a second GPT-4 call.
    The final GPT-4 response is saved to a markdown file.
    """
    # Load system messages and transcriptions for GPT-4 processing
    system_message_path = './content/prompt_clean.txt'
    transcription_file = './content/transcription.txt'
    meeting_notes_file = './content/cleaned_transcription.txt'
    final_system_message_path = './content/prompt_notes.txt'
    final_output_path = './content/final.md'

    # Process initial transcription with GPT-4
    try:
        with open(system_message_path, 'r') as file:
            system_message = file.read()
        with open(transcription_file, 'r') as file:
            user_message = file.read()
        response = send_openai_request(system_message, user_message)
        print("GPT-4 Response saved to cleaned_transcription.txt")
        with open(meeting_notes_file, 'w') as file:
            file.write(response)
    except Exception as e:
        print(f"Error in processing initial transcription: {e}")
        return

    # Second GPT-4 call using the first response
    try:
        with open(final_system_message_path, 'r') as file:
            new_system_message = file.read()
        # Use the first GPT-4 call response as the new user message
        new_user_message = response
        final_response = send_openai_request(new_system_message, new_user_message)
        print("Final GPT-4 Response saved to final.md")
        with open(final_output_path, 'w') as file:
            file.write(final_response)
    except Exception as e:
        print(f"Error in processing final GPT-4 call: {e}")

def main():
    """
    Main function to start the continuous recording and transcription process.
    """
    duration = 5  # Set the duration for each recording segment
    fs = 44100    # Set the sample rate for the recording
    
    print("Press Enter to start recording. Press Enter again at any time to stop.")
    input()  # Wait for the user to press Enter to start
    
    # Start recording in a separate thread
    recording_thread = threading.Thread(target=record_and_transcribe_continuous, args=(duration, fs, stop_recording))
    recording_thread.start()
    
    input()  # Wait for the user to press Enter to stop
    stop_recording.set()  # Signal the recording thread to stop
    
    recording_thread.join()  # Wait for the recording thread to finish
    
    print("Recording and transcription stopped by user.")
    process_final_transcription()  # Process the final transcription with GPT-4

if __name__ == "__main__":
    main()
