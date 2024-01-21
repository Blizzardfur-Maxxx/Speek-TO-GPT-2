import speech_recognition as sr
import gpt_2_simple as gpt2
from gtts import gTTS
from io import BytesIO
import os
import json
from pydub import AudioSegment
from pydub.playback import play as pydub_play
import tensorflow.compat.v1 as tf
import tempfile

tf.disable_v2_behavior()  # Use TensorFlow v1 behavior

# Set the path to your locally fine-tuned GPT-2 model
gpt2_model_path = "models/124M"
gpt2_model_path = os.path.abspath(gpt2_model_path)

# Create a TensorFlow session
tf.compat.v1.reset_default_graph()
sess = gpt2.start_tf_sess()

# Try to load the GPT-2 model with error handling
try:
    gpt2.load_gpt2(sess, model_name=os.path.abspath(gpt2_model_path))
except FileNotFoundError:
    print(f"Error: The model files are not found in {gpt2_model_path}. Make sure the path is correct.")

def convert_speech_to_text():
    recognizer = sr.Recognizer()

    while True:
        with sr.Microphone() as source:
            print("Say something...")

            try:
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
                text = recognizer.recognize_google(audio)
                return text
            except sr.WaitTimeoutError:
                print("Listening timed out. Please speak again.")
            except sr.UnknownValueError:
                print("No speech detected.")
            except sr.RequestError as e:
                print(f"Error connecting to Google API: {e}")

def load_encoder_file(model_path):
    encoder_file_path = os.path.join(model_path, 'encoder.json')  # Corrected path

    print("Full Encoder File Path:", encoder_file_path)  # Print full path for debugging

    try:
        with open(encoder_file_path, 'r') as f:
            encoder_content = json.load(f)
            print("Encoder Content:", encoder_content)
        return encoder_content
    except json.JSONDecodeError as e:
        print(f"Error decoding encoder file: {e}")
        return None

def generate_gpt2_response(user_input):
    encoder_content = load_encoder_file(gpt2_model_path)

    if encoder_content is None:
        print("Unable to load encoder file. Check for issues.")
        return None

    # Specify the correct checkpoint directory explicitly
    checkpoint_dir = gpt2_model_path  # Change this line

    response = gpt2.generate(sess, model_name=checkpoint_dir, prefix=user_input, length=124, return_as_list=True)[0]

    return response


def convert_text_to_audio_in_memory(text):
    tts = gTTS(text)
    
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        tts.save(temp_file.name)
    
    audio_content = AudioSegment.from_file(temp_file.name, format="mp3")
    
    return audio_content.raw_data

def adjust_pitch(audio_content, semitones):
    audio = AudioSegment(audio_content, sample_width=2, frame_rate=44100, channels=2)
    
    # Lower or increase the pitch based on the semitones value
    adjusted_audio = audio.set_frame_rate(int(audio.frame_rate * (2 ** (semitones / 12.0))))
    
    return adjusted_audio.raw_data

def play_audio_in_memory(audio_content):
    audio = AudioSegment(audio_content, sample_width=2, frame_rate=44100, channels=2)
    pydub_play(audio)

if __name__ == "__main__":
    while True:
        print("Full Model Path:", os.path.abspath(os.path.join(os.getcwd(), gpt2_model_path)))
        user_input = convert_speech_to_text()

        if user_input:
            print(f"You said: {user_input}")

            gpt2_response = generate_gpt2_response(user_input)

            if gpt2_response:
                print(f"GPT-2 response: {gpt2_response}")

                audio_content = convert_text_to_audio_in_memory(gpt2_response)

                # Set the pitch value to your desired number (positive or negative)
                adjusted_audio_content = adjust_pitch(audio_content, 20)

                play_audio_in_memory(adjusted_audio_content)