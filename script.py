import os
import pathlib
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import speech_recognition as sr
import ollama
from vosk_tts import Model, Synth


def denoise_audio(input_file_path, output_file_path):
    ans = pipeline(
        Tasks.acoustic_noise_suppression,
        model='damo/speech_dfsmn_ans_psm_48k_causal')

    result = ans(input_file_path, output_path=output_file_path)
    

def transcribe_audio(input_file):
    filename = input_file
    r = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_data = r.record(source)
        text = r.recognize_google(audio_data, language='ru')
        return text
def evaluate_message(message):
    response = ollama.generate(model='solar:10.7b-instruct-v1-q5_0', prompt="""Hi. Imagine that you are evaluating user posts. I send you a text message, and you rate it on a 100-point scale for aggressiveness. 
    pattern = {Message: {}, Rating: {}}
IN THE NEXT ANSWER, DO NOT WRITE ANYTHING EXCEPT A pattern! ANSWER THE WATCH ACCORDING TO THE PATTERN!

Now my task is:
text message = """+message+""" 

Help me, as my expert, to assess the AGGRESSIVENESS of this message!""")
    return(response['response'])

def categorize_messages(data):
    response = ollama.generate(model='solar:10.7b-instruct-v1-q5_0',prompt=
   """
        
    Hi. Imagine that you are solving the problem of collecting statistics. I'm sending you an array of data messages, and you're collecting statistics on them. CATEGORIZE the subject of messages by pattern. 
    USE ONLY THIS CATEGORY:
    [Bank Account;
    Credit Card;
    Remote Banking;
    Automatic Payments;
    Currency Exchange;
    Youth Banking;
    Loans;
    Mobile App;
    Security;
    Miscellaneous.]

    pattern = [Category{}, Message numbers[]]
    YOUR ANSWER SHOULD CONTAIN NOTHING BUT A pattern!

    the data corresponds to the form: {questions:[]}.

    So, let's move on to the task.
    data= {[
    """+data+"""
    }
    The response should contain ONLY a PATTERN of categories and messages. If the message cannot be categorized, then put them in a separate category

        """,
    )
    return(response['response'])

def generate_speech(text, output_file, speaker_id=2):
    model = Model(model_name="vosk-model-tts-ru-0.6-multi")
    synth = Synth(model)
    synth.synth(text, output_file, speaker_id=speaker_id)

# Sample usage:
# denoise_audio('/home/magica/Desktop/hack_noisy1.wav', '/home/magica/Desktop/test_outdenoisy.wav')
# transcribed_text = transcribe_audio('/home/magica/Desktop/test_outdenoisy.wav')
# print(transcribed_text)
# categorize_messages([{'role': 'user', 'content': 'Hello, can I find out my account balance?'}])
# generate_speech("Привет мир!", "out.wav")
input_file = '/home/magica/Desktop/test-noisy-voice-message.wav'

audio_file = '/home/magica/Desktop/test_outdenoisy.wav'

# denoise_audio(input_file, audio_file)
# transcribed_text = transcribe_audio(audio_file)
# categorized_message = categorize_messages(transcribed_text)
# evaluation = evaluate_message(transcribed_text)

# print("Transcribed Text:", transcribed_text)
# print("Categorized Message:", categorized_message)
# print("Evaluation:", evaluation)

generate_speech("Привет Как мне оформить аренду сейфа", "out.wav", speaker_id=2)