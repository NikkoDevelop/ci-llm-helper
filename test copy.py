import os
import pathlib

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks



# ans = pipeline(
#     Tasks.acoustic_noise_suppression,
#     model='damo/speech_dfsmn_ans_psm_48k_causal')

# result = ans('/home/magica/Desktop/hack_noisy1  .wav',
#     output_path='/home/magica/Desktop/test_outdenoisy.wav')


# import speech_recognition as sr


# filename = "/home/magica/Desktop/test_outdenoisy.wav"

# # initialize the recognizer
# r = sr.Recognizer()

# # open the file
# with sr.AudioFile(filename) as source:
#     # listen for the data (load audio to memory)
#     audio_data = r.record(source)
#     # recognize (convert from speech to text)
#     text = r.recognize_google(audio_data,language='ru')
#     print(text)

# import ollama
# response = ollama.chat(model='solar:10.7b-instruct-v1-q5_0', messages=[
#   {
#     'role': 'user',
#     'content': """Hi. Imagine that you are evaluating user posts. I send you a text message, and you rate it on a 100-point scale for aggressiveness. 
#     pattern = {Message: {}, Rating: {}}
# IN THE NEXT ANSWER, DO NOT WRITE ANYTHING EXCEPT A pattern! ANSWER THE WATCH ACCORDING TO THE PATTERN!
# Example of using a pattern:
# {Message: {I hate you!}, Score: {100}}

# Now my task is:
# task= God, stop being stupid!

# Help me, as my expert, to assess the AGGRESSIVENESS of this message!""",
#   },
# ])
# print(response['message']['content'])


import ollama
response = ollama.chat(model='solar:10.7b-instruct-v1-q5_0', messages=[
  {
    'role': 'user',
    'content': """
    
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
data= {
  "questions": [
   "Hello, can I find out my account balance?",
"How can I change the limit on my credit card?",
"I have problems accessing online banking. What should I do?",
"I lost my bank card. How can I block it?",
"How can I set up automatic payments?",
"How much does it cost to transfer money to another account in another currency?",
"Can I open an account for children?",
"What are the credit conditions for young entrepreneurs?",
 "I want to know about your bank's loyalty program.",
"How can I change my password to log in to online banking?",
"What is the mortgage interest rate?",
"I have problems with your bank's mobile application. How can I fix this?",
"Can I open an additional savings account?",
"What documents do I need to open an account?".
  ]
}
The response should contain ONLY a PATTERN of categories and messages. If the message cannot be categorized, then put them in a separate category

    """,
  },
])
print(response['message']['content'])

# from vosk_tts import Model, Synth

# model = Model(model_name="vosk-model-tts-ru-0.6-multi")
# synth = Synth(model)

# synth.synth("Привет мир!", "out.wav", speaker_id=2)

