
from argostranslate import translate
from argostranslate import package
package.install_from_path('/home/magica/Desktop/ci-llm-helper/translate-en_ru-1_9.argosmodel')

def get_argos_model(source, target):
    lang = f'{source} -> {target}'
    source_lang = [model for model in translate.get_installed_languages() if lang in map(repr, model.translations_from)]
    target_lang = [model for model in translate.get_installed_languages() if lang in map(repr, model.translations_to)]
    
    return source_lang[0].get_translation(target_lang[0])

argos_ru_en = get_argos_model('English', 'Russian')

text = (argos_ru_en.translate('''
                            Response: To rent an individual safety box at "Центр-инvest" bank, you can follow these steps:
1. Visit our website at www.centrinvest.
2. Navigate to the Safe Boxes section for physical persons by clicking on this link: https://www.centrinvest.ru/for-individuals/safe-boxes/.
3. Check out the offered safety box sizes and their pricing. If needed, reach out to our customer support at 8 800 200 99 29 (toll-free in Russia). You can also find contacts for other countries or for queries related to cards and ATMs mentioned on the website.
4. To finalize the safe box rental process, follow the instructions given by our team after discussing your specific needs.
                            
                            '''))
# "I think it's a disease, a real, complete disease."

from vosk_tts import Model, Synth

def generate_speech(text, output_file, speaker_id=2):
    model = Model(model_name="vosk-model-tts-ru-0.6-multi")
    synth = Synth(model)
    synth.synth(text, output_file, speaker_id=speaker_id)

generate_speech(text, "out.wav", speaker_id=2)