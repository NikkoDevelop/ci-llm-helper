import requests

# URL вашего FastAPI сервера
url = "http://127.0.0.1:8000/process_wav/"

# Ссылка на .wav файл, который вы хотите обработать
wav_url = "https://fincheck.nikkodev.space/ci-llm/test-noisy-voice-message.wav"

# Отправляем POST-запрос с ссылкой на .wav файл
response = requests.post(url, json={"url": wav_url})

# Проверяем ответ сервера
if response.status_code == 200:
    # Получаем обработанный .wav файл
    processed_wav = response.content
    # Здесь вы можете сохранить обработанный .wav файл или выполнить другие операции с ним
    with open("/home/magica/Desktop/processed_file.wav", "wb") as f:
        f.write(processed_wav)
    print("Файл успешно обработан и сохранен.")
else:
    print("Произошла ошибка:", response.text)
