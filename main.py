import os
import pathlib
import requests
from fastapi import FastAPI, File, UploadFile
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

app = FastAPI()

# Задаем путь для временного сохранения загруженного файла
TEMP_FILE_PATH = "temp.wav"

# Функция для загрузки файла по ссылке
def download_file(url: str, file_path: str):
    response = requests.get(url)
    with open(file_path, "wb") as f:
        f.write(response.content)

# POST-запрос для обработки .wav файла
@app.post("/process_wav/")
async def process_wav(url: str):
    # Скачиваем .wav файл по ссылке
    download_file(url, TEMP_FILE_PATH)

    # Инициализируем модель для подавления шума
    ans = pipeline(
        Tasks.acoustic_noise_suppression,
        model='damo/speech_dfsmn_ans_psm_48k_causal'
    )

    # Обрабатываем .wav файл
    result = ans(TEMP_FILE_PATH, output_path=None)

    # Возвращаем обработанный .wav файл
    return result
