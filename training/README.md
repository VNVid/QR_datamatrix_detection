# QR_datamatrix_detection

Как обучить детектор

0. Установим python3.
1. Клонируем репозиторий:
```
git clone git@github.com:VNVid/QR_datamatrix_detection.git
cd QR_datamatrix_detection/training
```
2. Скачиваем зависимости:
```
pip3 install -r requirements.txt --no-cache-dir
```
3. Запускаем train.py. В качестве аргументов можно передать путь к директории для сохранения чекпоинтов модели и путь до последнего чекпоинта, с которого продолжится обучение (при первом запуске обучения этот аргумент не указывается). По умолчанию чекпоинты будут сохранятся рядом с файлом в папке lightning_logs. Пример:
```
python3 train.py
```
