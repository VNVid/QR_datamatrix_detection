# QR_datamatrix_detection

Как запустить детектор на своих примерах

0. Установим python3.
1. Клонируем репозиторий:
```
git clone git@github.com:VNVid/QR_datamatrix_detection.git
cd QR_datamatrix_detection/inference
```
2. Теперь нужно скачать [чекпоинты моделей](https://disk.yandex.ru/d/wTyC3Z8EhiIu5g)("*.ckpt") и положить их в папку QR_datamatrix_detection рядом с файлом run.py.
3. Скачиваем зависимости:
```
pip3 install -r requirements.txt --no-cache-dir
```
4. Запускаем run.py, в качестве аргументов нужно передать версию детектора, путь к папке с картинками и пороговое значение уверенности модели для отсечения лишних детекций. Пороговое значение лежит в интервале (0, 1), по умолчанию для первой версии используется 0.6, для второй - 0.8. Пример:
```
python3 run.py 1 pics/ 0.6
```
Результаты будут сохранены в папке result.
