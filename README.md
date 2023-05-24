# f_xy_approximation

Построенная модель для аппроксимации функции 2-ух переменных.

## Установка

Требуется [Python](https://www.python.org/downloads/) версии 3.7 или выше и установленный [pip](https://pip.pypa.io/en/stable/getting-started/). Для установки необходимых зависимостей используйте команду:  
1. Для Unix/macOs:
```commandline
python -m pip install -r requirements.txt
```
2. Для Windows:
```commandline
py -m pip install --destination-directory DIR -r requirements.txt
```

## Настройка и тренировка модели

1. Измените, если необходимо, функцию `f_xy` в файле [functionXYZ.py](./functionXYZ.py). По ней будет производиться аппроксимация.
2. Пройти все шаги в [python notebook](./model.ipynb#) в главе "Модель с использованием библиотеки `keras`".
3. По итогу в корневой директории должна появиться директория `asl_model` - обученная модель.

## Проверка модели

1. В корневой директории выполните команду `python3 check.py`. Будут созданы точки реальной функции и результата моделирования и выведены в качестве построенной плоскости.
2. Если есть необходимость проверить специальные точки, выполните соответствующие действия в главе "Модель с использованием библиотеки `keras`" в подглаве "Использование обученной модели".