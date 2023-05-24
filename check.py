import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from functionXYZ import f_xy


def make_valid_data(x, y):
    xgrid, ygrid = np.meshgrid(x, y)
    zgrid = f_xy(xgrid, ygrid)
    return xgrid, ygrid, zgrid


def main():
    # загрузка модели
    model = keras.models.load_model('asl_model')

    # Границы для самой сетки и для проверяемых данных
    x_start = 0
    x_end = 2
    y_start = -3
    y_end = -1
    step_axes = 0.05  # шаг, через который будут данные для осей
    step_model = 0.2  # шаг, через который будут данные для проверки

    # Оси x и y
    x = np.arange(x_start, x_end, step_axes)
    y = np.arange(y_start, y_end, step_axes)
    x_point, y_point, z_point = make_valid_data(x, y)

    # Данные для модели
    x_model = np.arange(x_start, x_end, step_model)
    x_model_length = x_model.shape[0]
    y_model = np.arange(y_start, y_end, step_model)
    y_model_length = y_model.shape[0]

    # Входные данные для модели
    x_data = np.ones((x_model_length*y_model_length, 3))
    k = 0
    for x_i in x_model:
        for y_i in y_model:
            x_data[k][0] = x_i
            x_data[k][1] = y_i
            k += 1
    z_model = model.predict(x_data)
    xmp, ymp = np.meshgrid(x_model, y_model)
    zmp = z_model.reshape((xmp.shape[0], xmp.shape[1]))

    fig = plt.figure()
    plt.clf()
    axes = fig.add_subplot(projection='3d')
    axes.scatter(x_point, y_point, z_point, c='blue', marker='o')
    axes.scatter(xmp, ymp, zmp, c='red', marker='+')
    plt.show()


if __name__ == '__main__':
    main()
