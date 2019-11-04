import copy

from sklearn import metrics
import numpy as np
import warnings
from numpy import array
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

min_date = None

def save_model(model,file_name):
    pickle.dump(model, open(file_name, 'wb'))

def load_model(file_name):
    return pickle.load(open(file_name, 'rb'))

def parser(dt):
    delta = dt - min_date
    return int(delta.total_seconds())  # % 31536000

def graphics(x, y, year, x_start, x_end, color):
    gridsize = (6, 2)
    plt.figure(figsize=(15, 10))
    ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2)
    ax1.set_xticks(year)
    ax1.set_xticklabels(range(int(x_start.timetuple()[0]), int(x_end.timetuple()[0]) + 2))
    ax2 = plt.subplot2grid(gridsize, (2, 0), colspan=2, rowspan=2)
    ax2.set_xticks(year)
    ax2.set_xticklabels(range(int(x_start.timetuple()[0]), int(x_end.timetuple()[0]) + 2))
    ax3 = plt.subplot2grid(gridsize, (4, 0), colspan=2, rowspan=2)
    ax3.set_xticks(year)
    ax3.set_xticklabels(range(int(x_start.timetuple()[0]), int(x_end.timetuple()[0]) + 2))
    ax1.plot(x[0], y, color)
    ax2.plot(x[0], x[1], color)
    ax3.plot(x[0], x[2], color)
    plt.xticks(year, range(int(x_start.timetuple()[0]), int(x_end.timetuple()[0]) + 2))
    plt.show()

def regression(x_grap, x_train_reg, x_test_reg, y_train_reg, y_test_reg, model_reg, title, type_model):
    print("Хотите загрузить готовую модель?(0; 1)")
    user_enter = input()
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_x = scaler_x.fit(x_train_reg)
    if user_enter == '1':
        print("Введит название файла:")
        file_name = input()
        model_reg = load_model(type_model + file_name + ".sav")
    else:
        print("Модель обучается...")
        model_reg.fit(scaler_x.transform(x_train_reg), y_train_reg)
        print("Обучение прошло успешно.")
    print(model_reg.n_iter_ )
    print(model_reg.n_outputs_ )
    expected_y = y_test_reg
    model_reg.out_activation_ = 'relu'
    predicted_y = model_reg.predict(scaler_x.transform(x_test_reg))
    print("Среднеквадратичная ошибка: {0}.\n\n".format(metrics.mean_squared_error(expected_y, predicted_y)))
    if len(x_test_reg) != 1:
        x_test_reg = np.transpose(x_test_reg)
        x_test_reg = x_test_reg[0]
    plt.title(title)
    #model_reg.out_activation_ = 'relu'
    plt.plot(x_grap, model_reg.predict(scaler_x.transform(x_train_reg)), 'r-')
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.title(title)
    plt.plot(x_test_reg, y_test_reg, 'b-')
    plt.plot(x_test_reg, predicted_y, 'r-')
    plt.show()

    print(predicted_y[predicted_y < 0])

    if user_enter == '0':
        print("Хотите сохранить модель в файл?(0; 1)")
        user_enter = input()
        if user_enter == '1':
            print("Введит название файла:")
            file_name = input()
            save_model(model_reg, type_model + file_name + ".sav")


def regression2(x_grap, x_train_reg, x_test_reg, y_train_reg, model_reg, title, type_model):
    print("Хотите загрузить готовую модель?(0; 1)")
    user_enter = input()
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_x = scaler_x.fit(x_train_reg)
    if user_enter == '1':
        print("Введит название файла:")
        file_name = input()
        model_reg = load_model(type_model + file_name + ".sav")
    else:
        print("Модель обучается...")
        model_reg.fit(scaler_x.transform(x_train_reg), y_train_reg)
        print("Обучение прошло успешно.")
    print(model_reg.n_iter_ )
    print(model_reg.n_outputs_ )
    model_reg.out_activation_ = 'relu'
    predicted_y = model_reg.predict(scaler_x.transform(x_test_reg))
    if len(x_test_reg) != 1:
        x_test_reg = np.transpose(x_test_reg)
        x_test_reg = x_test_reg[0]
    plt.title(title)
    #model_reg.out_activation_ = 'relu'
    plt.plot(x_grap[len(x_train_reg):], predicted_y, 'r-')
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.title(title)
    plt.plot(x_test_reg, predicted_y, 'r-')
    plt.show()

    print(predicted_y[predicted_y < 0])

    if user_enter == '0':
        print("Хотите сохранить модель в файл?(0; 1)")
        user_enter = input()
        if user_enter == '1':
            print("Введит название файла:")
            file_name = input()
            save_model(model_reg, type_model + file_name + ".sav")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    plt.style.use('ggplot')
    dataset = pd.read_csv("33" + ".txt", sep="\s*;\s*", decimal=",")
    x = list()
    x.append(dataset.iloc[:, 0].values)
    x.append(dataset.iloc[:, 2].values.astype(float))
    x.append(dataset.iloc[:, 3].values.astype(float))
    y = dataset.iloc[:, 1].values.astype(float)
    x = array(x)

    #print(y[y < 0])

    x_start = pd.to_datetime('01.01.2012 00:00:00', errors='raise', format='%d.%m.%Y %H:%M:%S')
    x_end = pd.to_datetime('01.01.2019 00:00:00', errors='raise', format='%d.%m.%Y %H:%M:%S')
    min_date = x_start
    years = array(
        [parser(pd.to_datetime('01.01.' + str(i) + ' 00:00:00', errors='raise', format='%d.%m.%Y %H:%M:%S')) for i in
         range(int(x_start.timetuple()[0]), int(x_end.timetuple()[0])+2)])
    x[0] = pd.to_datetime(x[0], errors='raise', format='%d.%m.%Y %H:%M:%S')
    x[0] = array([parser(item) for item in x[0]])
    graphics(x, y, years, x_start, x_end, 'y-')
    for i, value in enumerate(x[0]):
        x[0][i] = (int(round(x[0][i] / 3600))) * 3600

    i = 0
    j = 0
    tempX0 = x[0]
    tempX1 = x[1]
    tempX2 = x[2]
    tempY0 = y
    povtoren = 0
    while i < len(tempX0):
        if i != (len(tempX0) - 1):
            j = i + 1
            while j < len(tempX0):
                if tempX0[i] == tempX0[j]:
                    if tempX0[i] == tempX0[j]:
                        tempX0 = np.delete(tempX0, j)
                        tempX1 = np.delete(tempX1, j)
                        tempX2 = np.delete(tempX2, j)
                        tempY0 = np.delete(tempY0, j)
                        j = j - 1
                    elif tempX0[j] - tempX0[j + 1] < -3600:
                        tempX0[j] = tempX0[j] + 3600
                    j = j + 1
                elif tempX0[i] != tempX0[j]:
                    i = j
                    break
        else:
            i = i + 1
    x = list()
    x.append(tempX0)
    x.append(tempX1)
    x.append(tempX2)
    y = tempY0

    #propusk = 0
    #for i, value in enumerate(x[0]):
    #    if i != (len(x[0]) - 1) and x[0][i + 1] - value > 3600:  # менять промежуток интерполяции
    #        # print("{0} - {1} = {2}".format(x[0][i + 1], value, x[0][i + 1] - value))
    #        propusk = propusk + 1

    propusk = 0
    tempX0 = list()
    tempX1 = list()
    tempX2 = list()
    tempY0 = list()
    start = 0
    for i, value in enumerate(x[0]):
        if i != (len(x[0]) - 1) and x[0][i + 1] - value > 3600:  # менять промежуток интерполяции
            kol = int((x[0][i + 1] - value) / 3600)
            tempX0.extend(np.ndarray.tolist(x[0][start:i + 1]))
            tempX1.extend(np.ndarray.tolist(x[1][start:i + 1]))
            tempX2.extend(np.ndarray.tolist(x[2][start:i + 1]))
            tempY0.extend(np.ndarray.tolist(y[start:i + 1]))
            razX1 = (x[1][i + 1] - x[1][i]) / kol
            razX2 = (x[2][i + 1] - x[2][i]) / kol
            razY0 = (y[i + 1] - y[i]) / kol
            for j in range(kol):
                tempX0.append(x[0][i] + 3600 * (j + 1))
                tempX1.append(x[1][i] + razX1 * (j + 1))
                tempX2.append(x[2][i] + razX2 * (j + 1))
                tempY0.append(y[i] + razY0 * (j + 1))
                # plt.plot(tempX[-1], tempY[-1], 'yo', markersize=3)
            start = i + 2
        elif i == (len(x[0]) - 1):
            tempX0.extend(np.ndarray.tolist(x[0][start:]))
            tempX1.extend(np.ndarray.tolist(x[1][start:]))
            tempX2.extend(np.ndarray.tolist(x[2][start:]))
            tempY0.extend(np.ndarray.tolist(y[start:]))
    x[0] = array(tempX0)
    x[1] = array(tempX1)
    x[2] = array(tempX2)
    y = array(tempY0)



    #print(y[y < 0])
    graphics(x, y, years, x_start, x_end, 'g-')

    number = list(x[0]).index(parser(x_end))

    x_data = copy.deepcopy(x[0])

    j = 1
    for i in range(0, len(x[0])):
        if x[0][i] == years[j]:
            if j != len(years) - 2:
                while x[0][i] < years[j + 1]:
                    x[0][i] -= years[j]
                    i += 1
                print(x[0][i])
                j += 1
            else:
                while i != len(x[0]):
                    x[0][i] -= years[j]
                    i += 1

    x[1] = array(tempX2)
    x[2] = array(tempX1)
    x_train = list()
    x_test = list()
    x_train.append(x[0][:number])
    x_train.append(x[1][:number])
    x_test.append(x[0][number:])
    x_test.append(x[1][number:])
    y_train = y[:number]
    y_test = y[number:]


    plt.figure(figsize=(20, 10))
    plt.xticks(years, range(int(x_start.timetuple()[0]), int(x_end.timetuple()[0])+2))
    plt.plot(x_data, y, 'b-')
    regression(x_data[:number], x_train[0].reshape(-1, 1), x_test[0].reshape(-1, 1), y_train.reshape(-1, 1),
               y_test.reshape(-1, 1), MLPRegressor(hidden_layer_sizes=(400, 400),
                                                   solver='lbfgs',
                                                   #random_state=9876,
                                                   activation='relu',
                                                   max_iter=1000000000,
                                                   tol=0.000000001,
                                                   learning_rate_init=0.000001,
                                                   learning_rate='adaptive',
                                                   batch_size=365*24*4,
                                                   alpha=0.000100,
                                                   shuffle=True,
                                                   verbose=True),
               "Предсказание по дате", "time_")

    print(x_data)

    plt.figure(figsize=(20, 10))
    plt.plot(x_data, y, 'b-')
    plt.xticks(years, range(int(x_start.timetuple()[0]), int(x_end.timetuple()[0]) + 2))
    regression(x_data[:number], np.transpose(x_train), np.transpose(x_test), y_train,
              y_test, MLPRegressor(hidden_layer_sizes=(500, 500),
                                   solver='lbfgs',
                                   #random_state=9876,
                                   activation='relu',
                                   max_iter=1000000000,
                                   tol=0.00000000001,
                                   learning_rate_init=0.00000000001,
                                   learning_rate='adaptive',
                                   batch_size=365*24/2,
                                   alpha=0.000001,
                                   shuffle=True,
                                   verbose=True),
              "Предсказание по дате и температуре {0}".format(19), "timeAndTemperature_")  # 4 14


    x_middle = pd.to_datetime('01.01.2019 00:00:00', errors='raise', format='%d.%m.%Y %H:%M:%S')
    x_end = pd.to_datetime('01.01.2021 00:00:00', errors='raise', format='%d.%m.%Y %H:%M:%S')

    x_list_time = list()
    x_list_time.append(x[0][0])
    x_list_time.append(parser(x_end))

    print(x_list_time)

    propusk = 0
    temp_x_list_time = list()
    for i, value in enumerate(x_list_time):
        if i != (len(x_list_time) - 1) and x_list_time[i + 1] - value > 3600:  # менять промежуток интерполяции
            kol = int((x_list_time[i + 1] - value) / 3600)
            for j in range(kol):
                temp_x_list_time.append(x_list_time[i] + 3600 * (j + 1))
    x_list_time = copy.deepcopy(temp_x_list_time)

    print(x_list_time)

    years = array(
        [parser(pd.to_datetime('01.01.' + str(i) + ' 00:00:00', errors='raise', format='%d.%m.%Y %H:%M:%S')) for i in
         range(int(x_start.timetuple()[0]), int(x_end.timetuple()[0])+2)])

    j = 1
    for i in range(0, len(x_list_time)):
        if x_list_time[i] == years[j]:
            if j != len(years) - 2:
                while x_list_time[i] < years[j + 1]:
                    x_list_time[i] -= years[j]
                    i += 1
                print(x_list_time[i])
                j += 1
            else:
                while i != len(x_list_time):
                    x_list_time[i] -= years[j]
                    i += 1

    x_train = array(x_list_time[:number])
    x_test = array(x_list_time[number:])
    y_train = array(y[:number])

    plt.figure(figsize=(20, 10))
    plt.xticks(years, range(int(x_start.timetuple()[0]), int(x_end.timetuple()[0])+2))
    plt.plot(temp_x_list_time[:len(y)], y, 'b-')
    regression2(temp_x_list_time, x_train.reshape(-1, 1), x_test.reshape(-1, 1), y_train.reshape(-1, 1),
                MLPRegressor(hidden_layer_sizes=(400, 400),
                             solver='lbfgs',
                             #random_state=9876,
                             activation='relu',
                             max_iter=1000000000,
                             tol=0.000000001,
                             learning_rate_init=0.000001,
                             learning_rate='adaptive',
                             batch_size=365*24*4,
                             alpha=0.000100,
                             shuffle=True,
                             verbose=True),
                "Предсказание по дате", "timePredict_")


