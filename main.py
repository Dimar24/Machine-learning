import copy

import datetime as dt
from sklearn import metrics
import numpy as np
import warnings
from numpy import array
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from sklearn.preprocessing import MinMaxScaler

min_date = None
max_date = None
predicted_temperatures_data = list()
x_start = pd.to_datetime('01.01.2012 00:00:00', errors='raise', format='%d.%m.%Y %H:%M:%S')
x_end = pd.to_datetime('01.01.2019 00:00:00', errors='raise', format='%d.%m.%Y %H:%M:%S')
x_future = pd.to_datetime('01.01.2021 00:00:00', errors='raise', format='%d.%m.%Y %H:%M:%S')


def save_predicted_data(predicted_data, model_name):
    print("Хотите сохранить будущие значения расхода в файл?(0; 1)")
    user_enter = input()
    if user_enter == '1':
        print("Введит название файла:")
        file_name = input()
        with open('predicted_data/' + model_name + '_' + file_name + '.txt', 'w') as f:
            for data in predicted_data:
                f.write(str(data) + '\n')

def save_model(model, file_name):
    pickle.dump(model, open(file_name, 'wb'))


def load_model(file_name):
    return pickle.load(open(file_name, 'rb'))


def parser(dt):
    delta = dt - min_date
    return int(delta.total_seconds())  # % 31536000


def graph(x_gr, y_gr, year, x_start_gr, x_end_gr, color):
    grid_size = (6, 2)
    plt.figure(figsize=(15, 10))
    ax1 = plt.subplot2grid(grid_size, (0, 0), colspan=2, rowspan=2)
    ax1.set_xticks(year)
    ax1.set_xticklabels(range(int(x_start_gr.timetuple()[0]), int(x_end_gr.timetuple()[0]) + 2))
    ax1.set(ylabel='Расход')
    ax2 = plt.subplot2grid(grid_size, (2, 0), colspan=2, rowspan=2)
    ax2.set_xticks(year)
    ax2.set_xticklabels(range(int(x_start_gr.timetuple()[0]), int(x_end_gr.timetuple()[0]) + 2))
    ax2.set(ylabel='Давление')
    ax3 = plt.subplot2grid(grid_size, (4, 0), colspan=2, rowspan=2)
    ax3.set_xticks(year)
    ax3.set_xticklabels(range(int(x_start_gr.timetuple()[0]), int(x_end_gr.timetuple()[0]) + 2))
    ax3.set(ylabel='Температура')
    ax1.set_title('База данных')
    ax1.plot(x_gr[0], y_gr, color)
    ax2.plot(x_gr[0], x_gr[1], color)
    ax3.plot(x_gr[0], x_gr[2], color)
    plt.xlabel("Время (Год)")
    plt.show()

def regression(x_graph, x_train_reg, x_test_reg, y_train_reg, y_test_reg, model_reg, title, type_model, years_reg):
    print("Хотите загрузить готовую модель?(0; 1)")
    user_enter = input()
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_x = scaler_x.fit(x_train_reg)
    if user_enter == '1':
        print("Введит название файла:")
        file_name = input()
        model_reg = load_model(type_model + "\\" + type_model + file_name + ".sav")
    else:
        print("Модель обучается...")
        model_reg.fit(scaler_x.transform(x_train_reg), y_train_reg)
        print("Обучение прошло успешно.")
    expected_y = y_test_reg
    model_reg.out_activation_ = 'relu'
    predicted_y = model_reg.predict(scaler_x.transform(x_test_reg))
    print("Среднеквадратическая ошибка (MSE): {0}.".format(metrics.mean_squared_error(expected_y, predicted_y)))
    print("Среднеквадратичная ошибка (R^2): {0}.".format(metrics.r2_score(expected_y, predicted_y)))
    print(model_reg.loss_)
    plt.title(title)
    plt.plot(x_graph, predicted_y, 'r-', label='предсказание')
    plt.ylabel('Расход')
    plt.xlabel('Время (Год)')
    plt.legend(loc='best')
    plt.show()

    start_month = parser(pd.to_datetime('01.01.2019 00:00:00', errors='raise', format='%d.%m.%Y %H:%M:%S'))
    months = array(
        [parser(pd.to_datetime('01.' + str(i) + '.2019 00:00:00', errors='raise', format='%d.%m.%Y %H:%M:%S'))
         - start_month for i in range(1, 13)])
    nameMonts = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь', 'Июль', 'Август', 'Сентябрь', 'Октябрь',
                 'Ноябрь', 'Декабрь']

    plt.figure(figsize=(20, 10))
    plt.title(title)
    plt.xticks(months, nameMonts)
    if len(np.transpose(x_test_reg)) != 1:
        plt.plot(np.transpose(x_test_reg)[0], y_test_reg, 'b-', label='данные')
        plt.plot(np.transpose(x_test_reg)[0], predicted_y, 'r-', label='предсказание')
    else:
        plt.plot(x_test_reg, y_test_reg, 'b-', label='данные')
        plt.plot(x_test_reg, predicted_y, 'r-', label='предсказание')
    plt.ylabel('Расход')
    plt.xlabel('Время (Месяца - 2019)')
    plt.legend(loc='best')
    plt.show()

    future_dates = array(
        [i for i in range(parser(min_date), parser(x_future), 3600)])
    future_dates_periodic = copy.deepcopy(future_dates)
    max_date_index = list(future_dates_periodic).index(parser(max_date))
    j = 1
    for i in range(0, len(future_dates_periodic)):
        if future_dates_periodic[i] == years_reg[j]:
            if j != len(years) - 2:
                while future_dates_periodic[i] < years_reg[j + 1]:
                    future_dates_periodic[i] -= years_reg[j]
                    i += 1
                j += 1
            else:
                while i != len(future_dates_periodic):
                    future_dates_periodic[i] -= years_reg[j]
                    i += 1
    future_dates_str = [str(i.year) + '.' + str(nameMonts[i.month-1])
                        for i in pd.date_range(start=str(max_date.timetuple()[0]) + '.'
                                                     + str(int(max_date.timetuple()[1]) + 1),
                                               end=str(x_future.timetuple()[0]) + '.'
                                                   + str(int(x_future.timetuple()[1]) + 1), freq='M')]
    future_dates_range = [parser(i)
                          for i in pd.date_range(start=max_date, end=x_future, freq='M')]

    future_dates_periodic = future_dates_periodic[max_date_index + 1:]

    if len(np.transpose(x_test_reg)) != 1:
        x_train_reg = np.transpose(x_train_reg)
        x_test_reg = np.transpose(x_test_reg)
        predicted_temperatures_data.append(future_dates_periodic)
        predicted_temperatures_data.append(regressionTemperatures(
                                                   np.concatenate([x_train_reg[0], x_test_reg[0]]).reshape(-1, 1),
                                                   np.concatenate([x_train_reg[1], x_test_reg[1]]),
                                                   future_dates_periodic.reshape(-1, 1), months , nameMonts))
        scaler_x2 = MinMaxScaler(feature_range=(0, 1))
        predicted_y = model_reg.predict(scaler_x2.fit_transform(np.transpose(predicted_temperatures_data)))
    else:
        scaler_x2 = MinMaxScaler(feature_range=(0, 1))
        predicted_y = model_reg.predict(scaler_x2.fit_transform(future_dates_periodic.reshape(-1, 1)))

    plt.figure(figsize=(20, 10))
    plt.title(title)
    plt.xticks(future_dates_range, future_dates_str)
    future_dates = future_dates[max_date_index + 1:]
    plt.plot(future_dates, predicted_y, 'r-', label='предсказание')
    plt.ylabel('Расход')
    plt.xlabel('Время (Год.Месяц)')
    plt.legend(loc='upper right')
    plt.show()

    if user_enter == '0':
        print("Хотите сохранить модель в файл?(0; 1)")
        user_enter = input()
        if user_enter == '1':
            print("Введит название файла:")
            file_name = input()
            save_model(model_reg,type_model + "\\" + type_model + file_name + ".sav")

    save_predicted_data(predicted_y, type_model)

def regressionTemperatures(x_train_reg, y_train_reg, future_dates_periodic, monthsStart, nameMonts):
    nameMonts.append(nameMonts[0])
    months = monthsStart.tolist()
    months.append(monthsStart[11] + monthsStart[1])
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_x = scaler_x.fit(x_train_reg)
    model = MLPRegressor(hidden_layer_sizes=(100, 100),
                         solver='lbfgs',
                         #random_state=9876,
                         activation='relu',
                         tol=0.0001,
                         learning_rate_init=0.001,
                         learning_rate='adaptive',
                         batch_size=365 * 24,
                         alpha=0.001,
                         shuffle=True,
                         verbose=True)
    model.fit(scaler_x.transform(x_train_reg), y_train_reg)
    predicted_y = model.predict(scaler_x.transform(future_dates_periodic))
    plt.figure(figsize=(20, 10))
    plt.title('Предсказание Температуры (цикличное представление года)')
    plt.xticks(months, nameMonts)
    plt.plot(x_train_reg, y_train_reg, 'g-', label='данные')
    plt.plot(future_dates_periodic, predicted_y, 'r-', label='предсказание')
    plt.ylabel('Температура')
    plt.xlabel('Время (Месяц)')
    plt.legend(loc='best')
    plt.show()
    return predicted_y

def round_time(date):
    hours = date.hour
    date -= dt.timedelta(hours=date.hour, minutes=date.minute, seconds=date.second)
    return date + dt.timedelta(hours=hours, minutes=60)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    plt.style.use('ggplot')
    dataset = pd.read_csv("DB\\5" + ".txt", sep="\s*;\s*", decimal=",")
    x = list()
    x.append(dataset.iloc[:, 0].values)
    x.append(dataset.iloc[:, 2].values.astype(float))
    x.append(dataset.iloc[:, 3].values.astype(float))
    y = dataset.iloc[:, 1].values.astype(float)
    x = array(x)

    min_date = x_start
    years = array(
        [parser(pd.to_datetime('01.01.' + str(i) + ' 00:00:00', errors='raise', format='%d.%m.%Y %H:%M:%S')) for i in
         range(int(x_start.timetuple()[0]), int(x_end.timetuple()[0]) + 3)])
    x[0] = pd.to_datetime(x[0], errors='raise', format='%d.%m.%Y %H:%M:%S')
    max_date = max(x[0])
    max_date = round_time(max_date)
    x[0] = array([parser(item) for item in x[0]])
    graph(x, y, years, x_start, x_end, 'r-')

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

    graph(x, y, years, x_start, x_end, 'g-')

    number = list(x[0]).index(parser(x_end))
    x_data = copy.deepcopy(x[0])
    j = 1
    for i in range(0, len(x[0])):
        if x[0][i] == years[j]:
            if j < len(years) - 3:
                while x[0][i] < years[j + 1]:
                    x[0][i] -= years[j]
                    i += 1
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
    plt.xticks(years, range(int(x_start.timetuple()[0]), int(x_end.timetuple()[0]) + 2))
    plt.plot(x_data, y, 'b-', label='данные')
    regression(x_data[number:], x_train[0].reshape(-1, 1), x_test[0].reshape(-1, 1), y_train.reshape(-1, 1),
               y_test.reshape(-1, 1), MLPRegressor(hidden_layer_sizes=(300, 300),
                                                   solver='lbfgs',
                                                   random_state=9876,
                                                   activation='relu',
                                                   max_iter=10000,
                                                   tol=0.0000001,
                                                   learning_rate_init=0.000001,
                                                   learning_rate='adaptive',
                                                   batch_size=365 * 24,
                                                   alpha=0.00001,
                                                   shuffle=True,
                                                   verbose=True),
               "Предсказание по дате", "time_", years)

    plt.figure(figsize=(20, 10))
    plt.plot(x_data, y, 'b-', label='данные')
    plt.xticks(years, range(int(x_start.timetuple()[0]), int(x_end.timetuple()[0]) + 2))
    regression(x_data[number:], np.transpose(x_train), np.transpose(x_test), y_train,
               y_test, MLPRegressor(hidden_layer_sizes=(300, 300),
                                    solver='lbfgs',
                                    random_state=9876,
                                    activation='relu',
                                    max_iter=10000,
                                    tol=0.0000001,
                                    learning_rate_init=0.000001,
                                    learning_rate='adaptive',
                                    batch_size=365 * 24,
                                    alpha=0.00001,
                                    shuffle=True,
                                    verbose=True),
               "Предсказание по дате и температуре", "timeAndTemperature_", years)  # 4 14