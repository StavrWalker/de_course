import matplotlib.pyplot as plt
import numpy as np
from newmwthods import *

f_call_count = 0


def f(x, t, context):
    k = context[0]
    global f_call_count
    f_call_count += 1
    return -k * x


x0 = 1
k = 3
h = 0.1
t0 = 0
t_end = 4

draw_exact(x0, k, t0, t_end)

x = []
t = []


def my_callback(tn, xn):
    t.append(tn)
    x.append(xn)


x = []
t = []
euler_backward(x0, t0, t_end, h, f, [k], my_callback)
print(f'Made {len(t) - 1} steps')
plt.plot(t, x, 'r--')
draw_exact(x0, k, t0, t_end)
plt.show()
plt.figure(1)

methods = [midpoint,
           ralston2,
           ralston3,
           rk4,
           rkf5,
           implicit_midpoint,
           adams_bashforth,
           adams_moulton]

x0 = 1
t0 = 0
t_end = 2
h = 0.01
k = 3


def relative_error(x, t):
    x_true = x0 * np.e ** (-k * np.asarray(t))
    diff = np.abs(np.asarray(x) - np.asarray(x_true)) / np.asarray(x_true)
    return diff


time_intervals = {}
for method in methods:
    current_t_end = t_end
    print(f'For method "{method.__name__}"')
    x = []
    t = []
    method(x0, t0, current_t_end, h, f, [k], my_callback)
    while x[-1] > x0 * (10 ** (-12)):
        # print(f't_end = {current_t_end} | x[-1] = {x[-1]}')
        current_t_end += h
        x = []
        t = []
        method(x0, t0, current_t_end, h, f, [k], my_callback, trunc_zero=False)
    print(f't_end = {current_t_end} | x[-1] = {x[-1]}')
    time_intervals[method.__name__] = current_t_end
    # print(f'Max error = {np.max(relative_error(x, t))}')
for method in time_intervals.keys():
    time_intervals[method] = round(time_intervals[method], 2)
print(time_intervals)

global_err = [0] * len(methods)
i = 0
for method in methods:
    f_call_count = 0
    cur_h = h
    print(f'For method "{method.__name__}"')
    x = []
    t = []
    method(x0, t0, time_intervals[method.__name__], cur_h, f, [k], my_callback)
    print(f'h = {cur_h} | err = {np.max(relative_error(x, t))}')
    while np.max(relative_error(x, t)) > 1e-9:
        cur_h *= 0.9
        x = []
        t = []
        method(x0, t0, time_intervals[method.__name__], cur_h, f, [k], my_callback)
        print(f'h = {cur_h} | err = {np.max(relative_error(x, t))} | f_call = {f_call_count}')
        f_call_count = 0
    global_err[i] = (t, x)
    i += 1

print(time_intervals)

k = 3
f_call_count = 0
x = []
t = []
rk45(x0, 0, t_end, 0.1, f, [k], my_callback)
print(f_call_count)
plt.plot(t, x, 'r--')
x_true = x0 * (np.e ** (-k * np.asarray(t)))
diff = x - x_true
print(f'Максимальная по модулю ошибка = {np.max(np.abs(diff))}')
print(f'Средняя ошибка = {np.mean(diff)}')
print(f'Стандартное отклонение ошибки = {np.std(diff)}')
plt.show()
plt.figure(6)

f_call_count = 0
x = []
t = []
dopri5(x0, 0, t_end, 0.1, f, [k], my_callback)
print(f_call_count)
plt.plot(t, x, 'b--')
plt.show()
plt.figure(7)
x_true = x0 * (np.e ** (-k * np.asarray(t)))
diff = x - x_true
print(f'Максимальная по модулю ошибка = {np.max(np.abs(diff))}')
print(f'Средняя ошибка = {np.mean(diff)}')
print(f'Стандартное отклонение ошибки = {np.std(diff)}')

plt.plot(t, x0 * (np.e ** (-k * np.asarray(t))), 'k')
plt.legend(["Рунге-Кутты-Фельберга 4(5)", "Дормана-Принса 5(4)", "x(t)"])
plt.show()
plt.figure(8)

k = 4
for h in [0.1, 0.01, 0.001]:
    f_call_count = 0
    x = []
    t = []
    adams_moulton(x0, 0, 2, h, f, [k], my_callback)
    x_true = x0 * (np.e ** (-k * np.asarray(t)))
    diff = x - x_true
    print(f'\nМаксимальная по модулю ошибка = {np.max(np.abs(diff))}')
    print(f'Средняя ошибка = {np.mean(diff)}')
    print(f'Стандартное отклонение ошибки = {np.std(diff)}')
    print(f_call_count)

t = []
x = []
h = 0.01
x0 = 1
k = 3
t0 = 0
t_end = 9.3
f_call_count = 0

dopri5(x0, t0, t_end, h, f, [k], my_callback)
print(f_call_count)
plt.plot(t, x)
plt.show()
plt.figure(9)
print(np.max(relative_error(x, t, x0, k)))
plt.plot(t, relative_error(x, t, x0, k))
plt.show()
plt.figure(10)

def rk45(x0, t0, t_end, h, func, context, callback, trunc_zero=True):
    atol = rtol = 10 ** (-10)
    x = x0
    t = t0
    callback(t, x)
    true_error = [0]
    eval_error = [0]
    while t < t_end:
        print(f"t = {t} h = {h}")
        new_f = lambda time: x * (np.e ** (-k * time))
        k1 = func(x, t, context)
        k2 = func(x + h * (k1 / 4), t + h / 4, context)
        k3 = func(x + h * (3 / 32 * k1 + 9 / 32 * k2), t + 3 / 8 * h, context)
        k4 = func(x + h * (1932 / 2197 * k1 - 7200 / 2197 * k2 + 7296 / 2197 * k3), t + 12 / 13 * h, context)
        k5 = func(x + h * (439 / 216 * k1 - 8 * k2 + 3680 / 513 * k3 - 845 / 4104 * k4), t + h, context)
        k6 = func(x + h * (-8 / 27 * k1 + 2 * k2 - 3544 / 2565 * k3 + 1859 / 4104 * k4 - 11 / 40 * k5), t + 0.5 * h,
                  context)
        # оценка ошибки 5 порядок
        x_hat = x + h * (16 / 135 * k1 + 6656 / 12825 * k3 + 28561 / 56430 * k4 - 9 / 50 * k5 + 2 / 55 * k6)
        # оценка метода 4 порядок
        xp = x + h * (25 / 216 * k1 + 1408 / 2565 * k3 + 2197 / 4104 * k4 - 0.2 * k5)
        tol = atol + np.max(np.abs([x_hat, xp])) * rtol
        err = np.sqrt((x_hat - xp) ** 2 / tol)
        # h_opt = h * (1 / err)**(1 / (min(p, p_hat) + 1)
        h = h * (1 / err) ** 0.2
        t += h
        x = xp
        true_error.append(x - new_f(t))
        eval_error.append(err)
        if x < 0 and trunc_zero:
            x = 0
        callback(t, x)
    return eval_error, true_error


t0 = 0
t_end = 3
h = 0.1
k = 1

x = []
t = []
eval_error, true_error = rk45(x0, t0, t_end, h, f, [k], my_callback)
print(f'Made {len(t) - 1} steps')
plt.plot(t, x, 'r')
plt.grid()
plt.show()
plt.figure(11)

plt.plot(t, eval_error, 'r--')
plt.plot(t, true_error, 'b-.')
plt.legend(["Оценка локальной ошибки", "Истинное значение локальной ошибки"])
plt.show()
plt.figure(12)


def dopri5(x0, t0, t_end, h, func, context, callback, trunc_zero=True):
    #     atol = float(input('A_tol = '))
    #     rtol = float(input('R_tol = '))
    atol = rtol = 10 ** (-10)
    x = x0
    t = t0
    callback(t, x)
    true_error = [0]
    eval_error = [0]
    while t < t_end:
        new_f = lambda time: x * (np.e ** (-k * time))
        k1 = func(x, t, context)
        k2 = func(x + h * (k1 / 5), t + h / 5, context)
        k3 = func(x + h * (3 / 40 * k1 + 9 / 40 * k2), t + 3 / 10 * h, context)
        k4 = func(x + h * (44 / 45 * k1 - 56 / 15 * k2 + 32 / 9 * k3), t + 4 / 5 * h, context)
        k5 = func(x + h * (19372 / 6561 * k1 - 25360 / 2187 * k2 + 64448 / 6561 * k3 - 212 / 729 * k4), t + 8 / 9 * h,
                  context)
        k6 = func(x + h * (9017 / 3168 * k1 - 355 / 33 * k2 + 46732 / 5247 * k3 + 49 / 176 * k4 - 5103 / 18656 * k5),
                  t + h,
                  context)
        xp = x + h * (35 / 384 * k1 + 500 / 1113 * k3 + 125 / 192 * k4 - 2187 / 6784 * k5 + 11 / 84 * k6)
        k7 = func(xp, t + h, context)

        x_hat = x + h * (
                5179 / 57600 * k1 + 7571 / 16695 * k3 + 393 / 640 * k4 - 92097 / 339200 * k5 + 187 / 2100 * k6 + 1 / 40 * k7)
        tol = atol + np.max(np.abs([x_hat, xp])) * rtol
        err = np.sqrt((x_hat - xp) ** 2 / tol)
        # h_opt = h * (1 / err)**(1 / (min(p, p_hat) + 1)
        h = h * (1 / err) ** 0.2
        t += h
        x = xp
        true_error.append(x - new_f(t))
        eval_error.append(err)
        if x < 0 and trunc_zero:
            x = 0
        callback(t, x)
    return eval_error, true_error


t = []
x = []
eval_error, true_error = dopri5(x0, t0, t_end, h, f, [k], my_callback)
print(f'Made {len(t) - 1} steps')
plt.plot(t, x, 'r--')
plt.grid()
plt.show()
plt.figure(13)

plt.plot(t, eval_error, 'r--')
plt.plot(t, true_error, 'b-.')
plt.legend(["Оценка локальной ошибки", "Истинное значение локальной ошибки"])
plt.show()
plt.figure(14)