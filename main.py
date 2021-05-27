from matplotlib import pyplot as plt
import math, numpy as np
from methods import *

f_call_count = 0



def f(x, t, context):
    k = context[0]
    global f_call_count
    f_call_count += 1
    return -k * x


def integrate(x0, context, h, method, nsteps, trunc_zero=True):
    print(f'Используется метод {method.__name__}')
    t = 0
    x = x0
    global f_call_count
    f_call_count = 0
    x_history = [x]

    # для метода РКФ 4(5)
    if method.__name__ == 'rk45':
        atol = float(input('A_tol = '))
        rtol = float(input('R_tol = '))
        max_time = float(input('До какого момента времени интегрировать: '))
#         atol = 0.0000000001
#         rtol = 0.0000000001
#         max_time = 5
        t_history = [t]
        step_and_time = [h, t]
        while t_history[-1] < max_time:
            x = rk45(x, f, context, step_and_time, atol, rtol)
            x_history.append(x)
            t_history.append(step_and_time[1])
            if x_history[-1] < 0 and trunc_zero:
                x_history[-1] = 0
        print(f'Функция правой части была вызвана {f_call_count} раз')
        f_call_count = 0
        return (x_history, t_history)

    # для метода Адамса-Башфорта
    if method.__name__ == "adams_bashforth":
        acceleration = [0] * 3
        acceleration[0] = x
        acceleration[1] = step_rk4(x, t+h, h, f, context)
        acceleration[2] = step_rk4(acceleration[1], t+2*h, h, f, context)
        x_history = acceleration.copy()
        x = step_rk4(acceleration[2], t+3*h, h, f, context)
        t = t + 3*h
        for i in range(nsteps - 3):
            x = adams_bashforth(x, t, h, f, context, acceleration)
            acceleration = acceleration[1:]
            acceleration.append(x)
            x_history.append(x)
            t += h
            if x_history[-1] < 0 and trunc_zero:
                x_history[-1] = 0
        print(f'Функция правой части была вызвана {f_call_count} раз')
        f_call_count = 0
        return x_history

    # для метода Адамса-Мултона
    if method.__name__ == "adams_moulton":
        acceleration = [0] * 3
        acceleration[0] = x
        acceleration[1] = step_rk4(x, t+h, h, f, context)
        acceleration[2] = step_rk4(acceleration[1], t+2*h, h, f, context)
        x_history = acceleration.copy()
        x = step_rk4(acceleration[2], t+3*h, h, f, context)
        t = t + 3*h
        for i in range(nsteps - 3):
            x = adams_moulton(x, t, h, f, context, acceleration)
            acceleration = acceleration[1:]
            acceleration.append(x)
            x_history.append(x)
            t += h
            if x_history[-1] < 0 and trunc_zero:
                x_history[-1] = 0
        print(f'Функция правой части была вызвана {f_call_count} раз')
        f_call_count = 0
        return x_history

    for i in range(nsteps):
        x = method(x, t, h, f, context)
        t += h
        x_history.append(x)
        if x_history[-1] < 0 and trunc_zero:
            x_history[-1] = 0
    print(f'Функция правой части была вызвана {f_call_count} раз')
    f_call_count = 0
    return x_history


def draw_diff(N0, k, h, n_steps, x):
    # t = np.arange(0, (n_steps+1) * h, h)
    t = np.array([h*i for i in range(len(x))])
    x_true = N0 * (np.e ** (-k * t))
    diff = x_true - x
    plt.plot(t, diff, 'b-.')
    plt.grid()
    plt.show()
    

def display_error(N0, k, h, n_steps, x):
    t = np.array([h*i for i in range(len(x))])
    x_true = N0 * (np.e ** (-k * t))
    diff = x - x_true
    print(f'Максимальная по модулю ошибка = {np.max(np.abs(diff))}')
    print(f'Средняя ошибка = {np.mean(diff)}')
    print(f'Стандартное отклонение ошибки = {np.std(diff)}\n')


def draw_exact(N0, k, h, n_steps):
    t = np.arange(0, n_steps * h + 0.1, 0.001)
    x_true = N0 * (np.e ** (-k * t))
    plt.plot(t, x_true, 'k')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    N0 = 1
    k = 1
    h = 0.1
    n_steps = 50

    plt.figure(figsize=(8, 5))

    # draw exact function
    t = np.arange(0, n_steps * h + 0.1, 0.01)
    x_true = N0 * (np.e ** (-k * t))
    plt.plot(t, x_true, 'k')

    # time intervals for approximation
    t_full = [h * t for t in range(n_steps + 1)]

    x = integrate(N0, [k], h, step_euler_forward, n_steps)
    plt.plot(t_full, x, 'r--')

    x = integrate(N0, [k], h, step_rk4, n_steps)
    plt.plot(t_full, x, 'g:')

    x = integrate(N0, [k], h, step_euler_backward, n_steps)
    plt.plot(t_full, x, 'b-.')

    x = integrate(N0, [k], h, step_implicit_midpoint, n_steps)
    plt.plot(t_full, x, 'c-.')

    x = integrate(N0, [k], h, step_rkf5, n_steps)
    plt.plot(t_full, x, 'm--')

    # show plot
    plt.legend(['Exact function', 'Forward Euler', \
                'RK-4', 'Backward Euler', 'Implicit midpoint'])
    plt.show()
