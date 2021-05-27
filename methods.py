import numpy as np
from collections import deque
from scipy.optimize import fsolve

def step_euler_forward(x0, t0, h, func, context):
    return x0 + h * func(x0, t0, context)


def step_rk4(x0, t0, h, func, context):
    k1 = func(x0, t0, context)
    k2 = func(x0 + h * (k1 / 2), t0 + h / 2, context)
    k3 = func(x0 + h * (k2 / 2), t0 + h / 2, context)
    k4 = func(x0 + h * k3, t0 + h, context)
    xp = x0 + 1 / 6 * h * (k1 + 2 * k2 + 2 * k3 + k4)
    # print("step rk4 {} {} {} {} {}".format(k1, k2, k3, k4, xp))
    return xp


def step_euler_backward(x0, t0, h, func, context):
    t = t0
    x = x0
    xp = x + h * func(x, t, context)
    f_ = lambda xp, f, t0, x0, h, context: xp - x0 - h * f(xp, t0 + h, context)
    result = fsolve(f_, xp, args=(func, t, x, h, context))
    return result


# # fixed point does not always converge
# def step_euler_backward(x0, t0, h, func, context):
#     tp = t0 + h
#     current = x0 + h * func(x0, t0, context)
#     # fixed point
#     for _ in range(50):
#         buffer = current
#         current = x0 + h * func(current, tp, context)
#         if np.abs(current - buffer) < 1e-10:
#             break
#     return current


def step_midpoint(x0, t0, h, func, context):
    k1 = func(x0, t0, context)
    k2 = func(x0 + h * (k1 / 2), t0 + h / 2, context)
    xp = x0 + h * k2
    return xp


def step_ralston2(x0, t0, h, func, context):
    k1 = func(x0, t0, context)
    k2 = func(x0 + 2 * h * (k1 / 3), t0 + 2 * h / 3, context)
    xp = x0 + h * (k1 / 4 + 3 * k2 / 4)
    return xp


def step_ralston3(x0, t0, h, func, context):
    k1 = func(x0, t0, context)
    k2 = func(x0 + h * (k1 / 2), t0 + h / 2, context)
    k3 = func(x0 + 3 * h * (k2 / 4), t0 + 3 * h / 4, context)
    xp = x0 + h * (2 * k1 / 9 + k2 / 3 + 4 * k3 / 9)
    return xp


def step_rkf5(x0, t0, h, func, context):
    k1 = func(x0, t0, context)
    k2 = func(x0 + h * (k1 / 4), t0 + h / 4, context)
    k3 = func(x0 + h * (3 / 32 * k1 + 9 / 32 * k2), t0 + 3 / 8 * h, context)
    k4 = func(x0 + h * (1932 / 2197 * k1 - 7200 / 2197 * k2 + 7296 / 2197 * k3), t0 + 12 / 13 * h, context)
    k5 = func(x0 + h * (439 / 216 * k1 - 8 * k2 + 3680 / 513 * k3 - 845 / 4104 * k4), t0 + h, context)
    k6 = func(x0 + h * (-8 / 27 * k1 + 2 * k2 - 3544 / 2565 * k3 + 1859 / 4104 * k4 - 11 / 40 * k5), t0 + 0.5 * h,
              context)
    xp = x0 + h * (16 / 135 * k1 + 6656 / 12825 * k3 + 28561 / 56430 * k4 - 9 / 50 * k5 + 2 / 55 * k6)
    return xp


def step_implicit_midpoint(x0, t0, h, func, context):
    tp = t0 + 0.5 * h
    k = x0
    # fixed point
    for _ in range(50):
        buffer = k
        k = func(x0 + 0.5 * h * k, tp, context)
        if np.abs(k - buffer) < 1e-10:
            break
    xp = x0 + h * k
    return xp


# Адаптивный шаг
def rk45(x0, func, context, step_and_time, atol, rtol):
    h = step_and_time[0]
    t0 = step_and_time[1]
    k1 = func(x0, t0, context)
    k2 = func(x0 + h * (k1 / 4), t0 + h / 4, context)
    k3 = func(x0 + h * (3 / 32 * k1 + 9 / 32 * k2), t0 + 3 / 8 * h, context)
    k4 = func(x0 + h * (1932 / 2197 * k1 - 7200 / 2197 * k2 + 7296 / 2197 * k3), t0 + 12 / 13 * h, context)
    k5 = func(x0 + h * (439 / 216 * k1 - 8 * k2 + 3680 / 513 * k3 - 845 / 4104 * k4), t0 + h, context)
    k6 = func(x0 + h * (-8 / 27 * k1 + 2 * k2 - 3544 / 2565 * k3 + 1859 / 4104 * k4 - 11 / 40 * k5), t0 + 0.5 * h,
              context)
    # оценка ошибки 5 порядок
    x_hat = x0 + h * (16 / 135 * k1 + 6656 / 12825 * k3 + 28561 / 56430 * k4 - 9 / 50 * k5 + 2 / 55 * k6)
    # оценка метода 4 порядок
    xp = x0 + h * (25 / 216 * k1 + 1408 / 2565 * k3 + 2197 / 4104 * k4 - 0.2 * k5)
    tol = atol + np.max([x_hat, xp]) * rtol
    err = np.sqrt((x_hat - xp) ** 2 / tol)
    # h_opt = h * (1 / err)**(1 / (min(p, p_hat) + 1)
    h_opt = h * (1 / err) ** 0.2
    step_and_time[0] = h_opt
    step_and_time[1] = t0 + h_opt
    return xp


# Многошаговый метод (число шагов - 4)
# acc - [x_(n-3) x_(n-2) x_(n-1)]
def adams_bashforth(x0, t0, h, func, context, acc):
    # посчитать f используя иксы (acc и x0) и подставить в формулу
    f_3 = func(acc[0], t0 - 3*h, context)
    f_2 = func(acc[1], t0 - 2*h, context)
    f_1 = func(acc[2], t0 - 1*h, context)
    f_0 = func(x0, t0, context)
    xp = x0 + h * (55 / 24 * f_0 - 59 / 24 * f_1 + 37 / 24 * f_2 - 3 / 8 * f_3)
    return xp


# Многошаговый метод (число шагов - 4)
# context[1] - предыдущие 3 значения функции [f_(n-3) f_(n-2) f_(n-1)]
def adams_moulton(x0, t0, h, func, context, acc):
    f_3 = func(acc[0], t0 - 3*h, context)
    f_2 = func(acc[1], t0 - 2*h, context)
    f_1 = func(acc[2], t0 - 1*h, context)
    f_0 = func(x0, t0, context)
    # Predictor
    x_next = x0 + h * (55 / 24 * f_0 - 59 / 24 * f_1 + 37 / 24 * f_2 - 3 / 8 * f_3)
    # Evaluator
    f_next = func(x_next, t0 + h, context)
    # Corrector
    x_next = x0 + h / 24 * (9 * f_next + 19 * f_0 - 5 * f_1 + f_2)
    # Evaluator
    f_next = func(x_next, t0 + h, context)
    # Corrector
    x_next = x0 + h / 24 * (9 * f_next + 19 * f_0 - 5 * f_1 + f_2)
    return x_next
