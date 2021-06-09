import numpy as np
from scipy.optimize import fsolve
from matplotlib import pyplot as plt

f_call_count = 0


def f(x, t, context):
    k = context[0]
    global f_call_count
    f_call_count += 1
    return -k * x


def draw_diff(N0, k, h, x):
    # t = np.arange(0, (n_steps+1) * h, h)
    t = np.array([h*i for i in range(len(x))])
    x_true = N0 * (np.e ** (-k * t))
    diff = x_true - x
    plt.plot(t, diff, 'b-.')
    plt.grid()
    plt.show()
    

def display_error(N0, k, h, x):
    t = np.array([h*i for i in range(len(x))])
    x_true = N0 * (np.e ** (-k * t))
    diff = x - x_true
    print(f'Максимальная по модулю ошибка = {np.max(np.abs(diff))}')
    print(f'Средняя ошибка = {np.mean(diff)}')
    print(f'Стандартное отклонение ошибки = {np.std(diff)}\n')


def draw_exact(N0, k, t0, t_end):
    t = np.arange(0, (t_end - t0) + 0.1, 0.001)
    x_true = N0 * (np.e ** (-k * t))
    plt.plot(t, x_true, 'k')
    plt.grid()
    plt.show()


# def method(x0, t0, t_end, h, func, context):
# return x_history, t_history


def euler_forward(x0, t0, t_end, h, func, context, callback, trunc_zero=True):
    nsteps = int((t_end - t0) / h)
    x = x0
    t = t0
    callback(t, x)
    for i in range(nsteps):
        x =  x + h * func(x, t, context)
        t += h
        if x < 0 and trunc_zero:
            x = 0
        callback(t, x)
    return


def rk4(x0, t0, t_end, h, func, context, callback, trunc_zero=True):
    nsteps = int((t_end - t0) / h)
    x = x0
    t = t0
    callback(t, x)
    for i in range(nsteps):
        k1 = func(x, t, context)
        k2 = func(x + h * (k1 / 2), t + h / 2, context)
        k3 = func(x + h * (k2 / 2), t + h / 2, context)
        k4 = func(x + h * k3, t + h, context)
        x = x + 1 / 6 * h * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if x < 0 and trunc_zero:
            x = 0
        callback(t, x)
    return 


def euler_backward(x0, t0, t_end, h, func, context, callback, trunc_zero=True):
    nsteps = int((t_end - t0) / h)
    x = x0
    t = t0
    callback(t, x)
    for i in range(nsteps):
        x = x + h * func(x, t, context)
        f_ = lambda xp, f, t0, x0, h, context: xp - x0 - h * f(xp, t0 + h, context)
        x = fsolve(f_, x, args=(func, t, x, h, context))
        t += h
        if x < 0 and trunc_zero:
            x = 0
        callback(t, x)
    return


def midpoint(x0, t0, t_end, h, func, context, callback, trunc_zero=True):
    nsteps = int((t_end - t0) / h)
    x = x0
    t = t0
    callback(t, x)
    for i in range(nsteps):
        k1 = func(x, t, context)
        k2 = func(x + h * (k1 / 2), t + h / 2, context)
        x = x + h * k2
        t += h
        if x < 0 and trunc_zero:
            x = 0
        callback(t, x)
    return


def ralston2(x0, t0, t_end, h, func, context, callback, trunc_zero=True):
    nsteps = int((t_end - t0) / h)
    x = x0
    t = t0
    callback(t, x)
    for i in range(nsteps):
        k1 = func(x, t, context)
        k2 = func(x + 2 * h * (k1 / 3), t + 2 * h / 3, context)
        x = x + h * (k1 / 4 + 3 * k2 / 4)
        t += h
        if x < 0 and trunc_zero:
            x = 0
        callback(t, x)
    return


def ralston3(x0, t0, t_end, h, func, context, callback, trunc_zero=True):
    nsteps = int((t_end - t0) / h)
    x = x0
    t = t0
    callback(t, x)
    for i in range(nsteps):
        k1 = func(x, t, context)
        k2 = func(x + h * (k1 / 2), t + h / 2, context)
        k3 = func(x + 3 * h * (k2 / 4), t + 3 * h / 4, context)
        x = x + h * (2 * k1 / 9 + k2 / 3 + 4 * k3 / 9)
        t += h
        if x < 0 and trunc_zero:
            x = 0
        callback(t, x)
    return


def rkf5(x0, t0, t_end, h, func, context, callback, trunc_zero=True):
    nsteps = int((t_end - t0) / h)
    x = x0
    t = t0
    callback(t, x)
    for i in range(nsteps):
        k1 = func(x, t, context)
        k2 = func(x + h * (k1 / 4), t + h / 4, context)
        k3 = func(x + h * (3 / 32 * k1 + 9 / 32 * k2), t + 3 / 8 * h, context)
        k4 = func(x + h * (1932 / 2197 * k1 - 7200 / 2197 * k2 + 7296 / 2197 * k3), t + 12 / 13 * h, context)
        k5 = func(x + h * (439 / 216 * k1 - 8 * k2 + 3680 / 513 * k3 - 845 / 4104 * k4), t + h, context)
        k6 = func(x + h * (-8 / 27 * k1 + 2 * k2 - 3544 / 2565 * k3 + 1859 / 4104 * k4 - 11 / 40 * k5), t + 0.5 * h,
                  context)
        x = x + h * (16 / 135 * k1 + 6656 / 12825 * k3 + 28561 / 56430 * k4 - 9 / 50 * k5 + 2 / 55 * k6)
        t += h
        if x < 0 and trunc_zero:
            x = 0
        callback(t, x)
    return

def implicit_midpoint(x0, t0, t_end, h, func, context, callback, trunc_zero=True):
    nsteps = int((t_end - t0) / h)
    x = x0
    t = t0
    callback(t, x)
    for i in range(nsteps):
        tp = t + 0.5 * h
        k = x
        # fixed point
        for _ in range(50):
            buffer = k
            k = func(x + 0.5 * h * k, tp, context)
            if np.abs(k - buffer) / np.abs(k) < 1e-16:
                break
        x = x + h * k
        t += h
        if x < 0 and trunc_zero:
            x = 0
        callback(t, x)
    return


def rk45(x0, t0, t_end, h, func, context, callback, trunc_zero=True):
#     atol = float(input('A_tol = '))
#     rtol = float(input('R_tol = '))
    atol = rtol = 10**(-10)
    x = x0
    t = t0
    callback(t, x)
    while t < t_end:
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
        if x < 0 and trunc_zero:
            x = 0
        callback(t, x)
    return


def dopri5(x0, t0, t_end, h, func, context, callback, trunc_zero=True):
#     atol = float(input('A_tol = '))
#     rtol = float(input('R_tol = '))
    atol = rtol = 10**(-10)
    x = x0
    t = t0
    callback(t, x)
    while t < t_end:
        k1 = func(x, t, context)
        k2 = func(x + h * (k1 / 5), t + h / 5, context)
        k3 = func(x + h * (3 / 40 * k1 + 9 / 40 * k2), t + 3 / 10 * h, context)
        k4 = func(x + h * (44/45 * k1 - 56/15 * k2 + 32/9 * k3), t + 4/5 * h, context)
        k5 = func(x + h * (19372/6561 * k1 - 25360/2187 * k2 + 64448/6561 * k3 - 212/729 * k4), t + 8/9 * h, context)
        k6 = func(x + h * (9017/3168 * k1 - 355/33 * k2 + 46732/5247 * k3 + 49/176 * k4 - 5103/18656 * k5), t + h,
                  context)
        xp = x + h * (35/384 * k1 + 500/1113 * k3 + 125/192 * k4 - 2187/6784 * k5 + 11/84 * k6)
        k7 = func(xp, t + h, context)
        
        x_hat = x + h * (5179/57600 * k1 + 7571/16695 * k3 + 393/640 * k4 - 92097/339200 * k5 + 187/2100 * k6 + 1/40 * k7)
        tol = atol + np.max(np.abs([x_hat, xp])) * rtol
        err = np.sqrt((x_hat - xp) ** 2 / tol)
        # h_opt = h * (1 / err)**(1 / (min(p, p_hat) + 1)
        h = h * (1 / err) ** 0.2
        t += h
        x = xp
        if x < 0 and trunc_zero:
            x = 0
        callback(t, x)
    return 


def adams_bashforth(x0, t0, t_end, h, func, context, callback, trunc_zero=True):
    nsteps = int((t_end - t0) / h)
    x = x0
    t = t0
    
    def step_rk4(x0, t0, h, func, context):
        k1 = func(x0, t0, context)
        k2 = func(x0 + h * (k1 / 2), t0 + h / 2, context)
        k3 = func(x0 + h * (k2 / 2), t0 + h / 2, context)
        k4 = func(x0 + h * k3, t0 + h, context)
        xp = x0 + 1 / 6 * h * (k1 + 2 * k2 + 2 * k3 + k4)
        return xp
    
    acceleration = [0] * 3
    
    acceleration[0] = x
    callback(t, x)
    
    acceleration[1] = step_rk4(x, t, h, func, context)
    callback(t+h, acceleration[1])
    
    acceleration[2] = step_rk4(acceleration[1], t+h, h, func, context)
    callback(t+2*h, acceleration[2])
    
    x = step_rk4(acceleration[2], t+2*h, h, func, context)
    callback(t+3*h, x)
    
    t = t0 + 3*h
    
    for i in range(nsteps - 3):
        f_3 = func(acceleration[0], t - 3*h, context)
        f_2 = func(acceleration[1], t - 2*h, context)
        f_1 = func(acceleration[2], t - 1*h, context)
        f_0 = func(x, t, context)
        
        acceleration = acceleration[1:]
        acceleration.append(x)
        
        x = x + h * (55 / 24 * f_0 - 59 / 24 * f_1 + 37 / 24 * f_2 - 3 / 8 * f_3)
        t += h
        
        if x < 0 and trunc_zero:
            x = 0
        callback(t, x)
    return


def adams_moulton(x0, t0, t_end, h, func, context, callback, trunc_zero=True):
    nsteps = int((t_end - t0) / h)
    x = x0
    t = t0
    
    def step_rk4(x0, t0, h, func, context):
        k1 = func(x0, t0, context)
        k2 = func(x0 + h * (k1 / 2), t0 + h / 2, context)
        k3 = func(x0 + h * (k2 / 2), t0 + h / 2, context)
        k4 = func(x0 + h * k3, t0 + h, context)
        xp = x0 + 1 / 6 * h * (k1 + 2 * k2 + 2 * k3 + k4)
        return xp
    
    acceleration = [0] * 3
    
    acceleration[0] = x
    callback(t, x)
    
    acceleration[1] = step_rk4(x, t, h, func, context)
    callback(t+h, acceleration[1])
    
    acceleration[2] = step_rk4(acceleration[1], t+h, h, func, context)
    callback(t+2*h, acceleration[2])
    
    x = step_rk4(acceleration[2], t+2*h, h, func, context)
    callback(t+3*h, x)
    
    t = t0 + 3*h
    
    for i in range(nsteps - 3):
        f_3 = func(acceleration[0], t - 3*h, context)
        f_2 = func(acceleration[1], t - 2*h, context)
        f_1 = func(acceleration[2], t - 1*h, context)
        f_0 = func(x, t, context)
        
        acceleration = acceleration[1:]
        acceleration.append(x)
        
        # Predictor
        x_next = x + h * (55 / 24 * f_0 - 59 / 24 * f_1 + 37 / 24 * f_2 - 3 / 8 * f_3)
        # Evaluator
        f_next = func(x_next, t + h, context)
        # Corrector
        x_next = x + h / 24 * (9 * f_next + 19 * f_0 - 5 * f_1 + f_2)
        # Evaluator
        f_next = func(x_next, t + h, context)
        # Corrector
        x_next = x + h / 24 * (9 * f_next + 19 * f_0 - 5 * f_1 + f_2)
        
        x = x_next
        t += h
        
        if x < 0 and trunc_zero:
            x = 0
        callback(t, x)
    return