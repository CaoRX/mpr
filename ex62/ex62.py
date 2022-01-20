import numpy as np
import matplotlib.pyplot as plt

xLow = -5.0
xHigh = 5.0
xStep = 0.001
def plot(f):
    x = np.arange(xLow, xHigh, xStep)
    y = f(x)
    plt.plot(x, y, 'o-', markersize = 1)
    plt.savefig('mpr/ex62/func.eps')

eps = 1e-8

def gradientDescentOptimize(f, df, x0, eta):
    traj = []
    x = x0
    nextX = x - df(x) * eta
    while np.abs(x - nextX) > eps:
        traj.append(x)
        x = nextX
        nextX = x - df(x) * eta
    
    return x, f(x), traj

def geneticOptimize(f, x0, n = 100, m = 10, sigma = 0.01, sigmaDecreaseStep = None, totalStep = 10000):
    xs = [x0] * n
    traj = []
    for step in range(totalStep):
        traj.append(xs[0])
        if (sigmaDecreaseStep is not None) and ((step + 1) % sigmaDecreaseStep == 0):
            sigma *= 0.5
        
        fs = [f(x) for x in xs]

        idx = np.argsort(fs)
        
        newXs = [0.0] * n
        newXs[0] = xs[idx[0]]
        for i in range(1, n):
            newXs[i] = xs[idx[np.random.randint(m)]] + sigma * np.random.randn()

        xs = newXs

    # plt.plot(res)
    # plt.show()
    return xs[0], f(xs[0]), traj

def annealingOptimize(f, x0, eta, temp = 50.0, tempDecreaseStep = 1000, etaDecreaseStep = 10000, totalStep = 100000):
    traj = []

    x = x0
    for step in range(totalStep):
        traj.append(x)

        newX = 2 * (np.random.randint(2) - 0.5) * eta + x
        if (f(newX) < f(x)):
            x = newX
        elif np.random.rand() < np.exp((f(x) - f(newX)) / temp):
            x = newX

        if ((step + 1) % etaDecreaseStep == 0):
            eta *= 0.5
        if ((step + 1) % tempDecreaseStep == 0):
            temp *= 0.8

    # plt.plot(traj)
    # plt.show()
    return x, f(x), traj
        
if __name__ == '__main__':

    f = lambda x: x ** 4 - 10 * (x ** 2) - 3 * x
    df = lambda x: 4 * (x ** 3) - 20 * x - 3

    plot(f)


    optX, optY, traj = gradientDescentOptimize(f, df, x0 = -5, eta = 0.01)
    print('optX = {}, optY = {}'.format(optX, optY)) # local minimum
    optX, optY, _ = gradientDescentOptimize(f, df, x0 = 5, eta = 0.01)
    print('optX = {}, optY = {}'.format(optX, optY))

    aOptX, aOptY, aTraj = annealingOptimize(f, x0 = -5, eta = 0.1)
    print('simulated annealing optX = {}, optY = {}'.format(aOptX, aOptY))

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(traj, 'o', label = 'gradient descent', markersize = 1)
    ax1.set_xlabel('step')
    ax1.set_ylabel('x')
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.plot(aTraj, 'o', label = 'simulated annealing', markersize = 1)
    ax2.set_xlabel('step')
    ax2.set_ylabel('x')
    ax2.legend()

    plt.savefig('mpr/ex62/landscape.eps')
    plt.show()

    # geneticOptX, geneticOptY, _ = geneticOptimize(f, x0 = -5, sigma = 0.01)
    # print('genetic algorithm optX = {}, optY = {}'.format(geneticOptX, geneticOptY))
    # geneticOptX, geneticOptY, _ = geneticOptimize(f, x0 = 5, sigma = 0.01)
    # print('genetic algorithm optX = {}, optY = {}'.format(geneticOptX, geneticOptY))



    # plot(lambda x: x ** 4 - 10 * (x ** 2) - 3 * x)
