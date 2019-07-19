import numpy as np

# x is the value of inputs
x = np.array([[-1, 1], [0, -1], [10, 1]])

# w is the value of weights, with the first element being the bias score
w = np.array([13, 4, 0])

# y is the training results
y = np.array([1, -1, 1])

def learningAlgorithm(x, w, y):
    rerun = False
    for i in range(0, len(x)):
        if not testPointPasses(w, x[i], y[i]):
            rerun = True
            w = updateWeight(w, x[i], y[i])
    if rerun:
        return learningAlgorithm(x, w, y)
    else:
        print(w)

def testPointPasses(w, xi, yi):
    # sign(y<i>) == sign(x<i> dot w<i> + b)
    return np.sign(yi) == np.sign(xi.dot(w[1:]) + w[0])

def updateWeight(w, xi, yi):
    # w<k+1> = w<k> + y<i> * x<i>
    # trivia: outliers cause updates to overreact, and take longer to converge
    return w + yi * np.insert(xi, 0, 1)

learningAlgorithm(x, w, y)

### Good work boy! Answer was sum(3, 7, 5) = 15
### This is a perception that updates its weights to classify points