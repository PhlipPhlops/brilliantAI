import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import interactive
import keras
from keras.datasets import mnist

# increase max line length of numpy prints
np.set_printoptions(linewidth=1000)

# input image dimensions
img_rows, img_cols = 28, 28

# data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# grab data
test_data = np.array(x_train[20])
# change data from 1d array to 28x28 matrix
test_data = np.reshape(test_data, (img_rows, img_cols))
print(test_data)

def plot_matrix(matrix):
    plt.imshow(matrix)
    plt.show(block=False)
    raw_input()
    plt.close()

'''
Returns an array of submatrices of size (n_width)x(n_width) of the given matrix.
@param matrix the matrix to be subsectionalized
@param n_width the width and height of the submatrices
@param stride the size of the jumps per iteration
@param padding <unused> additional padding added around the image
'''
def stride_matrix(matrix, n_width, stride=1, padding=0):
    # coords of the top left corner of the submatrix
    curr_x, curr_y = 0, 0

    # calculate length of submatrices array
    width, height = matrix.shape[0], matrix.shape[1]
    num_submatrices = ((width - n_width)/stride + 1)*((height - n_width)/stride + 1)

    # the array of matrices to be returned
    submatrices = np.empty((num_submatrices, n_width, n_width))

    def grab_submatrix(x, y, n):
        # init empty matrix to be filled
        sub_matrix = np.empty((n_width, n_width))
        for row in range(0, n_width):
            sub_matrix[row] = matrix[row + y][x:x + n]
        return sub_matrix

    # iterate over every position and add submatrix to submatrices
    i = 0
    while curr_y + n_width <= matrix.shape[1]:
        while curr_x + n_width <= matrix.shape[0]:
            submatrices[i] = grab_submatrix(curr_x, curr_y, n_width)
            curr_x += stride
            i += 1
        curr_x = 0
        curr_y += stride

    return submatrices

'''
Uses max pooling to return a matrix smaller than the original, its entries are the maximum
value entry of the submatrices of size (n_width)x(n_width) collected by a stride of 2
@param matrix the matrix to be pooled
@param n_width size of submatrices to pool. Convention sets this to either 2 or 3, any more
    loses too much information to be useful
@param stride step width to iterate over matrix over. Convention sets this to 2
'''
def downsample_matrix(matrix, n_width=2, stride=2):
    # get array of matrices to reduce into new matrix
    submatrices = stride_matrix(matrix, n_width, stride)
    
    # set dimensions of new matrix
    width, height = matrix.shape[0], matrix.shape[1]
    pooled_matrix = np.empty((width/n_width, height/n_width))

    def get_max_value(matrix):
        curr_max = 0
        for entry in np.nditer(matrix):
            if entry > curr_max:
                curr_max = entry
        return curr_max

    # populated pooled_matrix with max value of submatrices
    pool_rows, pool_cols = pooled_matrix.shape[0], pooled_matrix.shape[1]
    i = 0
    for x in range(0, pool_rows):
        for y in range(0, pool_cols):
            pooled_matrix[x, y] = get_max_value(submatrices[i])
            i += 1
    
    return pooled_matrix

downsampled = downsample_matrix(test_data)
subslicies = stride_matrix(downsampled, 7, 7)
print(subslicies)