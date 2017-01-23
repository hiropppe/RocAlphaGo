import h5py as h5
import numpy as np
import time

from tqdm import tqdm


n_feature = 170
data_file = './rollout.hdf5'

# response
# n_feature = 1
# data_file = './response_d.hdf5'

# save-atari
# n_feature = 1
# data_file = './save_atari_d.hdf5'

# neighbour
# n_feature = 8
# data_file = './neighbour_d.hdf5'

# one-hot color pattern
# n_feature = 48
# data_file = './response_pattern2_d.hdf5'
# n_feature = 32
# data_file = './non_response_pattern2_d.hdf5'

# one-hot color and liberty pattern
# n_feature = 96
# data_file = './response_pattern3_d.hdf5'
# n_feature = 64
# data_file = './non_response_pattern3_d.hdf5'

lr = 0.001
iter_num = 1
test_size = .2


def softmax(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


def run_training():
    # W = 0.01 * np.random.randn(n_feature)
    rgen = np.random.RandomState(1)
    W = rgen.normal(loc=0.0, scale=0.01, size=n_feature)

    dataset = h5.File(data_file)

    states = dataset['states']
    actions = dataset['actions']

    board_size = (int)(np.sqrt(len(states[0])))

    n_total = len(states)
    n_test = (int)(n_total*test_size)
    n_train = n_total - n_test

    train_acc_list, train_loss_list, test_acc_list = [], [], []
    for i in range(iter_num):
        n_train_acc, n_train_total_loss = 0, 0.
        n_test_acc = 0

        indices = np.random.permutation(n_total)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        # train
        for j in tqdm(range(n_train)):
            # import pdb; pdb.set_trace()
            X = states[train_indices[j]]

            # one-hot
            t = np.zeros(board_size**2)
            a = actions[train_indices[j]]
            t[a[0]*board_size+a[1]] = 1
 
            y = softmax(np.dot(X, W))
            loss = cross_entropy_error(y, t)

            dy = y - t
            grad = np.dot(X.T, dy)

            W -= lr*grad

            n_train_acc += t[np.argmax(y)]
            n_train_total_loss += loss

        train_acc_list.append(n_train_acc*100/n_train)
        train_loss_list.append(n_train_total_loss/n_train)

        # test
        for j in range(n_test):
            X = states[test_indices[j]]

            # one-hot
            t = np.zeros(board_size**2)
            a = actions[test_indices[j]]
            t[a[0]*board_size+a[1]] = 1

            y = softmax(np.dot(X, W))

            n_test_acc += t[np.argmax(y)]

        test_acc_list.append(n_test_acc*100/n_test)

        print('Acc. {:.3f} ({:.0f}/{:.0f}) Loss. {:.3f} Val Acc. {:.3f} ({:.0f}/{:.0f})' \
              .format(train_acc_list[-1],
                      n_train_acc,
                      n_train,
                      train_loss_list[-1],
                      test_acc_list[-1],
                      n_test_acc,
                      n_test))


if __name__ == '__main__':
    run_training()
