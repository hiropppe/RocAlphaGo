import h5py as h5
import numpy as np

from tqdm import tqdm


n_feature = 170
#n_feature = 90
#n_feature = 10
K = 5
lr = 0.001
iter_num = 1000
test_size = .2


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

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
    #W = 0.01 * np.random.randn(n_feature)
    rgen = np.random.RandomState(1)
    W = rgen.normal(loc=0.0, scale=0.01, size=n_feature)

    dataset = h5.File('./rollout_features.hdf5')

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
            #import pdb; pdb.set_trace()
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
