import numpy as np
import scipy as sp

import tables

from scipy.sparse import csr_matrix

from tqdm import tqdm


# n_feature = 182790
# data_file = './sparse_roll_feat.hdf5'

# response + save_atari + neighbour + pattern
# n_feature = 10
# data_file = './roll_feat1.hdf5'

# response + save_atari + neighbour + pattern2
# n_feature = 90
# data_file = './roll_feat3.hdf5'

# response + save_atari + neighbour + pattern3
# n_feature = 170
# data_file = './roll_feat3.hdf5'

# response
# n_feature = 1
# data_file = './roll_feat_response.hdf5'

# save-atari
# n_feature = 1
# data_file = './roll_feat_save_atari.hdf5'

# neighbour
# n_feature = 8
# data_file = './roll_feat_neighbour.hdf5'

# one-hot color pattern
n_feature = 175689 + 1
data_file = './sparse_roll_feat_non_response_pattern.hdf5'
# n_feature = 7090 + 1
# data_file = './sparse_roll_feat_response_pattern.hdf5'


lr = 0.001
iter_num = 1
test_size = .2

board_size = 19


def softmax(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    return -np.sum(t * np.log(y))


def get_data_path(X_i, group_size):
    group_name = 'g' + str(X_i/group_size).rjust(5, '0')
    state_name = 's' + str(X_i).rjust(8, '0')
    data_path = group_name + '/' + state_name
    return data_path


def run_training():
    rgen = np.random.RandomState(1)
    W = rgen.normal(loc=0.0, scale=0.01, size=n_feature)
    W = csr_matrix(W).T

    h5 = tables.open_file(data_file, 'r')
    root = h5.root
    # group_size = h5.get_node_attr(root, 'group_size')
    group_size = 100

    n_total = h5.get_node_attr(root, 'size')
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
            X_i = train_indices[j]

            data_path = get_data_path(X_i, group_size)

            csr_data = getattr(root, 'state/data/' + data_path).read()
            csr_indices = getattr(root, 'state/indices/' + data_path).read()
            csr_indptr = getattr(root, 'state/indptr/' + data_path).read()
            action = getattr(root, 'action/' + data_path).read()

            X = csr_matrix((csr_data, csr_indices, csr_indptr), shape=(board_size**2, n_feature))

            # one-hot
            t = np.zeros((board_size**2, 1))
            t[action[0]*board_size+action[1]] = 1

            y = softmax(sp.dot(X, W).toarray())
            loss = cross_entropy_error(y, t)

            dy = y - t
            grad = sp.dot(X.T, csr_matrix(dy))

            W -= lr*grad

            n_train_acc += t[np.argmax(y)][0]
            n_train_total_loss += loss

        train_acc_list.append(n_train_acc*100/n_train)
        train_loss_list.append(n_train_total_loss/n_train)

        # test
        for j in range(n_test):
            X_i = test_indices[j]

            data_path = get_data_path(X_i, group_size)

            csr_data = getattr(root, 'state/data/' + data_path).read()
            csr_indices = getattr(root, 'state/indices/' + data_path).read()
            csr_indptr = getattr(root, 'state/indptr/' + data_path).read()
            action = getattr(root, 'action/' + data_path).read()

            X = csr_matrix((csr_data, csr_indices, csr_indptr), shape=(board_size**2, n_feature))

            # one-hot
            t = np.zeros((board_size**2, 1))
            t[action[0]*board_size+action[1]] = 1

            y = softmax(sp.dot(X, W).toarray())

            n_test_acc += t[np.argmax(y)][0]

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
