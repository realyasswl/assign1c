# Source: http://neuralnetworksanddeeplearning.com/chap1.html
# You need only train and test sets. Simply disregard the validation set.
import numpy as np
import cPickle
import gzip
import datetime


def load_data():
    file_location = '/run/media/liwang/Other/STUDY/Y1Q3/fdm/1454491793_864__data1a/data1a/mnist.pkl.gz';
    f = gzip.open(file_location, 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return training_data, validation_data, test_data


# don't need function vectorized_result if we don't need the pre-process


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    # don't really need the pre-process

    # normalize the training data
    norm_tr_d = ([t / np.reshape(np.linalg.norm(t), (-1, 1)) for t in tr_d[0]],
                 tr_d[1])
    # normalize the test data
    norm_te_d = ([v / np.reshape(np.linalg.norm(v), (-1, 1)) for v in te_d[0]],
                 te_d[1])
    # array to store successful results
    succ_list = [];
    # array to store failure results
    fail_list = [];
    start_time = datetime.datetime.now()
    print("starts at:", start_time)
    # the double loop statement, outer loop is for each data in test dataset
    for d in range(len(norm_te_d[0])):
        if d % 100 == 0:
            print(datetime.datetime.now(), d)
            print("succ:", len(succ_list))
            print("fail:", len(fail_list))
        # variable to store information for each test data, the first element is the max dot value,
        # the second value is the corresponding estimated result
        nearest = (0, 0)
        # inner loop is for each data in training dataset
        for k in range(len(norm_tr_d[0])):
            # compute dot value and always put the pair with the larger dot value into nearest
            dot = np.dot(norm_te_d[0][d][0], norm_tr_d[0][k][0])
            if nearest[0] < dot:
                nearest = (dot, norm_tr_d[1][k])
        # compare the estimated value and the true value
        if nearest[1] == norm_te_d[1][d]:
            succ_list.append((nearest, norm_te_d[1][d]))
        else:
            fail_list.append((nearest, norm_te_d[1][d]))
    end_time = datetime.datetime.now()
    print(end_time - start_time)
    # we simply output the numbers of successful and failed results, with this two arrays,
    # we can do statistical analysis such as which number has the most failure number or the highest failure rate.
    print("succ:", len(succ_list))
    print("fail:", len(fail_list))
    return succ_list, fail_list


if __name__ == "__main__":
    load_data_wrapper();
