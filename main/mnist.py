# Source: http://neuralnetworksanddeeplearning.com/chap1.html
# You need only train and test sets. Simply disregard the validation set.
import numpy as np
import cPickle
import gzip
import datetime


def load_data():
    file_location = '../mnist.pkl.gz'
    f = gzip.open(file_location, 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return training_data, validation_data, test_data


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()

# normalize the training data
    norm_tr_d = ([t / np.reshape(np.linalg.norm(t), (-1, 1))
                  for t in tr_d[0]], tr_d[1])
# normalize the test data
    norm_te_d = ([v / np.reshape(np.linalg.norm(v), (-1, 1))
                  for v in te_d[0]], te_d[1])
# array to store successful results
    succ_list = []
# array to store failure results
    fail_list = []
    start_time = datetime.datetime.now()
# outer loop is for each data in test dataset
    for d in xrange(len(norm_te_d[0])):
        if d % 100 == 0:
            print(datetime.datetime.now())
            print(len(succ_list))
            print(len(fail_list))
# variable to store information for each test data,
# the first element is the max dot value,
# the second element is the corresponding estimated result
        nearest = (0, 0)
# inner loop is for each data in training dataset
        for k in xrange(len(norm_tr_d[0])):
# compute dot value
# always put the pair with the larger dot value into nearest
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
    print("succ:", len(succ_list))
    print("fail:", len(fail_list))
    return succ_list, fail_list


# here we compute confusion matrix
def compute_matrix(succ, fail):
    matrix = [[0 for x in xrange(11)] for x in xrange(11)]
    for s in succ:
# true value and estimated value are the same
        matrix[s[1]][s[1]] += 1
    for f in fail:
# put true value in row and estimated value in column
        matrix[f[1]][f[0][1]] += 1
# calculate the precision and recall
    correct_count = 0
    for i in xrange(10):
        matrix[i][10] = round(float(matrix[i][i]) / sum(matrix[i]), 3)
        matrix[10][i] = round(float(matrix[i][i]) / sum((zip(*matrix)[i])), 3)
        correct_count += matrix[i][i]
    accuracy = float(correct_count)/10000
    print("accuracy=", accuracy)
    for e in matrix:
        print ('|'.join(['%5s' % (e[n]) for n in xrange(len(e))]))


if __name__ == "__main__":
    load=False
    if load:
        load_data_wrapper()
    else:
        succfileloc = '../succ_list'
        failfileloc = '../fail_list'
        f1 = open(succfileloc, 'rb')
        f2 = open(failfileloc, 'rb')
        succ = cPickle.load(f1)
        fail = cPickle.load(f2)
        f1.close()
        f2.close()
        if succ is None or fail is None:
            succ, fail = load_data_wrapper()
            f1 = open(succfileloc, 'wb')
            f2 = open(failfileloc, 'wb')
            cPickle.dump(succ, f1)
            cPickle.dump(fail, f2)
            f1.close()
            f2.close()
        compute_matrix(succ, fail)