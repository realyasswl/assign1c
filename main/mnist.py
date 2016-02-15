__author__ = 'liwang'
# Source: http://neuralnetworksanddeeplearning.com/chap1.html
# You need only train and test sets. Simply disregard the validation set.
import numpy as np
import cPickle
# import cPickle
import gzip


def load_data():
    file_location='/run/media/liwang/Other/STUDY/Y1Q3/fdm/1454491793_864__data1a/data1a/mnist.pkl.gz';
    f = gzip.open(file_location, 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return training_data, validation_data, test_data


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()

    # for k in range(len(tr_d[0])):
    #     print("k=",k,tr_d[0][k],tr_d[1][k])

    # print(tr_d[0][0])

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
# normalize
    norm_tr_d=([t/np.reshape(np.linalg.norm(t),(-1,1)) for t in tr_d[0]],
              tr_d[1])

    norm_te_d=([v/np.reshape(np.linalg.norm(v),(-1,1)) for v in te_d[0]],
              te_d[1])

    succ_count=0;
    for d in range(len(norm_te_d[0])):
        nearest=(0,0)
        for k in range(len(norm_tr_d[0])):
            dot=np.dot(norm_te_d[0][d][0],norm_tr_d[0][k][0])
            if nearest[0]<dot:
                nearest=(dot,norm_tr_d[1][k])
        # print(nearest,norm_te_d[1][d])
        if(nearest[1]==norm_te_d[1][d]):
            succ_count+=1
    # print(succ_count)

    return training_data, validation_data, test_data

if __name__ == "__main__":
    load_data_wrapper();