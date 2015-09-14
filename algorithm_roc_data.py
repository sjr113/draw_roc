__author__ = 'shen'

import csv
from numpy import *


# Load data_set from file named "file_name"
def load_data(file_name):
    num_feat = len(open(file_name).readline().split("\t"))
    data_mat = []
    label_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split("\t")
        for i in range(num_feat - 1):
            line_arr.append(float(cur_line[i]))
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat

from sklearn.neighbors import KNeighborsClassifier


def knn_classify(train_data, train_label, test_data, num):
    m, n = shape(test_data)
    test_label = zeros([m+1, num])
    test_label[0, :] = range(1, num+1)
    for x in range(1, num+1, 1):
        knn_clf = KNeighborsClassifier(n_neighbors=x)
        knn_clf.fit(train_data, ravel(train_label))
        test_label[1:, x-1] = knn_clf.predict(test_data)  # the shape of test_label[:, x-1] is [1,68]
    save_result(test_label, "sk_learn_knn_Result.csv")
    return test_label


from sklearn import svm


# Use svm to predict test_label
# Default:C=1.0, kernel = 'rbf'. you can try kernel:"linear", "poly", "rbf", "sigmoid", "precomputed"
def svm_classify(train_data, train_label, test_data, num):
    kernel_svm = ["linear", "poly", "rbf"]
    number_kernel = len(kernel_svm)
    m, n = shape(test_data)
    test_label = zeros([m+1, num])
    test_label[0, :] = range(1, num+1)  # parameter of C
    for x in range(1, num+1, 1):
        svm_clf = svm.SVC(C=x*0.2, kernel=kernel_svm[mod(x, number_kernel)])
        svm_clf.fit(train_data, ravel(train_label))
        test_label[1:, x-1] = svm_clf.predict(test_data)
    save_result(test_label, 'sk_learn_SVM_C=5.0_Result.csv')
    return test_label


from sklearn.naive_bayes import MultinomialNB


# Default alpha=1.0,Setting alpha = 1 is called Laplace smoothing, while alpha < 1 is called Lidstone smoothing.
def bayes_multinomial_classify(train_data, train_label, test_data, num):
    m, n = shape(test_data)
    test_label = zeros([m+1, num])
    test_label[0, :] = range(1, num+1, 1)  # parameter of C
    for x in range(1, num+1, 1):
        bayes_m_clf = MultinomialNB(alpha=x/20, class_prior=None)
        bayes_m_clf.fit(train_data, ravel(train_label))
        test_label[1:, x-1] = bayes_m_clf.predict(test_data)
    save_result(test_label, 'sk_learn_multinomial_Result.csv')
    return test_label


def save_result(result, csv_name):
    with open(csv_name, 'wb') as myFile:
        my_writer = csv.writer(myFile)
        my_writer.writerows(result)

# myFile.close()


def data_recognition(method, train_data, train_label, test_data, test_given_label, num):
    test_given_label = mat(test_given_label)
    if method == "KNN":
        result = knn_classify(train_data, train_label, test_data, num)
    elif method == "SVM":
        result = svm_classify(train_data, train_label, test_data, num)
    elif method == "multinomial":
        result = bayes_multinomial_classify(train_data, train_label, test_data, num)
    m, n = shape(test_data)

    Roc_array = zeros([4, num])
# Roc_array[0, :] means true_positive
# Roc_array[1, :] means false_positive
# Roc_array[2, :] means false_negatives
# Roc_array[3, :] means true_negatives

    for j in range(1, num+1, 1):
        for i in xrange(m-1):
            if result[i, j-1] != test_given_label[0, i] and test_given_label[0, i] == 1:
                Roc_array[1, j-1] += 1
            elif result[i, j-1] != test_given_label[0, i] and test_given_label[0, i] == -1:
                Roc_array[2, j-1] += 1
            elif result[i, j-1] == test_given_label[0, i] and test_given_label[0, i] == 1:
                Roc_array[0, j-1] += 1
            else:
                Roc_array[3, j-1] += 1

    return Roc_array


train_data_mat, train_label_mat = load_data('horseColicTraining2.txt')
# print shape(train_data_mat), shape(train_label_mat)

test_data_mat, test_label_mat = load_data('horseColicTest2.txt')
# print shape(test_data_mat), shape(test_label_mat)

num = 20
Roc_array = data_recognition("multinomial", train_data_mat, train_label_mat, test_data_mat, test_label_mat, num)
# print Roc_array

