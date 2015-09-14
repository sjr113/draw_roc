__author__ = 'shen'


import os
from numpy import *


# Load data_set from file named "file_name"
def load_data(file_name):
    tuple2 = os.path.splitext(file_name)
    data = []
    if tuple2[1] == ".txt":
        num_feat = len(open(file_name).readline().split("\t"))
        fr = open(file_name)
        for line in fr.readlines():
            line_arr = []
            cur_line = line.strip().split("\t")
            for i in range(num_feat):
                line_arr.append(float(cur_line[i]))
            data.append(line_arr)
    elif tuple2[1] == ".csv":
        num_feat = len(open(file_name).readline().split(","))
        fr = open(file_name)
        for line in fr.readlines():
            line_arr = []
            cur_line = line.strip().split(",")
            for i in range(num_feat):
                line_arr.append(float(cur_line[i]))
            data.append(line_arr)
    return data


# data_mat, label_mat = load_data('horseColicTraining2.txt')


def draw_roc(file1_name, file2_name):
    data = load_data(file1_name)
    data_mat = array(data)
    parameter = data_mat[0, :]   # size = n, 1
    result = data_mat[1:, :]
    data2 = load_data(file2_name)
    data2_mat = array(data2)
    test_given_label2 = data2_mat[:, -1]
    m, n = shape(result)
    test_given_label = test_given_label2.reshape([1, m])
    Roc_array = zeros([4, n])
# Roc_array[0, :] means true_positive
# Roc_array[1, :] means false_positive
# Roc_array[2, :] means false_negatives
# Roc_array[3, :] means true_negatives
    for j in range(1, n+1, 1):
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


import matplotlib.pyplot as plt


def draw2_roc(RoC_array, par="number", font_family="serif", font_color="blue", font_weight="normal", font_size=16):
    font = dict(family=font_family, color=font_color, weight=font_weight, size=font_size)
# Accuracy : the proportion of true result among the total number of cases examined
# accuracy = (true_positive + true_negatives)/(true_positive + false_positive + false_negatives + true_negatives)
# accuracy_array = (RoC_array[0, :, i] + RoC_array[3, :, i])/(RoC_array[0, :, i] + RoC_array[1, :, i]
#  + RoC_array[2, :, i] + RoC_array[3, :, i])
# Precision : the proportion of the true positive against all the positive results
# precision = true_positive/(true_positive + false_positive)

# Recall : the proportion of the true positives against the true positives and false negatives
# recall = true_positive/(true_positive + false_negatives)
# recall_array = RoC_array[0, :, i]/(RoC_array[0, :, i] + RoC_array[2, :, i])
# False Alarm rate (False positive ): the proportion of the false positive against the false positives
# false_alarm_rate = false_positive/(false_positive + true_negatives)

    line_style = ["r--", "g^-", "bo-", "k--", "rs-", "ko-", "r^-"]
    line_width = [.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    num = len(line_style)
    plt.figure(1)
    i = 0
    if par == "rate":
        while i < (shape(RoC_array))[0]:
            roc_data = array(RoC_array[i])
            recall_array = (roc_data[0, :]*1.0)/(roc_data[0, :] + roc_data[2, :] + 1e-6)
            false_alarm_rate_array = (roc_data[1, :]*1.0)/(roc_data[1, :] + roc_data[3, :] + 1e-6)
            argsort_array = argsort(false_alarm_rate_array)
            plt.plot(false_alarm_rate_array[argsort_array], recall_array[argsort_array], line_style[mod(i, num)], linewidth=line_width[mod(i, num)])
            i += 1
        plt.xlabel('false_alarm_rate', fontdict=font)
        plt.ylabel('recall_rate', fontdict=font)
    elif par == "number":
        while i < (shape(RoC_array))[0]:
            roc_data = array(RoC_array[i])
            argsort_array = argsort(roc_data[1, :])
            plt.plot((roc_data[1, :])[argsort_array], (roc_data[0, :])[argsort_array], line_style[mod(i, num)], linewidth=line_width[mod(i, num)])
            i += 1
        plt.xlabel('false_alarm', fontdict=font)
        plt.ylabel('precision', fontdict=font)
    elif par == "log":
        while i < (shape(RoC_array))[0]:
            roc_data = array(RoC_array[i])
            argsort_array = argsort(roc_data[1, :])
            plt.plot(log((roc_data[1, :])[argsort_array]+ 1e-6), log((roc_data[0, :])[argsort_array]+ 1e-6), line_style[mod(i, num)], linewidth=line_width[mod(i, num)])
            i += 1
        plt.xlabel('false_alarm_log', fontdict=font)
        plt.ylabel('precision_log', fontdict=font)
    plt.title("RoC Curve", fontdict=font)
    plt.text(2, 0.65, "RoC Lines", fontdict=font)

    plt.show()

# def f(first_file_name, second_file_name):
# f_data, f_label = load_data(first_file_name)
# s_data, s_label = load_data(second_file_name)

Roc_array1 = draw_roc("sk_learn_knn_Result.csv", "horseColicTest2.txt")
Roc_array2 = draw_roc("sk_learn_SVM_Result.csv", "horseColicTest2.txt")
Roc_array3 = draw_roc("sk_learn_multinomial_Result.csv", "horseColicTest2.txt")
Roc_array = []
msize, nsize = shape(Roc_array1)
Roc_array.append(Roc_array1)
Roc_array.append(Roc_array2)
Roc_array.append(Roc_array3)
# temp = concatenate((Roc_array1[:, :, newaxis], Roc_array2[:, :, newaxis]), axis=0)  # stack array
# Roc_array = concatenate((temp[:, :, :], Roc_array3[:, :, newaxis]), axis=0)  # stack array

Roc_array = array(Roc_array)
print Roc_array
draw2_roc(Roc_array, par="rate")