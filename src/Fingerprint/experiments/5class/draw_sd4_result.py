import matplotlib.pyplot as plt
import numpy as np
import itertools
from experiments.util import *

def draw_confustion_matrix_net():
    print('sd4_5class_net')
    plt.figure(figsize=(4.5, 4.5))
    cnf_matrix = np.array([[359  , 1  , 7 , 33  , 0],
                            [  4 ,370 ,  1,  24   ,1],
                            [  1 ,  0, 381 , 16  , 2],
                            [ 17 , 18 , 27 ,338 ,  0],
                            [  0  , 3  , 2  , 0 ,395]])
    class_names = ['Arch','Left','Right','Tented Arch', 'Whorl']
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion Matrix')

    #plt.show()
    plt.savefig('confusion_matrix_net_sd4.pdf')
    plt.close()

    compute_precision_recall_f1(cnf_matrix)

def draw_confustion_matrix_svm():
    print('sd4_5class_svm')

    plt.figure(figsize=(4.5, 4.5))
    cnf_matrix = np.array([[355 ,  0  , 3 , 42  , 0],
                            [  3 ,368 ,  0 , 28,   1],
                            [  0  , 0 ,374 , 25  , 1],
                            [ 10 , 11 , 16, 363 ,  0],
                            [  0  , 1,   4 ,  0 ,395]])
    class_names = ['Arch','Left','Right','Tented Arch', 'Whorl']
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion Matrix')

    #plt.show()
    plt.savefig('confusion_matrix_svm_sd4.pdf')
    plt.close()

    compute_precision_recall_f1(cnf_matrix)



if __name__ == '__main__':
    draw_confustion_matrix_net()
    draw_confustion_matrix_svm()
