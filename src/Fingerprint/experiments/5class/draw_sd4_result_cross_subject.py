import matplotlib.pyplot as plt
import numpy as np
import itertools
from experiments.util import *

def draw_confustion_matrix_net():
    print('sd4_5class_net')
    plt.figure(figsize=(4.5, 4.5))
    cnf_matrix = np.array([[377  , 5  , 7 , 30  , 1],
                            [  2 ,353 ,  0,  28   ,3],
                            [  1 ,  0, 355 , 32  , 4],
                            [ 22 , 27 , 28 ,321 ,  2],
                            [  0  , 5  , 6  , 0 ,391]])
    class_names = ['Arch','Left','Right','Tented Arch', 'Whorl']
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion Matrix')

    #plt.show()
    plt.savefig('confusion_matrix_net_sd4_cross_subject.pdf')
    plt.close()

    compute_precision_recall_f1(cnf_matrix)

def draw_confustion_matrix_svm():
    print('sd4_5class_svm')

    plt.figure(figsize=(4.5, 4.5))
    cnf_matrix = np.array([[395 ,  0  , 1 , 24  , 0],
                            [  4 ,356 ,  0 , 25,   1],
                            [  1  , 0 ,355 , 35  , 1],
                            [ 29 , 20 , 23, 328 ,  0],
                            [  1  , 6,   5 ,  0 ,390]])
    class_names = ['Arch','Left','Right','Tented Arch', 'Whorl']
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion Matrix')

    #plt.show()
    plt.savefig('confusion_matrix_svm_sd4_cross_subject.pdf')
    plt.close()

    compute_precision_recall_f1(cnf_matrix)



if __name__ == '__main__':
    draw_confustion_matrix_net()
    draw_confustion_matrix_svm()
