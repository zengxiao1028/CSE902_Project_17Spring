import matplotlib.pyplot as plt
import numpy as np
import itertools
from experiments.util import *


def draw_confustion_matrix_svm():
    plt.figure(figsize=(4.5, 4.5))
    print('sd14_4class_svm')
    cnf_matrix = np.array([[1010  , 11  ,  5   , 1],
                             [  11 ,3316  , 19  , 18],
                             [   6 , 11 ,3331 ,  10],
                             [   0  , 18 , 25, 2998]])
    class_names = ['Arch','Left','Right','Whorl']
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion Matrix')

    #plt.show()
    plt.savefig('confusion_matrix_svm_sd14_4class.pdf')
    plt.close()

    compute_precision_recall_f1(cnf_matrix)



if __name__ == '__main__':

    draw_confustion_matrix_svm()
