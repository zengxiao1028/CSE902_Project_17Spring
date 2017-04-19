import matplotlib.pyplot as plt
import numpy as np
import itertools
from experiments.util import *


def draw_confustion_matrix_svm():
    plt.figure(figsize=(4.5, 4.5))
    print('sd4_4class_svm')
    cnf_matrix = np.array([[779 , 22,  19 ,  0],
                            [ 31 ,354  , 0  , 1],
                            [ 36 ,  0, 355 ,  1],
                            [  3  , 5 , 4 ,390]])
    class_names = ['Arch','Left','Right','Whorl']
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion Matrix')

    #plt.show()
    plt.savefig('confusion_matrix_svm_sd4_4class_cross_subject.pdf')
    plt.close()

    compute_precision_recall_f1(cnf_matrix)



if __name__ == '__main__':

    draw_confustion_matrix_svm()
