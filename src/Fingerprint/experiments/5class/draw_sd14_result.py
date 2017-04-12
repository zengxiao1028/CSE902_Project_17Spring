
from experiments.util import *

def draw_confustion_matrix_net():
    plt.figure(figsize=(4.5, 4.5))
    print('sd14_5class_net')
    cnf_matrix = np.array([[ 559 ,   3 ,   1   , 2 ,   0],
                    [   3, 3315 ,  18   , 4  , 24],
                    [   0 ,  14, 3330 ,   5   , 9],
                    [   5  , 15  ,  5 , 436  ,  1],
                    [   0 ,  10 ,  31   , 0 ,3000]])
    class_names = ['Arch','Left','Right','Tented Arch', 'Whorl']
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion Matrix')

    #plt.show()
    plt.savefig('confusion_matrix_net_sd14.pdf')
    plt.close()

    compute_precision_recall_f1(cnf_matrix)

def draw_confustion_matrix_svm():
    plt.figure(figsize=(4.5, 4.5))
    print('sd14_5class_svm')
    cnf_matrix = np.array([[ 551  ,  3  ,  4  ,  7   , 0],
                        [   1 ,3321 ,  18 ,   6 ,  18],
                        [   0  , 11 ,3331  ,  6  , 10],
                        [   4  , 12  ,  6 , 439  ,  1],
                        [   0 ,  18  , 25   , 0 ,2998]])
    class_names = ['Arch','Left','Right','Tented Arch', 'Whorl']
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion Matrix')

    #plt.show()
    plt.savefig('confusion_matrix_svm_sd14.pdf')
    plt.close()

    compute_precision_recall_f1(cnf_matrix)



if __name__ == '__main__':
    draw_confustion_matrix_net()
    draw_confustion_matrix_svm()
