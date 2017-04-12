import matplotlib.pyplot as plt
import numpy as np
import itertools

def compute_precision_recall_f1(cnf_matrix):
    np.set_printoptions(precision=4)
    print('accuracy', np.sum(np.diagonal(cnf_matrix))*1.0/np.sum(cnf_matrix))
    precision = np.divide(cnf_matrix.diagonal(),cnf_matrix.sum(0))
    print('precision',precision)

    recall = np.divide(cnf_matrix.diagonal(), cnf_matrix.sum(1))
    print('recall', recall)

    f1 = 2 * np.divide(np.multiply(precision,recall),precision+recall)
    print('f1',f1)

    total = np.array([precision,recall,f1])
    mean = np.mean(total,axis=1)
    print('mean',mean)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes,rotation=35)
    plt.yticks(tick_marks, classes)


    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",fontsize=12)

    plt.tight_layout()
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')