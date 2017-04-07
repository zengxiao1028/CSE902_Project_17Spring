import cv2
import os
import project_config
import numpy as np
import random
import RAM.config as cfg
#convert raw data pngs to small size pngs with labels in their filenames
def load(src_folder):

    files = os.listdir(src_folder)

    files = [each for each in files if each.endswith('png')]

    files = sorted(files)
    random.seed(1024)
    random.shuffle(files)
    split_point = int(len(files) / 10)

    X_train_files = files[split_point:]
    X_test_files = files[:split_point]

    y_train = [int(each.split('_')[3][0]) for each in X_train_files]
    y_test = [int(each.split('_')[3][0]) for each in X_test_files]

    X_train = []
    X_test = []
    c = cfg.Config()
    for each in X_train_files:
        img_path = os.path.join(src_folder,each)
        #cv2
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        #resize
        img = cv2.resize(img, (c.original_size, c.original_size))
        # cv2.imshow('',img)
        # cv2.waitKey()
        X_train.append(img)


    for each in X_test_files:
        img_path = os.path.join(src_folder,each)
        #cv2
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        #resize
        img = cv2.resize(img, (c.original_size, c.original_size))

        X_test.append(img)

    return (np.expand_dims(np.array(X_train),axis=3), np.array(y_train)), (np.expand_dims(np.array(X_test),axis=3), np.array( y_test))



if __name__ == '__main__':

    (X_train, y_train), (X_test, y_test) = load(project_config.SD4_DATA_FOLDER)
    print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)