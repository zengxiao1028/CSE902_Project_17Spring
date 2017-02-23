import os
import numpy as np
import cv2



def gen_label():
    sp_dict = dict()

    with open('SD4_SPs.txt') as f:
        lines = f.readlines()

    for each in lines:

        strs = each.split()
        image_id = strs[0]
        x_coor = int(strs[1])
        y_coor = int(strs[2])
        point_type = strs[3]
        fingerprint_type = strs[4]

        if image_id not in sp_dict:
            sp_dict[image_id] = [(x_coor, y_coor, point_type)]
        else:
            sp_dict[image_id].append((x_coor, y_coor, point_type))

    #print(sp_dict)
    return sp_dict


if __name__ == '__main__':
    path = 'data'
    files = os.listdir(path)
    d = gen_label()

    for file_name in files:
        img = cv2.imread(os.path.join(path,'each'))
        key = file_name[:8]
        points = d[key]

        for p in points:

            img = cv2.rectangle(img, (p[0]-5, p[1]-5), (p[0]+5, p[1]+5), (255, 0, 0), 2)
            cv2.imshow('img',img)
            cv2.waitKey()


