import tensorflow as tf, shutil
import cv2
import os
import project_config

def convert_label(x):
    return {'A':'0','L':'1','R':'2','T':'3','W':'4'}.get(x,'error')

def convert_gender(x):
    return {'F':'0','M':'1'}.get(x,'error')

#convert raw data pngs to small size pngs with labels in their filenames
def convert(src_folder, dst_folder):

   files = os.listdir(src_folder)

   files = [each for each in files if each.endswith('png')]

   for each in files:

        img_path = os.path.join(src_folder,each)

        #cv2
        img = cv2.imread(img_path)

        #resize
        img = cv2.resize(img,(project_config.IMG_SIZE,project_config.IMG_SIZE),cv2.IMREAD_GRAYSCALE)

        #read_label
        file_path = os.path.join(src_folder, each.replace('png','txt'))
        with open(file_path) as f:
            gender = f.readline().split(':')[1]
            if len(gender) != 3:
                print(gender,len(gender))
            gender = convert_gender(gender[-2])

            label = f.readline().split(':')[1]
            if len(label) != 3:
                print(label,len(label))
            label = convert_label(label[-2])

        #destination path
        #
        # new file is named as follows: 'filename_finger_gender_label.png'
        #
        new_img_name = each[:-4] + '_' + gender + '_' + label + '.png'
        dst_path = os.path.join(dst_folder, new_img_name)
        cv2.imwrite(dst_path, img)

def convert_original_size(src_folder, dst_folder):
    files = os.listdir(src_folder)

    files = [each for each in files if each.endswith('png')]

    for each in files:

        img_path = os.path.join(src_folder, each)

        # cv2
        img = cv2.imread(img_path)

        # read_label
        file_path = os.path.join(src_folder, each.replace('png', 'txt'))
        with open(file_path) as f:
            gender = f.readline().split(':')[1]
            if len(gender) != 3:
                print(gender, len(gender))
            gender = convert_gender(gender[-2])

            label = f.readline().split(':')[1]
            if len(label) != 3:
                print(label, len(label))
            label = convert_label(label[-2])

        # destination path
        #
        # new file is named as follows: 'filename_finger_gender_label.png'
        #
        new_img_name = each[:-4] + '_' + gender + '_' + label + '.png'
        dst_path = os.path.join(dst_folder, new_img_name)
        cv2.imwrite(dst_path, img)


if __name__ == '__main__':

    for i in range(8):
        scr_folder = os.path.join(project_config.RAWDATA_FOLDER, 'figs_'+str(i))
        convert_original_size(scr_folder,project_config.DES_ORIGIN_FOLDER)

