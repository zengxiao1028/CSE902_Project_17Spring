import cv2
import numpy as np
import os
import time
import project_config
def convert_label(x):
    return {'A':'0','L':'1','R':'2','T':'3','W':'4'}.get(x,'error')

def convert_gender(x):
    return {'F':'0','M':'1','f':'0','m':'1'}.get(x,'error')

def getDecodeFiles(dir, ext='.wsq'):
    decodeFiles = []
    for root,dirs,files in os.walk(dir):
        for file in files:
            if file.find(ext)<0:
                continue
            filepath = os.path.join(root,file)
            decodeFiles.append(filepath)
    return decodeFiles


def decodeRaw(files):
    counter = 1
    dwsq_path = '/home/xiao/programs/NBIS/bin/dwsq raw '
    for each in files:
        suffix = '-r'
        os.system(dwsq_path + each + ' ' + suffix)
        print( counter,each)
        counter+=1

def raw2png(root_dir):
    raw_files = getDecodeFiles(root_dir,'.raw')
    for idx,each in enumerate(raw_files):
        print(idx+1)
        ncm = each[:-3] + 'ncm'
        with open(ncm) as f:
            lines = f.readlines()
            finger_class = lines[3].split()[1]
            sex = lines[4].split()[1]
            width = lines[6].split()[1]
            height = lines[7].split()[1]

        with open(each, 'rb') as f:
            width = int(width)
            height = int(height)
            im = np.fromfile(f, dtype=np.uint8, count=width * height)
            im = im.reshape((height, width))  # notice row, column format
            name = each.split('/')[-1]
            png = name[:-4] + '_' + convert_label(finger_class) + '_' + '.png'
            cv2.imwrite(os.path.join(project_config.SD14_DATA_FOLDER,png), im)



def detect_errror(root_dir):

    raw_files = getDecodeFiles(root_dir, '.png')
    raw_files = [each for each in raw_files if each.find('error')>0]
    for each in raw_files:
        print (each)
    print (len(raw_files))


if __name__ == '__main__':
    sd14_path = '/data/xiao/fingerprint/sd14'


    #decodeRaw(getDecodeFiles(sd14_path))

    raw2png(sd14_path)

    #detect_errror(project_config.SD14_DATA_FOLDER)




