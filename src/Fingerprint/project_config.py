import os
IMG_SIZE = 224
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

RAWDATA_FOLDER = '/data/xiao/fingerprint/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/'
DATA_FOLDER = os.path.join(ROOT_DIR, 'fingerprint_data/data')
DATA_ORIGIN_FOLDER = os.path.join(ROOT_DIR, 'fingerprint_data/data_origin')
SP_LABEL_FOLDER = os.path.join(ROOT_DIR,'fingerprint_data/sp_label')
SP_DATA_FOLDER = os.path.join(ROOT_DIR,'fingerprint_data/sp_data')
