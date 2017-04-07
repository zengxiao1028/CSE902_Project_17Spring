import os



ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

RAWDATA_FOLDER = '/data/xiao/fingerprint/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/'



SD4_DATA_ORIGIN_FOLDER = os.path.join(ROOT_DIR, 'fingerprint_data/sd4/data_origin')
SP_LABEL_FOLDER = os.path.join(ROOT_DIR,'fingerprint_data/sd4/sp_label')
SP_DATA_FOLDER = os.path.join(ROOT_DIR,'fingerprint_data/sd4/sp_data')

SD4_DATA_FOLDER = os.path.join(ROOT_DIR, 'fingerprint_data/sd4/data')
SD4_INPUT_IMG_SIZE = 224
SD4_TRANSFORMED_IMG_SIZE = 224
SD4_SPS_PATH = os.path.join(ROOT_DIR,'fingerprint_data/sd4/SD4_SPs.txt')


SD14_INPUT_IMG_SIZE = 512
SD14_TRANSFORMED_IMG_SIZE = 512
SD14_DATA_FOLDER = os.path.join(ROOT_DIR, 'fingerprint_data/sd14/data')
