import glob
from PIL import Image

DIR = './data/task3_colorizing'

file_list = glob.glob(DIR + '/*')

for file in file_list:

    ext = file.split('.')[-1]
    ext_lower = ext.lower()

    if ext_lower == 'tif' or ext_lower == 'tiff':
        try:
            img = Image.open(file)
            img.mode = 'I' # signed integer pixels
            filename = file.split('.' + ext)[0]

            # point() accepts lookup table or function/lambda as parameter to process per point value
            # convert('L') convert image to 8-bit pixels, black and white
            img.point(lambda i:i*(1./256)).convert('L').save(filename + '.jpg', 'JPEG', quality = 100)
        except:
            print("error in processing", file)
            continue

