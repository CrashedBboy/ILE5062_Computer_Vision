import glob
from PIL import Image

DIR = './data/task1and2_hybrid_pyramid'

file_list = glob.glob(DIR + '/*')

for file in file_list:

    ext = file.split('.')[-1]
    ext_lower = ext.lower()

    if ext_lower == 'bmp':
        try:
            img = Image.open(file)
            filename = file.split('.' + ext)[0]

            img.save(filename + '.jpg', 'JPEG', quality = 100)
        except:
            print("error in processing", file)
            continue
