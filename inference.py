from PIL import Image
from Config import Model
import os
if __name__ == "__main__":
    scsnet = Model()
    path = 'Sample-img/'
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path,file)
        img = file_path
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            img_name = file_path.split('/')[1].split('.')[0]
            r_image = scsnet.detect_image(image)
            r_image.save('results/{}.jpg'.format(img_name))
    print('done.')