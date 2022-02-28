from fastai.vision.all import *

dir = "data/train"
all_images = get_image_files(dir)
print(len(all_images))

for image in all_images:
    print(image)
