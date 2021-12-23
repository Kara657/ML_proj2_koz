# import image utilities
from PIL import Image
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
# import os utilities
import os
TRAIN_NUM = 100
# define a function that rotates images in the current directory
# given the rotation in degrees as a parameter
def rotateImages(filename, rotation_degree, num_images):
  # for each image in the current directory
  for i in range(1, TRAIN_NUM+1):
    # open the image
    imageid = "satImage_%.3d" % i
    image_filename = filename + imageid + ".png"
    img = Image.open(image_filename)
    temp_rot_image =img.rotate(rotation_degree)
    #new_folder_name = "images_%.0d" % rotation_degree##comment out for image augmenting
    new_folder_name = "groundtruth_%.0d" % rotation_degree#comment out for gt augmenting
    new_image_folder= new_folder_name
    if not os.path.exists(new_image_folder): # additional, creating folder if not existed
        os.makedirs(new_image_folder)
    new_image_name = "satImage_%3d"% (num_images+i)+".png"
    temp_rot_image.save(os.path.join(new_image_folder, new_image_name)) # save roatetd image
    # close the image
    img.close()
    print(i)
# examples of use
filename = 'groundtruth/'#change to images or groundtruth
rotateImages(filename, 90, 100)
rotateImages(filename, 180, 2*100)
rotateImages(filename, 270, 3*100)