# CS433-Machine Learning Project-2
Road Segmentation on satellite images

### Structure of Code

##### "pixelwise_model.py"
This is our final CNN model. The code trains a new model or restores a saved model. You can simply change RESTORE_MODEL to False to train the model again. The code also runs prediction on test set and produces mask images. You can comment out corresponding lines to run prediction on training and validation.

##### "mask_to_submission.py"
This file produces csv file from mask images of testing predictions

##### "rotate.py"
This file is used for image augmentation which simply rotates images and saves in a folder.

##### "run.py"
This is the file that can be used to produce the final submission file based on our pixelwise cnn model.

Beside the codes, we have folders: training includes augmented images and corresponding groundtruths. testing includes the testing images to produce submission file. Images in both of these files have the size 512x512.
### How to run the code
You can simply run the run.py, which produces testing prediction masks by restoring the model and creates submission csv file.

### Warnings
The codes was tested with TensorFlow 1.13.2 and python version of 3.7.0
