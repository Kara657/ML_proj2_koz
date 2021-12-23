# CS433-Machine Learning Project-2
Road Segmentation on satellite images

### Structure of Code

##### "pixelwise_model.py"
This is our final CNN model. The code trains a new model or restores a saved model. You can simply change RESTORE_MODEL to False to train the model again. The code also runs prediction on test set and produces mask images. You can comment out corresponding lines to run prediction on training and validation. The code is a modified version of tf_aerial_images.py. Note: To restore the model you need to download the folder which includes model parameters from https://drive.google.com/file/d/1BdGou3kvx9HFgWqOjhLStsMNv3XXp8fH/view?usp=sharing and if you wish to train the model you need to download training set from https://drive.google.com/file/d/1wfulkIZs5BnLSCRw2XcU8L6pcPxSCWzq/view?usp=sharing

##### "mask_to_submission.py"
This file produces csv file from mask images of testing predictions.

##### "rotate.py"
This file is used for image augmentation which simply rotates images and saves in a folder.

##### "run.py"
This is the file that can be used to produce the final submission file based on our pixelwise CNN model.

Beside the codes, we have folders: testing includes the original testing images and testing_masks includes mask images of testing predictions which are used to produce submission file.
### How to run the code
You can simply run the run.py, which produces submission csv file from testing prediction masks. Note: If you downloaded model parameters, you can comment out the first line and use the restored model to produce mask images of testing predictions.

### Warnings
The codes was tested with TensorFlow 1.13.2 and python version of 3.7.0. Due to size limitations of github you need to download model parameters from drive as mentioned above.
