# HCR
A simple handwriting character recognition model aka my side chick
This repo contains only the code for the model built.
Please note, this is still in progress and I will keep updating the repo as keep working on this.

The End goal of this model is to accept any text as an image and then convert that to text.

Create a folder called "DataSets" and unzip "sample.zip" there.

## Data Preprocessing.

One of the critical steps for making the machine learning models better.
The following techniques are applied to the model.
• The images are rotated randomly by 10,15,20 degrees angles since end users might not always write it in a straight line and also there is going to be production data bias.
• The ink color can be blue, green, red, or black. The only color in this dataset is black and so I have converted all the images to black and white.
• THe next step applied is having different saturation levels to the images so that images taken under different environments will also work with the model.

## Model Architecture

The model uses a simple custom CNN with 15 layers overall.
In the IAM dataset, there are over 11000 unique words each of which is a class to be predicted
The model starts with CNN and then leads to a couple of fully connected layers with dropouts to avoid overfitting.

I achieved an overall accuracy of 90% in the test dataset and 92% accuracy in the training dataset.
