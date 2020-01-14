import numpy as np
import os 
import math
from classifiers import *
#from pipeline import *

from keras.preprocessing.image import ImageDataGenerator

# 1 - Load the model and its pretrained weights
classifier = Meso4()
classifier.load('weights/Meso4_DF')

# 2 - Minimial image generator
# We did use it to read and compute the prediction by batchs on test videos
# but do as you please, the models were trained on 256x256 images in [0,1]^(n*n)

dataGenerator = ImageDataGenerator(rescale=1./255)
generator = dataGenerator.flow_from_directory(
        'test_images',
        target_size=(256, 256),
        batch_size=1,
        class_mode='binary',
        subset='training')

cwd_df = os.getcwd() + "/test_images/df"
cwd_real = os.getcwd() + "/test_images/real"

num_images = len(os.listdir(cwd_df)) + len(os.listdir(cwd_real))
print(num_images)

# 3 - Predict
i = 0
correct = 0

for i in range(num_images):
    X,y = generator.next()
    #print('Predicted :', classifier.predict(X), '\nReal class :', y)

    z = classifier.predict(X)
    if z >= 0.5:
        z = math.ceil(z)
    else:
        z = math.floor(z)

    if z == y:
        correct = correct + 1
    
    i = i + 1
    #print(1.0 * correct/i)

print(1.0 * correct/num_images)

