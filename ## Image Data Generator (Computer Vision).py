#!/usr/bin/env python
# coding: utf-8

# ## Image Data Generator (Computer Vision)

# ## Initialization

# In[1]:


#import libraries

#general 
import numpy as np
import pandas as pd

#machine learning needs 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#image specific
from IPython.display import display
from PIL import Image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet import ResNet50
import matplotlib.pyplot as plt


# ## Load Data

# The dataset is stored in the `/datasets/faces/` folder, there you can find
# - The `final_files` folder with 7.6k photos
# - The `labels.csv` file with labels, with two columns: `file_name` and `real_age`
# 
# Given the fact that the number of image files is rather high, it is advisable to avoid reading them all at once, which would greatly consume computational resources. Build a generator with the ImageDataGenerator generator. 

# In[2]:


labels = pd.read_csv('/datasets/faces/labels.csv')


# ## EDA

# In[3]:


labels.info()


# In[4]:


labels.shape


# In[5]:


print('Duplicates:',labels.duplicated().sum())
print('NA:',labels.isna().sum())
print('Null',labels.isnull().sum())


# In[6]:


#print example photos to explore
display(labels.tail(15))


# In[7]:


#show age distribution in dataset

# Create the histogram
plt.hist(labels['real_age'], bins=10, color='pink', edgecolor='black') 

# Add labels and title
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Ages')


# In[8]:


sorted = labels.sort_values(by='real_age')
display(sorted)


# In[9]:


#create brackets for each decade 
def age_bracket(age):
  """Categorizes age into brackets."""
  if age < 10:
    return "Children"
  elif 10 <= age <= 19:
    return "Teenagers"
  elif 20 <= age <= 29:
    return "Twenties"
  elif 30 <= age <= 39:
    return "Thirties"
  elif 40 <= age <= 49:
    return "Forties"
  elif 50 <= age <= 59:
    return "Fifties"
  elif 60 <= age <= 69:
    return "Sixties"
  elif 70 <= age <= 70:
    return "Seventies"
  elif 80 <= age <= 89:
    return "Eighties"
  else:
    return "Nineties"

# Apply the function to create a new 'Age Bracket' column
labels['age_bracket'] = labels['real_age'].apply(age_bracket)
age_range=labels['age_bracket'].value_counts()
print(age_range)


# In[10]:


#print sample images to explore

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the image data generator with rescaling
train_datagen = ImageDataGenerator(validation_split=0.25, rescale=1/255)

# Load training data
train_gen_flow = train_datagen.flow_from_dataframe(
        dataframe=labels,
        directory='/datasets/faces/final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='training',
        seed=12345
)

# Load validation data
val_gen_flow = train_datagen.flow_from_dataframe(
        dataframe=labels,
        directory='/datasets/faces/final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='validation',
        seed=12345
)

# Fetch a batch of images and labels
features, target = next(train_gen_flow)

# Plot images
fig = plt.figure(figsize=(10,10))
for i in range(16):
    ax = fig.add_subplot(4, 4, i+1)
    ax.imshow(features[i])  # Ensure features is an array
    ax.axis('off')

plt.show()


# ### Findings

# Highest age bracket represented is from 20-30 years old.  The people in their seventies are least represented. In evaluating some sample images the photos are not unified in point of view of the camerea.  Some eyes are not visible, some photos have more of a side view, and varied lighting in each photo.  Plan to perform several augmentations to ensure the model is trained correctly.  

# ## Modelling

# Define the necessary functions to train model on the GPU platform and build a single script containing all of them along with the initialization section.

# In[11]:


import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam


# In[12]:


def load_train(path):
    labels = pd.read_csv(path + 'labels.csv')
    
    train_datagen = ImageDataGenerator(
        validation_split=0.25, 
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=90)
    

    train_gen_flow = train_datagen.flow_from_dataframe(
        dataframe=labels,
        directory=path + 'final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='training',
        seed=12345)
    
    return train_gen_flow


# In[13]:


def load_test(path):
    
    """
    It loads the validation/test part of dataset from path
    """    
    labels = pd.read_csv(path + 'labels.csv')
    #do not have any augmentations on test data image generation
    val_datagen = ImageDataGenerator(validation_split=0.25, rescale=1./255)
    
    val_gen_flow = val_datagen.flow_from_dataframe(
        dataframe=labels,
        directory=path + 'final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='validation',
        seed=12345)

    return val_gen_flow


# In[14]:


def create_model(input_shape):
    
    """
    It defines model
    """
    
    backbone = ResNet50(weights='imagenet', 
                        input_shape=input_shape,
                        include_top=False)

    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='relu'))

    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model


# In[15]:


def train_model(model, train_data, test_data, batch_size=None, epochs=20,
                steps_per_epoch=None, validation_steps=None):

    """
    Trains the model given the parameters
    """
    
    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
        
    if validation_steps is None:
        validation_steps = len(test_data)

    model.fit(train_data, 
              validation_data=test_data,
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2)

    return model


# In[16]:


load_train('/datasets/faces/')


# In[17]:


load_test('/datasets/faces/')


# In[18]:


model=create_model((224,224,3))
trained_model=train_model(model, train_data,test_data,batch_size=32, epochs=20,steps_per_epoch=178, validation_steps=60,)


# ## Prepare the Script to Run on the GPU Platform

# In[ ]:


# prepare a script to run on the GPU platform

init_str = """
import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
"""

import inspect

with open('run_model_on_gpu.py', 'w') as f:
    
    f.write(init_str)
    f.write('\n\n')
        
    for fn_name in [load_train, load_test, create_model, train_model]:
        
        src = inspect.getsource(fn_name)
        f.write(src)
        f.write('\n\n')


# ### Output

# Place the output from the GPU platform as an Markdown cell here.
Epoch 1/20
356/356 - 35s - loss: 95.3532 - mae: 7.4339 - val_loss: 124.3362 - val_mae: 8.4921
Epoch 2/20
356/356 - 35s - loss: 76.8372 - mae: 6.6707 - val_loss: 127.6357 - val_mae: 8.6035
Epoch 3/20
356/356 - 35s - loss: 69.9428 - mae: 6.3992 - val_loss: 91.1531 - val_mae: 7.4454
Epoch 4/20
356/356 - 35s - loss: 64.4249 - mae: 6.1407 - val_loss: 124.0287 - val_mae: 8.3481
Epoch 5/20
356/356 - 35s - loss: 52.8486 - mae: 5.5913 - val_loss: 109.1004 - val_mae: 8.2192
Epoch 6/20
356/356 - 35s - loss: 46.3094 - mae: 5.2223 - val_loss: 85.1038 - val_mae: 7.0332
Epoch 7/20
356/356 - 35s - loss: 38.2617 - mae: 4.7951 - val_loss: 92.0900 - val_mae: 7.3359
Epoch 8/20
356/356 - 35s - loss: 37.4804 - mae: 4.7402 - val_loss: 80.0016 - val_mae: 6.7239
Epoch 9/20
356/356 - 35s - loss: 33.5237 - mae: 4.4271 - val_loss: 83.2579 - val_mae: 6.8529
Epoch 10/20
356/356 - 35s - loss: 28.5170 - mae: 4.1411 - val_loss: 83.5056 - val_mae: 6.9629
Epoch 11/20
356/356 - 35s - loss: 27.0142 - mae: 3.9700 - val_loss: 92.1290 - val_mae: 7.1866
Epoch 12/20
356/356 - 35s - loss: 27.4564 - mae: 4.0428 - val_loss: 185.6307 - val_mae: 11.4591
Epoch 13/20
356/356 - 35s - loss: 23.7961 - mae: 3.7407 - val_loss: 92.3429 - val_mae: 7.2467
Epoch 14/20
356/356 - 35s - loss: 24.6167 - mae: 3.8116 - val_loss: 92.4542 - val_mae: 7.1401
Epoch 15/20
356/356 - 35s - loss: 22.2604 - mae: 3.6746 - val_loss: 82.5822 - val_mae: 6.7841
Epoch 16/20
356/356 - 35s - loss: 20.1899 - mae: 3.4430 - val_loss: 86.3830 - val_mae: 6.8304
Epoch 17/20
356/356 - 35s - loss: 17.3425 - mae: 3.2205 - val_loss: 78.4369 - val_mae: 6.6419
Epoch 18/20
356/356 - 35s - loss: 16.5249 - mae: 3.1295 - val_loss: 81.7731 - val_mae: 6.7226
Epoch 19/20
356/356 - 35s - loss: 16.6140 - mae: 3.1421 - val_loss: 80.9727 - val_mae: 6.9908
Epoch 20/20
356/356 - 35s - loss: 17.0187 - mae: 3.1785 - val_loss: 93.4115 - val_mae: 7.6512
# ## Conclusions

# The model shows  improvement in training performance over 20 epochs, with the training loss decreasing from 95.35 to 17.02.  Each epoch is a complete pass through training dataset.  Trailing loss measure how well the model's predictions match the actual values during training. The model is better at minimizing errors from the training set. 
# 
# Mean Absolute Error (MAE) calculates the average absolute difference between predicted and actual values which gives a metric of error in same units as target variable. MAE improved from 7.43 to 3.18. Lower MAE indicates that the predictions from the model are approaching the true values. 
# 
# The validation performance is inconsistent. Initially it decreases but it fluctuates and worsens in later epochs.  The peak was at 185.63 (epoch 12).  In final epoch was at 93.41. This gives evidence of overfitting. Regularization techniques or adjusting the learning rate may help improve generalization.
