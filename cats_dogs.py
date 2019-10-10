# Importing all necessary libraries 
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K
from keras.preprocessing import image
import numpy as np

img_width, img_height = 224, 224

train_data_dir = 'C:/Users/sourg/Desktop/cars_data/v_data/train'
validation_data_dir = 'C:/Users/sourg/Desktop/cars_data/v_data/validation'
nb_train_samples = 400 
nb_validation_samples = 100
epochs = 5
#RE-RUN WITH LIKE 10+ EPOCHS
batch_size = 16
  
  
if K.image_data_format() == 'channels_first': 
    input_shape = (3, img_width, img_height) 
else: 
    input_shape = (img_width, img_height, 3) 
  
train_datagen = ImageDataGenerator(
  rescale=1. / 255,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
  train_data_dir,
  target_size=(img_width, img_height),
  batch_size=batch_size,
  class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory(
  validation_data_dir,
  target_size=(img_width, img_height),
  batch_size=batch_size,
  class_mode = 'binary')

model = Sequential() 
model.add(Conv2D(32, (3, 3), input_shape = input_shape)) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 

model.summary()
  
model.add(Conv2D(32, (3, 3))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
  
model.add(Conv2D(64, (3, 3))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
  
model.add(Flatten()) 
model.add(Dense(64)) 
model.add(Activation('relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(1)) 
model.add(Activation('sigmoid')) 

model.summary()
  
model.compile(loss ='binary_crossentropy', 
                     optimizer ='rmsprop', 
                   metrics =['accuracy']) 
  

model.fit_generator(train_generator, 
    steps_per_epoch = nb_train_samples // batch_size, 
    epochs = epochs, validation_data = validation_generator, 
    validation_steps = nb_validation_samples // batch_size) 

model.save('model_saved.h5') 


for i in range(10):
  pathString = 'C:/Users/sourg/Desktop/cars_data/v_data/validation/planes/'+ str(i+1) + '.jpg'
  img_pred = image.load_img(pathString)
  img_pred = image.img_to_array(img_pred)
  img_pred = np.expand_dims(img_pred, axis = 0)


  result = model.predict(img_pred)
  print(result)
  if result[0][0] == 1:
    prediction = "plane"
  else:
    prediction = "car"

  print(prediction)
