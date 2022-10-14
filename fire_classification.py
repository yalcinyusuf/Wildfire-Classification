#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator


# In[4]:


training_dir = "dataset/Training"
validation_dir = "dataset/Validation"
input_shape = (224,224,3)


# ### Veri artırma teknikleri

# In[3]:


validation_datagen = ImageDataGenerator(rescale = 1./255)

training_datagen = ImageDataGenerator(rescale = 1./255,
                                      horizontal_flip=True,
                                      rotation_range=30,
                                      height_shift_range=0.2,
                                      fill_mode='nearest')

train_generator = training_datagen.flow_from_directory(training_dir,
                                                       target_size=(224,224),
                                                       class_mode='categorical',
                                                       batch_size = 64)

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              target_size=(224,224),
                                                              class_mode='categorical',
                                                              batch_size= 16)


# In[4]:


import keras
from keras import layers
from tensorflow.keras.optimizers import Adam
#Adam dogrudan parametreli bir şekilde kullanabilmek için yukarıdakini çağırdım.
def fireNet(input_shape):
    #Conv2D ilk parametre tensor boyutu
    #Conv'da bu model için ilk tensor boyutunu küçük yaptım sonra artırdım,Dense'te tam tersi.
    #2 çıktım olacağı için en son Dense katmanını 2'ye düşürdüm.
    #Sequential liste biçiminde modele veriyorum.
    #layers.Dropout rastgele bağlantıları koparıp modeli iyileştirmeye çalışıyor.
    model = keras.models.Sequential([ layers.Conv2D(96, (11,11), strides=(4,4), activation="relu", input_shape = input_shape),
                                      layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)),
                                      
                                      layers.Conv2D(256, (5,5), activation="relu"),
                                      layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)),
                                      
                                      layers.Conv2D(384, (5,5), activation="relu"),
                                      layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)),
                                    
                                      layers.Flatten(),
                                      layers.Dropout(0.3),
                                     
                                      layers.Dense(2048, activation="relu"),
                                      layers.Dropout(0.3),
                                     
                                      layers.Dense(1024, activation="relu"),
                                      layers.Dropout(0.3),
                                     
                                      layers.Dense(2, activation="softmax")
                                     ])
    
    model.compile(loss = "categorical_crossentropy",
                  optimizer = Adam(lr = 1e-4),
                  metrics = ["acc"])
    
    return model


# In[5]:


model = fireNet(input_shape)
model.summary()


# In[6]:


history = model.fit( train_generator,
                     steps_per_epoch = 15,
                     epochs = 50,
                     validation_data = validation_generator,
                     validation_steps = 15 )


# In[7]:


import matplotlib.pyplot as plt

acc = history.history["acc"]
val_acc = history.history["val_acc"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(0,50)


plt.plot(epochs, acc, "g", label="Training Accuracy")
plt.plot(epochs, val_acc, "black", label="Validation Accuracy")
plt.title("Training|Validation Accuracy")

plt.legend(loc=0) #sag alt
plt.figure()
plt.show()


plt.plot(epochs, loss, "r",label="Training Loss")
plt.plot(epochs, val_loss, "blue",label="Validation Loss")
plt.title("Training|Validation Loss")

plt.legend(loc=0)
plt.figure()
plt.show()


# In[8]:


model.save("models/fire_model_yusuf.h5")


# In[5]:


import cv2
import numpy as np
from keras.models import load_model


# In[6]:


model = load_model("models/fire_model_yusuf.h5")
path = "test/test.jpg"
video_path = "test/test.mp4"


# In[7]:


test_img = cv2.imread(path)

img = np.asarray(test_img)
img = cv2.resize(img, (224,224))

img = img/255
#print(img.shape)

img = img.reshape(1,224,224,3)
#print(img.shape)

predictions = model.predict(img)
pred = np.argmax(predictions[0])

probability = predictions[0][pred]
probability_ = "% {:.2f}".format(probability*100)

if pred == 1:
    label = "Fire"
else:
    label = "Neutral"
    
font = cv2.FONT_HERSHEY_SIMPLEX
color = (0,255,0)

cv2.putText(test_img, label, (35,60), font, 1, color, 2)
cv2.putText(test_img, probability_, (35,100), font, 1, color, 2)

cv2.imshow("Prediction", test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[8]:


cap = cv2.VideoCapture(video_path)

while True:
    ret,frame = cap.read()
    
    img = np.asarray(frame)
    img = cv2.resize(img, (224,224))

    img = img/255
    #print(img.shape)

    img = img.reshape(1,224,224,3)
    #print(img.shape)

    predictions = model.predict(img)
    pred = np.argmax(predictions[0])

    probability = predictions[0][pred]
    probability_ = "% {:.2f}".format(probability*100)

    if pred == 1:
        label = "Fire"
    else:
        label = "Neutral"

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0,255,0)

    cv2.putText(frame, label, (35,60), font, 1, color, 2)
    cv2.putText(frame, probability_, (35,100), font, 1, color, 2)

    cv2.imshow("Prediction", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
    
    
cap.release()
cv2.destroyAllWindows()
    

