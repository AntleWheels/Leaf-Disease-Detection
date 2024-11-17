from keras import models
from keras import layers
from keras import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Convolutional base
model = models.Sequential()

#First layer
model.add(layers.Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(128,128,3)))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.BatchNormalization())

# Second layer
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.BatchNormalization())

# Third layer
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.BatchNormalization())

#Fourth layer
model.add(layers.Conv2D(96,(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.BatchNormalization())

#Fifth layer
model.add(layers.Conv2D(32,(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.BatchNormalization())

model.add(layers.Dropout(0.2))
model.add(layers.Flatten())

#Fully connected layer
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dropout(0.3))

model.add(layers.Dense(25,activation='softmax'))# we have 25 classes in our dataset

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale =None,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip =True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/test',
                                                 target_size =(128,128),
                                                 batch_size =32, 
                                                 class_mode ='categorical')
labels = (training_set.class_indices)
print(labels)

test_set = test_datagen.flow_from_directory('dataset/test',
                                            target_size=(128,128),
                                            batch_size=32,
                                            class_mode='categorical')
labels2 = (test_set.class_indices)
print(labels2)
steps_per_epoch = len(training_set) // training_set.batch_size
model.fit(training_set,
                   steps_per_epoch=steps_per_epoch,
                   epochs=20,
                   validation_data=test_set,
                   validation_steps=125)
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.weights.h5")      
print("Saved model to disk")