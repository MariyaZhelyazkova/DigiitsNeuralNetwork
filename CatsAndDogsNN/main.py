import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

classifier = Sequential()
classifier.add(Convolution2D(64, 3, 3, input_shape = (128, 128, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(64, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 64,
                                                 class_mode = 'binary')

validation_datagen = ImageDataGenerator(rescale = 1./255)
validation_set = validation_datagen.flow_from_directory('dataset/test_set',
                                                        target_size = (128, 128),
                                                        batch_size = 64,
                                                        class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 1,
                         validation_data = validation_set,
                         validation_steps = 800,
                         workers = 4)

test_datagen = ImageDataGenerator(rescale = 1. / 255)
test_set = test_datagen.flow_from_directory('dataset/test_images',
                                            target_size = (128, 128),
                                            batch_size = 64,
                                            class_mode = 'binary')

result = classifier.predict(test_set);

x, y = test_set.next();

for i, img_pred in enumerate(result):          
    plt.imshow(x[i])
    plt.show()
    print('Dog: %f' % (img_pred[0]))
    print('Cat: %f' % (1 - img_pred[0]))