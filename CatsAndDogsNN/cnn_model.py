import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def create_model():
    model = Sequential()
    
    model.add(Convolution2D(64, 3, 3, input_shape = (128, 128, 3), activation = tf.nn.relu))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Convolution2D(64, 3, 3, activation = tf.nn.relu))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dense(units = 256, activation = tf.nn.relu))
    model.add(Dense(units = 3, activation = tf.nn.softmax))
    
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return model

def train_model(model, train_data_path, validation_data_path):
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

    training_set = train_datagen.flow_from_directory(train_data_path,
                                                     target_size = (128, 128),
                                                     batch_size = 64,
                                                     class_mode = 'categorical')

    validation_datagen = ImageDataGenerator(rescale = 1./255)
    validation_set = validation_datagen.flow_from_directory(validation_data_path,
                                                            target_size = (128, 128),
                                                            batch_size = 64,
                                                            class_mode = 'categorical')

    model.fit_generator(training_set,
                        steps_per_epoch = 12000,
                        epochs = 3,
                        validation_data = validation_set,
                        validation_steps = 3000,
                        workers = 8)

def save_model(model, save_path):
    model.save_weights(save_path)

def restore_model(model, save_path):
    model.load_weights(save_path)

def get_labels(data_path):
    return [f.name for f in os.scandir(data_path) if f.is_dir()]    