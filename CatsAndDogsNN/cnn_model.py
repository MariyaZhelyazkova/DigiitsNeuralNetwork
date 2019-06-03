from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_model():
    model = Sequential()
    
    model.add(Convolution2D(64, 3, 3, input_shape = (128, 128, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Convolution2D(64, 3, 3, activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dense(units = 256, activation = 'relu'))
    model.add(Dense(units = 1, activation = 'sigmoid'))
    
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model

def train_model(model, train_data_path, validation_data_path):
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

    training_set = train_datagen.flow_from_directory(train_data_path,
                                                     target_size = (128, 128),
                                                     batch_size = 64,
                                                     class_mode = 'binary')
    
    validation_datagen = ImageDataGenerator(rescale = 1./255)
    validation_set = validation_datagen.flow_from_directory(validation_data_path,
                                                            target_size = (128, 128),
                                                            batch_size = 64,
                                                            class_mode = 'binary')
    
    model.fit_generator(training_set,
                        steps_per_epoch = 100,
                        epochs = 1,
                        validation_data = validation_set,
                        validation_steps = 800,
                        workers = 8)
    
def save_model(model, save_path):
    model.save_weights(save_path)
    
    
def restore_model(model, save_path):
    model.load_weights(save_path)