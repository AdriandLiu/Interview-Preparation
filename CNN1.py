import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from keras import backend
import os
from keras.layers import BatchNormalization
from keras.layers import Activation




class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_id = 0
        self.losses = ''

    def on_epoch_end(self, epoch, logs={}):
        self.losses += "Epoch {}: accuracy -> {:.4f}, val_accuracy -> {:.4f}\n"\
            .format(str(self.epoch_id), logs.get('acc'), logs.get('val_acc'))
        self.epoch_id += 1

    def on_train_begin(self, logs={}):
        self.losses += 'Training begins...\n'

script_dir = os.path.dirname(__file__)
training_set_path = os.path.join(script_dir, 'F:/dataset/training_set')
test_set_path = os.path.join(script_dir, 'F:/dataset/test_set')

##### Initialising the CNN
classifier = Sequential()

##### Step 1 - Convolution (two covolutional layers)
input_size = (128, 128)
classifier.add(Conv2D(32, (3, 3), input_shape=(*input_size, 3)))
classifier.add(Activation("relu"))
classifier.add(Conv2D(32, (3, 3), activation = "relu"))

##### Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))  # 2x2 is optimal
classifier.add(BatchNormalization())
classifier.add(Dropout(0.2))

##### Adding a second convolutional(two)-pooling pair of layers
classifier.add(Conv2D(32, (3, 3)))
classifier.add(Activation("relu"))
classifier.add(Conv2D(32, (3, 3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.2))

##### Adding a third convolutional(three)-pooling pair of layers
classifier.add(Conv2D(64, (3, 3)))
classifier.add(Activation("relu"))
classifier.add(Conv2D(64, (3, 3), activation = "relu"))
classifier.add(Conv2D(64, (3, 3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.2))


##### Step 3 - Flattening
classifier.add(Flatten())

##### Step 4 - Full connection
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=1, activation='sigmoid'))

##### Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images
batch_size = 32
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(training_set_path,
                                                 target_size=input_size,
                                                 batch_size=batch_size,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory(test_set_path,
                                            target_size=input_size,
                                            batch_size=batch_size,
                                            class_mode='binary')

# Create a loss history
history = LossHistory()

classifier.fit_generator(training_set,
                         steps_per_epoch=8000/batch_size,
                         epochs=90,
                         validation_data=test_set,
                         validation_steps=2000/batch_size,
                         workers=12,
                         callbacks=[history])


# Save model
model_backup_path = os.path.join(script_dir, 'F:/dataset/cat_or_dogs_model.h5')
classifier.save(model_backup_path)
print("Model saved to", model_backup_path)

# Save loss history to file
loss_history_path = os.path.join(script_dir, 'F:/dataset/loss_history.log')
myFile = open(loss_history_path, 'w+')
myFile.write(history.losses)
myFile.close()

backend.clear_session()
print("The model class indices are:", training_set.class_indices)
