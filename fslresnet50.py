import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, History
import pathlib
import pickle

def build_resnet_model():
    model = Sequential()
    pretrained_model = ResNet50(
        include_top=False,
        input_shape=(224, 224, 3),
        pooling='avg',
        classes=26,
        weights='imagenet'
    )
    for layer in pretrained_model.layers:
        layer.trainable = False
    model.add(pretrained_model)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(26, activation='softmax'))
    model.summary()
    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_generator, val_generator, epochs=10):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history_callback = History()
    history = model.fit(train_generator, validation_data=val_generator, epochs=epochs, callbacks=[early_stopping, history_callback])
    return history

def save_model(model, filepath):
    model.save(filepath)

def save_training_history(history, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(history.history, file)

# Data directories
data_dir = "data1/train"
data_dir = pathlib.Path(data_dir)

# ImageDataGenerator
datagen = ImageDataGenerator(
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.2, 1.2]
)

# Train and validation generators
train_generator = datagen.flow_from_directory(
    data_dir,
    class_mode='categorical',
    target_size=(224, 224),
    batch_size=64,
    subset='training',
    seed=46
)

val_generator = ImageDataGenerator(validation_split=0.2).flow_from_directory(
    data_dir,
    class_mode='categorical',
    target_size=(224, 224),
    batch_size=64,
    subset='validation',
    seed=46
)

# Build, train, and save the model
resnet_model = build_resnet_model()
history = train_model(resnet_model, train_generator, val_generator)
save_model(resnet_model, 'Model1/resnet50_saved_model')
save_training_history(history, 'training_history.pkl')

# Load the saved model
loaded_model = load_model("Model1/resnet50_saved_model")