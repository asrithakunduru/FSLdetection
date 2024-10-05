
import matplotlib.pyplot as fslplt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Upload the learned model
fslmodel_path = "Model1/resnet50_saved_model"
fslmodel = load_model(fslmodel_path)

# Open the training log
with open('training_history.pkl', 'rb') as file:
    fslhistory = pickle.load(file)

# Specify the route to your dataset for testing
fsltesting_dir = "data1/test/"

# Obtain the class labels for the testing dataset depending on its subdirectories.
fslclass_labels = sorted(tf.io.gfile.listdir(fsltesting_dir))

# lists to hold true and predicted labels should be initialized
fsltrue_labels = []
fslpredicted_labels = []

# Go over every class in the testing dataset in a loop
for fslclass_label in fslclass_labels:
    fslclass_path = tf.io.gfile.glob(f"{fsltesting_dir}/{fslclass_label}/*")
    
    # go over every picture in the class in a loop
    for fslimg_path in fslclass_path:
        # Load and preprocess the image
        fslimg = tf.keras.preprocessing.image.load_img(fslimg_path, target_size=(224, 224))
        fslx = tf.keras.preprocessing.image.img_to_array(fslimg)
        fslx = np.expand_dims(fslx, axis=0)
        fslx = tf.keras.applications.resnet50.preprocess_input(fslx)

        # Make an estimation
        fslprediction = fslmodel.predict(fslx)
        fslpredicted_label = np.argmax(fslprediction)

        # Add expected and real labels to the lists
        fsltrue_labels.append(fslclass_labels.index(fslclass_label))
        fslpredicted_labels.append(fslpredicted_label)

# To enable more analysis, convert lists to numpy arrays
fsltrue_labels = np.array(fsltrue_labels)
fslpredicted_labels = np.array(fslpredicted_labels)

# Compute the total accuracy
fslaccuracy = np.sum(fsltrue_labels == fslpredicted_labels) / len(fsltrue_labels)
print(f"Overall Accuracy: {fslaccuracy * 100:.2f}%")

# Accuracy of plot validation and training
fslplt.plot(fslhistory['accuracy'])
fslplt.plot(fslhistory['val_accuracy'])
fslplt.title('Model Accuracy')
fslplt.ylabel('Accuracy')
fslplt.xlabel('Epochs')
fslplt.legend(['Train', 'Validation'], loc='upper left')
fslplt.savefig('Model1/acc1.png')  # Retain the figure before presenting it
fslplt.show()

# Plotting validation and training losses
fslplt.plot(fslhistory['loss'])
fslplt.plot(fslhistory['val_loss'])
fslplt.title('Model Loss')
fslplt.ylabel('Loss')
fslplt.xlabel('Epochs')
fslplt.legend(['Train', 'Validation'], loc='upper left')
fslplt.savefig('Model1/loss1.png')  # Retain the figure before presenting it
fslplt.show()
