
# French Sign Language Recognition using ResNet50

## Project Overview
This project focuses on recognizing French Sign Language (FSL) alphabets using a deep learning approach, leveraging the power of the pre-trained ResNet50 model through transfer learning. The project includes **a custom-created dataset** for FSL alphabets, and it is designed to detect hand gestures and predict FSL alphabets in real-time using a webcam.

## Methodology
### 1. Dataset Creation
- A custom dataset of approximately 2500 images was created specifically for this project. Each image represents a letter in the FSL alphabet.
- The dataset is split into `train` and `test` directories with images pre-processed to a size of 224x224 pixels to fit the ResNet50 model requirements.
- For testing purposes, 10 images per alphabet were reserved.

### 2. Model Training
- Used transfer learning from the pre-trained ResNet50 model on the ImageNet dataset.
- Added custom layers to classify 23 FSL alphabet classes.
- Applied data augmentation techniques (rotation, zoom, flips, etc.) to enhance the model's generalization capabilities.
- Achieved an accuracy of **96.01%** on training data and **84.78%** on the test dataset.

### 3. Model Evaluation
- Evaluated the model using validation and test datasets.
- Plots for accuracy and loss over the training epochs were generated to assess model performance.
- Recorded training history and saved the model for further use.

### 4. Real-Time Prediction
- Integrated hand-tracking with the model for real-time FSL alphabet detection using a webcam.
- Pre-processed each detected hand image and used the trained model to predict the alphabet, which was displayed on the live video feed.

## Results
- The model demonstrated an impressive training accuracy of **96.01%**, validation accuracy of **90.91%**, and testing accuracy of **84.78%**.
- Real-time predictions were successfully visualized, showcasing the model’s potential for aiding individuals with hearing impairments in real-world scenarios.

## Conclusion
This project showcases a robust system for FSL recognition using the ResNet50 model, achieving high accuracy and real-time prediction capability. Future improvements could include expanding the dataset, dynamic gesture recognition, and exploring novel architectures for enhanced performance.

## Future Work
- Expanding the dataset for increased diversity.
- Implementing dynamic sign recognition (recognizing full words and phrases).
- Exploring advanced architectures and augmented reality applications.

## License
This project is licensed under the MIT License.


##Sample images in the dataset created

![image](https://github.com/user-attachments/assets/70833067-e418-4513-8f72-215548effaaa)

##Visualization of training, testing and validation accuracies

![image](https://github.com/user-attachments/assets/28e03634-680d-4432-9295-8e6171fce9dc)

##Real-time Detection

![image](https://github.com/user-attachments/assets/d04e9f4f-85ee-4f82-8272-3a83b93f8c6b)




