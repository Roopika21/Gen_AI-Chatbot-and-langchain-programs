from keras.models import load_model
from keras.preprocessing import image
import numpy as Np
import matplotlib.pyplot as plot

#Load the training model
model = load_model('model.h5')

#Define the target size for resizing images
target_size = (400, 400)

#Function to load and preprocess the image for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size = target_size)
    img_array = image.img_to_array(img)
    image_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 #Normalize pixel values
    return img_array

    #Function to make predictions
    def predict_image(image_path):
        #Preprocess the image
        img_array = preprocess_image(image_path)
        #Make prediction
        prediction = model.predict(img_array)
        #Get the predicted class label
        predicted_class_index = np.argmax(prediction)
        #Map the class index to the class name
        class_labels = ['Healthy Leaf','Diseased Leaf']
        predicted_class_label = class_labels[predicted_class_index]
        #Get the confidence score
        cofidence_score = prediction[0][predicted_class_index]
        return predicted_class_label, confidence_score

#Corrected file path format
image_path = r'3.jpg'  #Change this to your image file path

#Make prediction
predicted_class, confidence = predicted_image(image_path)

#Print the predicted class name and confidence score
print(f'Predicted Class: {predicted_class}')
print(f'Confidence: {confidence:.2f}')

#Load and display the image
img = image.load_img(image_path)
plt.imshow(img)
plt.axis('off')
plt.title(f'Predicted Class: {predicted_class}\nConfidence: {confidence:.2f}')
plt.show()