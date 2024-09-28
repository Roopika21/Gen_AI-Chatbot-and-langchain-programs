from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import tensorflow as tf

# Define batch size and number of epochs
batch_size = 8
n_epochs = 40

# Define data generators for training
train_datagen = ImageDataGenerator(rescale = 1/255)
train_generator = train_datagen.flow_from_directory(
    'dataset', #Source directory for training images
    target_size = (400,400), #Resize all images to 400 x 400
    batch_size = batch_size,
    classes=['Healthy Leaf','Diseased Leaf'],
    class_mode = 'categorical'
)

# Debugging print
print(f"Number of classes: {train_generator.num_classes}")
print(f"Number of samples: {train_generator.samples}")
print(f"Class indices: {train_generator.class_indices}")

# Try to fetch a batch to see if it works
try:
    images, labels = next(train_generator)
    print(f"Fetched batch size: {image.shape[0]}")
except Exception as e:
    print(f"Error fetching a batch: {e}")

#Define the CNN model architecture
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(400, 400, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax') #Assuming binary classification with 2class
])

model.summary()

#Compile the model
model.compile(loss='categorical_crossentropy',
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001),
metrics=['accuracy'])

#Calculate steps_per_epoch
steps_per_epoch = max(1, train_generator.samples // batch_size)

#Train the model and store the history
history = model.fit(
    train_generator,
    steps_per_epoch = steps_per_epoch,
    epochs = n_epochs,
    verbose=1
)

#plot trainig accuracy and loss
plt.figure(figsize=(10, 5))

#Plot training accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['Accuracy'], label='Training Accuracy')
plt.title('Trainig Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

#Plot training accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['Loss'], label='Training Accuracy')
plt.title('Trainig Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

#Show p[lots
plt.tight_layout()
plt.show()

#Save the training model
model.save('model.h5')