import streamlit as st

st.set_page_config(page_title="Neuron Network")
st.title("Weather classification")
st.write("Start by classifying the five types of weather as follows: 1.dew 2.fogsmog 3.frost 4.glaze 5.hail 6.lightning")

st.markdown("---")

st.title("Data Overview")
st.markdown('[kaggle.com](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset) This dataset contains 6862 images of different types of weather, but only 6 weather types were selected because some, like "rum" and "rainbow," are not actual weather conditions.')
st.markdown('Example for weather Picture')

col1, col2 = st.columns([1, 1])  

with col1:
    st.image("Picture/randompic/11.jpg")
    st.image("Picture/randompic/13.jpg")
with col2:
    st.image("Picture/randompic/102.jpg")
    st.image("Picture/randompic/103.jpg")

st.markdown("---")

st.title("Preprocessing images")
preprocess = ''' import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory

# กำหนด path ของ dataset
dataset_path = "/root/.cache/kagglehub/datasets/jehanbhathena/weather-dataset/versions/3/dataset"

# 🔹 เลือกเฉพาะ 5 คลาสที่ต้องการ
selected_classes = ['fogsmog', 'lightning', 'rain', 'sandstorm', 'snow']

# 🔹 โหลด dataset เฉพาะคลาสที่ต้องการ
train_dataset = image_dataset_from_directory(
    dataset_path,
    validation_split=1 - 0.8,  # ใช้ 80% เทรน 20% เทส
    subset="training",
    seed=123,
    image_size=(128, 128),
    batch_size=32,
    class_names=selected_classes  # ระบุคลาสที่ต้องการโหลดเท่านั้น!
)

val_dataset = image_dataset_from_directory(
    dataset_path,
    validation_split=1 - 0.8,
    subset="validation",
    seed=123,
    image_size=(128, 128),
    batch_size=32,
    class_names=selected_classes
)

print("โหลดเฉพาะคลาสที่ต้องการแล้ว:", train_dataset.class_names)

# Normalize ด้วย Rescaling
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

# ปรับปรุงประสิทธิภาพของ dataset
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

#สร้าง Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2)
])

#ทดสอบ Data Augmentation
for image_batch, label_batch in train_dataset.take(1):
    augmented_images = data_augmentation(image_batch)

print("Dataset เตรียมพร้อมสำหรับการเทรน!")
'''

st.code(preprocess,language='python')

st.subheader("Load and Split Data")
st.write("Load images from the designated folder and split the dataset into **80% training** and **20% validation**.")

st.subheader("Resize")
st.write("Standardize all images to **128×128 pixels** for consistency.")

st.subheader("Batch Processing")
st.write("Use a **batch size of 32**, grouping images into sets of 32 to enhance training efficiency.")

st.subheader("Class Selection")
st.write("Focus on specific weather types: **fogsmog, lightning, rain, sandstorm, and snow**.")

st.subheader("Pixel Scaling")
st.write("Convert pixel values from **0–255 to 0–1** by dividing by 255. This normalization improves learning and minimizes color variations.")

st.subheader("Data Augmentation")
st.write("Enhance image diversity to improve model generalization and reduce overfitting:")

st.markdown("- **RandomFlip(\"horizontal\")** – Flips images left to right at random.")
st.markdown("- **RandomRotation(0.2)** – Rotates images by up to **±20%**.")
st.markdown("- **RandomZoom(0.2)** – Zooms in and out randomly.")

st.markdown("---")

st.title("Model Used")
model = ''' 
    data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2)
])

'''
st.code(model,language='python')
st.write("dataset variability to improve generalization and reduce overfitting.")
st.write("RandomFlip(horizontal) → Randomly flips images horizontally.")
st.write("RandomRotation(0.2) → Randomly rotates images by 20%.")
st.write("RandomZoom(0.2) → Randomly zooms in/out by 20%.")
st.write("RandomContrast(0.2) → Randomly adjusts image contrast.")

model2 = ''' 
layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
layers.Conv2D(32, (3, 3), activation='relu'),
layers.MaxPooling2D(2, 2),
layers.Dropout(0.2),
])

'''
st.code(model2,language='python')
st.write("Extracts essential features such as edges, textures, and shapes.")
st.write("Conv2D(32, (3, 3), activation='relu') → Uses 32 filters of 3×3 size to detect basic features.")
st.write("MaxPooling2D(2, 2) → Reduces the spatial dimensions, lowering computational cost.")
st.write("Dropout(0.2) → Deactivates 20% of neurons to prevent overfitting.")

model3 = ''' 
layers.Conv2D(64, (3, 3), activation='relu'),
layers.Conv2D(64, (3, 3), activation='relu'),
layers.MaxPooling2D(2, 2),
layers.Dropout(0.2),
'''
st.code(model3,language='python')
st.write("Extracts more complex patterns and higher-level features.")

model4 = ''' 
layers.Flatten(),
layers.Dense(128, activation='relu'),
layers.Dropout(0.2),
layers.Dense(5, activation='softmax')
'''
st.code(model4,language='python')
st.write("Transforms feature maps into a classification decision.")
st.write("Flatten() → Converts multi-dimensional feature maps into a 1D vector.")
st.write("Dense(128, activation='relu') → A fully connected layer with 128 neurons to learn complex patterns.")
st.write("Dropout(0.2) → Reduces overfitting by randomly deactivating 20% of neurons.")
st.write("Dense(5, activation='softmax') → Output layer with 5 neurons, one for each class, using softmax for probability distribution.")

model5 = ''' 
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
'''
st.code(model5,language='python')
st.write("Gradually reduces the learning rate to stabilize training.")
st.write("Starts with learning rate = 0.001.")
st.write("Decreases by 10% (0.9 factor) every 1000 steps.")
st.write("Uses Adam Optimizer for adaptive learning rate adjustments.")

model6 = ''' 
epochs = 40  
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)
'''
st.code(model6,language='python')
st.write("This is the training process of your CNN model.  "
"The model adjusts its weights using backpropagation and gradient descent to minimize the loss function. It updates weights after each batch of data.")

st.markdown("---")

st.title("Train the model and Evaluate")
st.image("Picture/pictureforweb/Train.png")
Evaluate = '''   
# ดูค่า accuracy และ validation accuracy
train_acc = history.history['accuracy'][-1]  # ค่าความแม่นยำรอบสุดท้ายของ train
val_acc = history.history['val_accuracy'][-1]  # ค่าความแม่นยำรอบสุดท้ายของ validation

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
)
'''
st.code(Evaluate,language='python')
Evaluate2 = '''   
Train Accuracy: 0.8293
Validation Accuracy: 0.8385
'''
st.code(Evaluate2,language='python')