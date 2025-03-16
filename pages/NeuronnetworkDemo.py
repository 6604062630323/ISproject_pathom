import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import random

# โหลดโมเดล
MODEL_PATH = "Neuronnetwork/weatherpredict.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# ฟังก์ชันประมวลผลภาพ
def preprocess_image(image):
    image = image.resize((128, 128))  # ปรับขนาดให้ตรงกับที่โมเดลต้องการ
    image = np.array(image) / 255.0   # ปรับค่าพิกเซลให้อยู่ในช่วง 0-1
    image = np.expand_dims(image, axis=0)  # เพิ่มมิติให้เข้ากับโมเดล
    return image

# ฟังก์ชันสุ่มเลือกรูปภาพจากโฟลเดอร์
FOLDER_PATH = "Picture/randompic"

def get_random_image():
    if not os.path.exists(FOLDER_PATH):
        return None  # ถ้าโฟลเดอร์ไม่มีอยู่ให้คืนค่า None

    files = [f for f in os.listdir(FOLDER_PATH) if f.lower().endswith(("png", "jpg", "jpeg"))]
    
    if not files:
        return None  # ถ้าไม่มีไฟล์รูปภาพให้คืนค่า None
    
    random_file = random.choice(files)
    return os.path.join(FOLDER_PATH, random_file)

# ส่วน UI
st.title("Weather Classification Demo")
st.write("อัปโหลดรูปภาพหรือสุ่มรูปจากโฟลเดอร์เพื่อให้โมเดลจำแนกสภาพอากาศ")

# อัปโหลดรูป
uploaded_file = st.file_uploader("Select Picture", type=["jpg", "png", "jpeg"])

# ปุ่มสุ่มรูปภาพ
if st.button("Random picture"):
    random_image_path = get_random_image()

    if random_image_path:
        image = Image.open(random_image_path)
        st.image(image, caption="Random Pictire :", use_container_width=True)

        # ประมวลผลรูปและทำนาย
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)

        # แปลงผลลัพธ์เป็น label
        labels = ["fogsmog","lightning", "rain", "sandstorm", "snow"]
        predicted_label = labels[np.argmax(prediction)]

        # แสดงผลลัพธ์
        st.success(f"Predict: **{predicted_label}**")

    else:
        st.warning("Not Found")

# ตรวจสอบว่ามีไฟล์อัปโหลดหรือไม่
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="รูปที่อัปโหลด", use_container_width=True)

    # ประมวลผลรูปและทำนาย
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)

    # แปลงผลลัพธ์เป็น label
    labels = ["fogsmog","lightning", "rain", "sandstorm", "snow"]
    predicted_label = labels[np.argmax(prediction)]

    # แสดงผลลัพธ์
    st.success(f"โมเดลทำนายว่าสภาพอากาศคือ: **{predicted_label}**")
