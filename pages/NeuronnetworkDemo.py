import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import random

# โหลดโมเดล PyTorch
MODEL_PATH = "Neuronnetwork/scooby_doo_resnet18.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# สร้างโครงสร้างโมเดล ResNet-18 และแก้ output layer ให้ตรงกับ 5 คลาส
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)  # มี 5 คลาส

# โหลด state_dict ของโมเดล
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# ฟังก์ชันประมวลผลภาพ
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ปรับขนาดภาพ
        transforms.ToTensor(),          # แปลงเป็น Tensor
        transforms.Normalize((0.5,), (0.5,))  # ปรับค่าสีให้เหมาะสม
    ])
    image = transform(image).unsqueeze(0)  # เพิ่มมิติให้เข้ากับโมเดล
    return image.to(device)

# ฟังก์ชันสุ่มเลือกรูปภาพจากโฟลเดอร์
FOLDER_PATH = "Picture/randompic"

def get_random_image():
    if not os.path.exists(FOLDER_PATH):
        return None
    
    files = [f for f in os.listdir(FOLDER_PATH) if f.lower().endswith(("png", "jpg", "jpeg"))]
    if not files:
        return None
    
    random_file = random.choice(files)
    return os.path.join(FOLDER_PATH, random_file)

# ส่วน UI
st.title("Scooby-Doo Classification Demo")
st.write("อัปโหลดรูปภาพหรือสุ่มรูปจากโฟลเดอร์เพื่อให้โมเดลจำแนกตัวละคร Scooby-Doo")

# อัปโหลดรูป
uploaded_file = st.file_uploader("Select Picture", type=["jpg", "png", "jpeg"])

# ปุ่มสุ่มรูปภาพ
if st.button("Random picture"):
    random_image_path = get_random_image()
    
    if random_image_path:
        image = Image.open(random_image_path)
        st.image(image, caption="Random Picture", use_container_width=True)

        # ประมวลผลรูปและทำนาย
        processed_image = preprocess_image(image)
        with torch.no_grad():
            prediction = model(processed_image)
        
        # แปลงผลลัพธ์เป็น label
        labels = ['Daphne', 'Fred', 'Scooby', 'Shaggy', 'Velma']
        predicted_label = labels[torch.argmax(prediction).item()]

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
    with torch.no_grad():
        prediction = model(processed_image)
    
    # แปลงผลลัพธ์เป็น label
    labels = ['Daphne', 'Fred', 'Scooby', 'Shaggy', 'Velma']
    predicted_label = labels[torch.argmax(prediction).item()]

    # แสดงผลลัพธ์
    st.success(f"โมเดลทำนายว่าตัวละครคือ: **{predicted_label}**")
