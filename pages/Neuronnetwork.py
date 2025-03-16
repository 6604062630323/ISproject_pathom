import streamlit as st

st.set_page_config(page_title="Neuron Network")
st.title("scooby doo character classification")
st.write("Start by classifying the five characters of scooby doo show as follows: 1.daphne 2.fred 3.velma 4.scooby doo 5.shaggy")

st.markdown("---")

st.title("Data Overview")
st.markdown('[kaggle.com](https://www.kaggle.com/datasets/esmanurdeli/scooby-doo-classification-dataset) This dataset contains 221 images of different types of scooby doo\'s characters')
st.markdown('Example for Picture')

col1, col2 = st.columns([1, 1])  

with col1:
    st.image("Picture/randompic/fred_1.jpeg")
    st.image("Picture/randompic/fred_6.jpeg")
with col2:
    st.image("Picture/randompic/daphne_5.jpeg")
    st.image("Picture/randompic/daphne_1.jpeg")

st.markdown("---")

st.title("Preprocessing images")
preprocess = ''' from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

#  กำหนด Transform สำหรับ Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize เป็น 224x224 เพื่อใช้กับ ResNet
    transforms.ToTensor(),  # แปลงเป็น Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize ค่า Pixel
])

#  โหลด Dataset
dataset = ImageFolder(root=dataset_path, transform=transform)

#  ดู Class Mapping
print(" Class Mapping:", dataset.class_to_idx)
print(" จำนวนข้อมูลทั้งหมด:", len(dataset))

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print("✅ Train Data:", len(train_dataset))
print("✅ Validation Data:", len(val_dataset))
'''

st.code(preprocess,language='python')

st.subheader("Resize the Image")
st.write("ResNet-18 requires input images of size 224x224 pixels.")
st.write("This transformation ensures that all images are resized to this fixed shape.")

st.subheader("Convert the Image to a Tensor")
st.write("Converts the image from a PIL Image (or NumPy array) to a PyTorch Tensor.")
st.write("Pixel values, originally in the range [0, 255], are scaled down to [0, 1].")

st.subheader("Normalize Pixel Values")
st.write("Normalization helps stabilize training and improve performance.")

st.subheader(" Splitting the Dataset")
st.write("train_size is 80% of the dataset.val_size is the remaining 20%.")

st.markdown("---")

st.title("Model Used")
model = ''' 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("⚡ ใช้ Device:", device) 
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)  # 5 classes: Shaggy, Velma, Fred, Daphne, Scooby

'''
st.code(model,language='python')
st.write("If a GPU (CUDA) is available, it assigns \"cuda\" to device.")
st.write("Loads ResNet-18 from torchvision.models.")
st.write("ResNet-18's original FC layer is designed for 1,000 classes (ImageNet).We replace it with a new nn.Linear(num_ftrs, 5) layer to classify 5 characters.")

model2 = ''' 
import torch.optim as optim

criterion = nn.CrossEntropyLoss()  # ใช้ CrossEntropy เพราะเป็น Classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # ใช้ Adam Optimizer

'''
st.code(model2,language='python')
st.write("CrossEntropyLoss() is commonly used for multi-class classification problems.")
st.write("Adam (Adaptive Moment Estimation) is used for optimizing the model’s weights.")
st.write("model.parameters() → Passes all trainable parameters to the optimizer.")
st.write("lr=0.001 → Learning rate, which controls how much the weights are updated in each step.")

model3 = ''' 
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # เคลียร์ Gradient ก่อน
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()  # คำนวณ Gradient
            optimizer.step()  # ปรับ Weight

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        val_acc = evaluate(model, val_loader)

        print(f"📌 Epoch {epoch+1}/{epochs} | Loss: {running_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    print("🎉 เทรนเสร็จเรียบร้อย!")

# ฟังก์ชันวัดผล Validation Accuracy
def evaluate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total

'''
st.code(model3,language='python')
st.write("Trains the model for 5 epochs (default) using train_loader.")
st.write("Loops through epochs (training cycles).")
st.write("model.train() ensures layers like Dropout & BatchNorm work correctly.")
st.write("Initializes loss, correct predictions, and total samples counters.")
st.write("Moves images & labels to GPU (if available).")
st.write("Clears gradients before each batch (to avoid accumulation).")
st.write("Forward pass → Feeds images into the model to get predictions.")
st.write("Computes loss using criterion(outputs, labels).")
st.write("Backpropagation → Calculates gradients.")
st.write("Updates model weights with optimizer.step().")
st.write("Finds the predicted class (torch.max(outputs, 1)).")
st.write("Calls evaluate(model, val_loader) to get validation accuracy.")
st.write("model.eval() ensures the model runs in evaluation mode (disables dropout, etc.).")
st.write("torch.no_grad() reduces memory usage & speeds up evaluation (no gradient calculations).")
st.write("Loops through validation data, moves tensors to GPU/CPU, and gets predictions. Counts correct predictions for accuracy calculation.")
st.write("Computes validation accuracy percentage.")

st.markdown("---")

st.title("Train the model and Evaluate")

model4 = ''' 
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)
'''
st.code(model4,language='python')
st.write("You are training the ResNet-18 model for 10 epochs using:")
st.write("CrossEntropyLoss for classification.")
st.write("Dense(128, activation='relu') → A fully connected layer with 128 neurons to learn complex patterns.")
st.write("Adam Optimizer to update weights.")

model5 = ''' 
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        val_acc = evaluate(model, val_loader)

        print(f"📌 Epoch {epoch+1}/{epochs} | Loss: {running_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # ✅ Print Final Accuracy
    final_train_acc = 100 * correct / total
    final_val_acc = evaluate(model, val_loader)
    print("\n🎯 **Final Accuracy**")
    print(f"✅ Training Accuracy: {final_train_acc:.2f}%")
    print(f"✅ Validation Accuracy: {final_val_acc:.2f}%")
    print("🎉 Training Complete!")

# Run Training
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)
'''
st.code(model5,language='python')
st.write("Final Accuracy")
st.write(" Training Accuracy:  99.43%")
st.write("Validation Accuracy: 82.22%")