import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import tempfile

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(page_title="Fire Detection System", layout="centered")
st.title("🔥 Fire & Smoke Detection System")

# ----------------------------------
# Device
# ----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------
# Load ConvNeXt Model
# ----------------------------------
@st.cache_resource
def load_model():
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 3)
    model.load_state_dict(torch.load("convnext_fire_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

class_names = ['fire', 'normal', 'smoke']

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ----------------------------------
# Prediction Function
# ----------------------------------
def predict_image(image):
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    predicted_class = class_names[pred.item()]
    confidence = conf.item()

    return predicted_class, confidence

# ----------------------------------
# Sidebar Options
# ----------------------------------
option = st.sidebar.selectbox(
    "Choose Input Type",
    ["Image Upload", "Video Upload", "Webcam"]
)

# ----------------------------------
# IMAGE UPLOAD
# ----------------------------------
if option == "Image Upload":

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        predicted_class, confidence = predict_image(image)

        if predicted_class in ["fire", "smoke"] and confidence > 0.6:
            st.error(f"🔥 FIRE WARNING FIRE ALERT !!")
        else:
            st.success(f"✅ NO ISSUE")

# ----------------------------------
# VIDEO UPLOAD
# ----------------------------------
elif option == "Video Upload":

    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            predicted_class, confidence = predict_image(image)

            if predicted_class in ["fire", "smoke"] and confidence > 0.6:
                text = f"🔥 FIRE WARNING ALERT !!)"
                color = (0, 0, 255)
            else:
                text = f"✅ NO ISSUE"
                color = (0, 255, 0)

            cv2.putText(frame, text, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 2)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                          channels="RGB")

        cap.release()

# ----------------------------------
# WEBCAM
# ----------------------------------
elif option == "Webcam":

    camera_image = st.camera_input("Take a Picture")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        st.image(image, caption="Captured Image", use_column_width=True)

        predicted_class, confidence = predict_image(image)

        if predicted_class in ["fire", "smoke"] and confidence > 0.6:
            st.error(f"🔥 FIRE WARNING ALERT !!")
        else:
            st.success(f"✅ NO ISSUE ")
