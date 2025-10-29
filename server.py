from flask import Flask, render_template, request, json
from werkzeug.utils import secure_filename
import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import cv2
import face_recognition
from torch.autograd import Variable
from torch import nn
from torchvision import models
import warnings

warnings.filterwarnings("ignore")

# Create upload folder if not exists
UPLOAD_FOLDER = 'Uploaded_Files'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__, template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the Deepfake Detection Model
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

# Image transformation and normalization
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sm = nn.Softmax(dim=1)

# Convert tensor image to saveable format
def im_convert(tensor):
    inv_normalize = transforms.Normalize(mean=-1 * np.divide(mean, std), std=np.divide([1, 1, 1], std))
    image = tensor.cpu().clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy().transpose(1, 2, 0)
    image = np.clip(image, 0, 1)
    cv2.imwrite('./processed_image.png', image * 255)
    return image

# Prediction function
def predict(model, img):
    fmap, logits = model(img.to(torch.device('cpu')))  # Ensure compatibility with CPU
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    return [int(prediction.item()), confidence]

# Custom Dataset for Validation
class ValidationDataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        extracted_frames = list(self.frame_extract(video_path))

        # Pad if frames are less than expected
        while len(extracted_frames) < self.sequence_length:
            extracted_frames.append(np.zeros_like(extracted_frames[0]))  # Black frame padding

        for frame in extracted_frames[:self.sequence_length]:  # Limit to required sequence length
            faces = face_recognition.face_locations(frame)
            if faces:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            frame = cv2.resize(frame, (112, 112))  # Resize to fixed size
            frames.append(self.transform(frame))
        
        frames = torch.stack(frames)
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vid_obj = cv2.VideoCapture(path)
        success, prev_frame = vid_obj.read()
        frame_count = 0
        extracted_frames = []

        while success:
            if frame_count % 3 == 0:  # Extract every 3rd frame (Adjust as needed)
                extracted_frames.append(prev_frame)
            success, prev_frame = vid_obj.read()
            frame_count += 1

        vid_obj.release()
        return extracted_frames

# Deepfake Detection Function
def detect_fake_video(video_path):
    im_size = 112
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    video_dataset = ValidationDataset([video_path], sequence_length=20, transform=train_transforms)
    
    model = Model(2)
    path_to_model = 'model/df_model.pt'

    if not os.path.exists(path_to_model):
        return {'error': "Model file not found!"}

    model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
    model.eval()

    prediction = predict(model, video_dataset[0])
    return prediction

# Flask Routes
@app.route('/', methods=['GET', 'POST'])
def homepage():
    return render_template('index.html')

@app.route('/Detect', methods=['POST'])
def detect_page():
    if 'video' not in request.files:
        return render_template('index.html', data=json.dumps({'error': "No file uploaded!"}))

    video = request.files['video']

    if video.filename == '':
        return render_template('index.html', data=json.dumps({'error': "No selected file!"}))

    video_filename = secure_filename(video.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    video.save(video_path)

    try:
        prediction = detect_fake_video(video_path)
    except Exception as e:
        os.remove(video_path)
        return render_template('index.html', data=json.dumps({'error': str(e)}))

    os.remove(video_path)

    if isinstance(prediction, dict) and 'error' in prediction:
        return render_template('index.html', data=json.dumps(prediction))

    output = "FAKE" if prediction[0] == 0 else "REAL"
    confidence = prediction[1]
    data = json.dumps({'output': output, 'confidence': confidence})
    
    return render_template('index.html', data=data)

# Run Flask App
if __name__ == '__main__':
    app.run(port=5000, debug=True)
