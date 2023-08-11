import pickle
import cv2
import numpy as np
import torch
import torch.nn as nn
import PIL
from torchvision import models, transforms

class FeatureExtractor:
    def __init__(self):
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        img = img.unsqueeze(0)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        net = self.model.to(device)
        img = img.to(device)
        with torch.no_grad():
            feature = net(img).cpu().numpy().flatten()
        img = img.detach()
        torch.cuda.empty_cache()
        return feature / np.linalg.norm(feature)

class ImageSearcher:
    def __init__(self, feature_dict):
        self.images_name = list(feature_dict.keys())
        self.images_features = list(feature_dict.values())
        self.feature_extractor = FeatureExtractor()

    def search(self, input_img_data, top_k=3):
        input_img = cv2.imdecode(np.frombuffer(input_img_data, np.uint8), cv2.IMREAD_COLOR)
        if input_img is None:
            print("Failed to decode input image data.")
            return []

        faces = self.detect_faces(input_img)
        if len(faces) == 1:
            x, y, w, h = faces[0]
            roi_color = input_img[y:y+h, x:x+w]
            resized = cv2.resize(roi_color, (128, 128))
            input_img = resized
        else:
            print("No faces detected in the image.")

        input_feat = self.feature_extractor.extract(input_img)
        result = self.get_top_k(input_feat, top_k)
        return result

    def detect_faces(self, img):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        return faces

    def get_top_k(self, input_feature, top_k):
        my_res_list = []

        for idx, feature in enumerate(self.images_features):
            cur_similarity = np.dot(feature, input_feature) / (np.linalg.norm(feature) * np.linalg.norm(input_feature))
            my_res_list.append((cur_similarity, self.images_name[idx]))

        my_res_list.sort(reverse=True)
        return my_res_list[:top_k]