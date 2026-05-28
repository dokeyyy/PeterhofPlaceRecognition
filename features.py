import torch
import torch.nn as nn
import joblib
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from torchvision import models

# ========== ПУТИ (ВСЁ ЛОКАЛЬНО) ==========
DATASET_PATH = "dataset"  # папка с датасетом в корне проекта
MODEL_PATH = "finalModelTCS.pth"  # файл модели в корне проекта
SAVE_PATH = "recognizer_index.pkl"  # куда сохранить индекс

# ========== УСТРОЙСТВО ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")

# ========== ЗАГРУЗКА МОДЕЛИ ==========
checkpoint = torch.load(MODEL_PATH, map_location='cpu')

class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet50(weights=None)
        in_f = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5), 
            nn.Linear(in_f, 512), 
            nn.ReLU(),
            nn.BatchNorm1d(512), 
            nn.Dropout(0.3), 
            nn.Linear(512, num_classes)
        )
    def forward(self, x): 
      return self.model(x)

model = Model(num_classes=checkpoint['num_classes'])
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device).eval()
print("Модель загружена")

# ========== ТРАНСФОРМАЦИИ ==========
transform = transforms.Compose([
    transforms.Resize((256, 256)), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ========== ЗАГРУЗКА ДАТАСЕТА ==========
dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)
print(f"Загружено изображений: {len(dataset)}")
print(f"Классы: {dataset.classes}")

# ========== ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ ==========
feature_extractor = nn.Sequential(*list(model.model.children())[:-1]).to(device)

features, paths, labels = [], [], []
idx = 0
with torch.no_grad():
    for inputs, lbls in tqdm(loader, desc="Извлечение"):
        feat = feature_extractor(inputs.to(device)).cpu().numpy().reshape(len(inputs), -1)
        features.append(feat)
        labels.extend(lbls.numpy())
        paths.extend([dataset.samples[idx + i][0] for i in range(len(lbls))])
        idx += len(lbls)

features = np.vstack(features)

# ========== ПОСТРОЕНИЕ И СОХРАНЕНИЕ ИНДЕКСА ==========
nn_index = NearestNeighbors(n_neighbors=3, metric='cosine')
nn_index.fit(features)
joblib.dump({'features': features, 
             'image_paths': paths, 
             'labels': labels,
             'label_names': dataset.classes, 
             'nn': nn_index}, SAVE_PATH)