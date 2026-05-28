import os
import torch
import torch.nn as nn
import joblib
import random
import numpy as np
from dotenv import load_dotenv
from torchvision import transforms, models
from PIL import Image
import telebot
from telebot import TeleBot, types
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

index = joblib.load('recognizer_index.pkl')
features = index['features']
image_paths = index['image_paths']
labels = index['labels']
label_names = index['label_names']
nn_index = index['nn']

print(f"Индекс загружен: {len(features)} фото, {len(label_names)} классов")
print(f"Классы: {label_names}")

dataset_path = "dataset"
class_photos = {}

print("\nСканирование локальной папки dataset...")
for class_name in label_names:
    class_dir = os.path.join(dataset_path, class_name)
    if os.path.exists(class_dir):
        photos = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_photos[class_name] = photos
        print(f"   {class_name}: {len(photos)} фото")
    else:
        found = False
        for d in os.listdir(dataset_path):
            if class_name.lower() in d.lower() or d.lower() in class_name.lower():
                class_dir = os.path.join(dataset_path, d)
                photos = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                class_photos[class_name] = photos
                print(f"   {class_name} → найдена папка '{d}': {len(photos)} фото")
                found = True
                break
        if not found:
            class_photos[class_name] = []
            print(f"   {class_name}: папка не найдена!")


class PlaceRecognizerModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)
    
checkpoint = torch.load('finalModelTCS.pth', map_location='cpu')
model = PlaceRecognizerModel(num_classes=checkpoint['num_classes'])
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

feature_extractor = nn.Sequential(*list(model.model.children())[:-1]).to(device)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_features(img_path):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    feat = nn.Sequential(*list(model.model.children())[:-1]).to(device)
    with torch.no_grad():
        f = feat(img).cpu().numpy().reshape(1, -1)
    return f

def classify(img_path):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img)
        pred = torch.argmax(out, 1).item()
    return label_names[pred]

def find_similar(img_path, top_k=3):
    query = get_features(img_path)
    
    # Получаем предсказанный класс
    predicted_class = classify(img_path)
    predicted_idx = label_names.index(predicted_class)
    
    # Находим все индексы фото этого класса
    class_indices = [i for i, lbl in enumerate(labels) if lbl == predicted_idx]
    
    # Извлекаем признаки только этого класса
    class_features = features[class_indices]
    
    # Вычисляем косинусное сходство
    similarities = cosine_similarity(query, class_features)[0]
    
    # Берём top_k самых похожих
    top_local_indices = np.argsort(similarities)[-top_k:][::-1]
    
    result = []
    for local_idx in top_local_indices:
        global_idx = class_indices[local_idx]
        path = image_paths[global_idx]
        score = similarities[local_idx]
        result.append((path, score))
    
    return result


#==========================================

# Загружаем токен из .env
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Создаём бота
bot = telebot.TeleBot(BOT_TOKEN)

# Команда /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    text = (
        "Привет! Я бот, который может помочь найти здание СПбГУ в Петергофе по фото.\n\n"
        "Отправьте мне фотографию места — и я найду "
        "3 самых похожих снимка из базы."
    )
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("/start")
    btn2 = types.KeyboardButton("/help")
    btn3 = types.KeyboardButton("/about")
    markup.add(btn1, btn2)
    markup.row(btn3)

    bot.send_message(message.chat.id, text, reply_markup=markup)

# Команда /help
@bot.message_handler(commands=['help'])
def send_help(message):
    text = (
        "Как пользоваться ботом:\n\n"
        "1. Отправьте фотографию здания СПбГУ в Петергофе, которое хотите найти\n"
        "2. Бот сравнит её с базой данных\n"
        "3. Пришлет 3 самых похожих фото\n\n"
        "Бот распознает: ПМ-ПУ, Физфак, НИИФ, Химфак, Шайба, Матмех."
    )
    bot.send_message(message.chat.id, text)

# Команда /about
@bot.message_handler(commands=['about'])
def send_about(message):
    text = (
        "Этот бот был создан для выполнения проекта по теоретической информатике\n\n"
        "Тема проекта: Place Recognition (image retrieval)\n"
        "Описание: необходимо было собрать датасет примечательных мест Петергофа."
        "Также реализовать тг бота, в который можно загрузть фото места, "
        "а модель должна сопоставить 3 самых похожих фото из базы. \n\n"
        "Над проектом работали: Дмитриева Екатерина, Кричман Маргарита, Луненок София"
    )
    bot.send_message(message.chat.id, text)

@bot.message_handler(content_types=['photo'])
def handle_photo(m):
    try:
        msg = bot.send_message(m.chat.id, "🔍 Ищу...")
        
        # Скачиваем
        file = bot.get_file(m.photo[-1].file_id)
        data = bot.download_file(file.file_path)
        temp = f"temp_{m.chat.id}.jpg"
        with open(temp, 'wb') as f:
            f.write(data)
        
        # Определяем место
        place = classify(temp)
        
        # Ищем похожие
        similar = find_similar(temp)
        
        # Отправляем результат
        bot.edit_message_text(f"{place}", m.chat.id, msg.message_id)
        
        for path, score in similar:
            with open(path, 'rb') as f:
                bot.send_photo(m.chat.id, f, caption=f"Сходство: {score*100:.1f}%")
        
        os.remove(temp)
        
    except Exception as e:
        bot.send_message(m.chat.id, f"Ошибка: {e}")
        
# На случай, если пришлют не фото
@bot.message_handler(content_types=['text', 'sticker', 'document'])
def handle_other(message):
    bot.send_message(
        message.chat.id,
        "Пожалуйста, отправьте именно фотографию места в кампусе Петергофа. 🏛️"
    )

# Запуск бота
if __name__ == "__main__":
    print("Бот запущен!")
    bot.infinity_polling()