import cv2
from preprocessing_data import preprocess_data
import json, torch
from train import ResNetModel 
from warnings import *
import numpy as np
import matplotlib.pyplot as plt
filterwarnings("ignore")
import random , os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device="cpu"
# 2. Instantiate model and load JSON weights once outside function
with open('model_weights_res18.json', 'r') as f:
    loaded = json.load(f)
state = {k: torch.tensor(v) for k, v in loaded.items()}

model = ResNetModel(num_classes=5).to(device)
model.load_state_dict(state)
model.eval()

idx_to_name = {0:"alternarioz",
                   1:"kalmaraz-parsha",
                   2: "sog'lom",
                   3:"unshudring",
                   4:"zang"}

image_dir = "D:/AI/datasets/saralangan_dataset/olma_kasalliklari/test/"
number_of_imgs = 9

def get_random_images_from_directory(directory_path, num_images=5, extensions={'.jpg', '.jpeg', '.png',".heic"}):
    all_files = [f for f in os.listdir(directory_path) 
                 if os.path.isfile(os.path.join(directory_path, f)) 
                 and os.path.splitext(f)[1].lower() in extensions]
    selected_files = random.sample(all_files, min(num_images, len(all_files)))
    
    return [os.path.join(directory_path, f) for f in selected_files]

random_path = get_random_images_from_directory(image_dir,num_images=number_of_imgs)

def test(image_path):
    image = preprocess_data(image_path)
    x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)    
    with torch.no_grad():
        outputs = model(x.to(device))
        arr = outputs.cpu().numpy()[0]  
        max_idx = arr.argmax()
        arr_no_max = arr[ arr != arr[max_idx] ]
        max_idx = np.argmax(arr_no_max)
        arr_no_max = arr_no_max[ arr_no_max!= arr_no_max[max_idx] ]
        max_idx = np.argmax(arr_no_max)
    prediction = idx_to_name[max_idx ]
    print("Aniqlangan kasallik nomi :", prediction)
    return idx_to_name[max_idx]    
def visualize(image_path,num_images):
    plt.figure(figsize=(12,8))
    for i in range(num_images):
        img = preprocess_data(image_path[i])
        title = test(image_path[i])

        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    return True  
    
visualize(random_path,num_images = number_of_imgs)
