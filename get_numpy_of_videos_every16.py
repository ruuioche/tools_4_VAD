import os
from PIL import Image
import numpy as np
import cv2
import torch
from clip import clip
from transformers import CLIPProcessor, CLIPModel


def video_crop(video_frame, type):
    """
    refer to : https://github.com/nwpu-zxr/VadCLIP/tree/main/src
    """
    l = video_frame.shape[0]
    new_frame = []
    for i in range(l):
        img = cv2.resize(video_frame[i], dsize=(340, 256))
        new_frame.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    img = np.array(new_frame)
    if type == 0:
        img = img[:, 16:240, 58:282, :]
    elif type == 1:
        img = img[:, :224, :224, :]
    elif type == 2:
        img = img[:, :224, -224:, :]
    elif type == 3:
        img = img[:, -224:, :224, :]
    elif type == 4:
        img = img[:, -224:, -224:, :]
    elif type == 5:
        img = img[:, 16:240, 58:282, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    elif type == 6:
        img = img[:, :224, :224, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    elif type == 7:
        img = img[:, :224, -224:, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    elif type == 8:
        img = img[:, -224:, :224, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    elif type == 9:
        img = img[:, -224:, -224:, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    return img






def video_to_numpy(video_path, batch_size=16):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    while True:
        frames = []
        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return None
            frames.append(frame)
        yield np.array(frames)




def process_video(video_path, output_path, model, preprocess, device,type):
    frame_generator = video_to_numpy(video_path)
    video_features = torch.zeros(0).to(device)

    for frames in frame_generator:
        crop_video = video_crop(frames, type)
        with torch.no_grad():
            mean_features = torch.zeros(0).to(device)
            for frame in crop_video:
                img = Image.fromarray(frame)
                img = preprocess(img).unsqueeze(0).to(device)
                features = model.encode_image(img)
                mean_features = torch.cat([mean_features, features], dim=0)
            mean_features = mean_features.mean(dim=0, keepdim=True)
            video_features = torch.cat([video_features, mean_features], dim=0)
    video_features = video_features.detach().cpu().numpy().astype(np.float16)
    print("features shape: ", video_features.shape)
    np.save(output_path + '.npy', video_features)




def read_from_txt_process_video(txt_sample_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(txt_sample_file, 'r') as file:
        for line in file:
            file_path = line.strip()
            try:
                for j in range(10):
                    video_path = file_path
                    file_name_with_extension = os.path.basename(video_path)
                    file_name, extension = os.path.splitext(file_name_with_extension)
                    print("start get numpy of '%s' __ '%d'" % (file_name, j))
                    output_path = f'path/to/your/output/video_npy/{file_name}__{j}'
                    model, preprocess = clip.load("ViT-B/16", device=device)
                    process_video(video_path, output_path, model, preprocess, device,j)
            except Exception as e:
                print(f"An error occurred while reading the file{file_path}:{e}")


"""
feel free to alter the path
"""
os.chdir('path/to/your/videos/')
txt_sample_file = 'your_directions.txt'
read_from_txt_process_video(txt_sample_file)