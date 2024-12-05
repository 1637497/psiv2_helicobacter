import torch
from torch import optim
from CrearModelDeb import AEConfigs, Standard_Dataset, AutoEncoderCNN
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from skimage import io
from skimage.util import img_as_float32, img_as_ubyte
import pickle
import os
import torchvision.transforms as transforms
import cv2



inputmodule_paramsEnc={}
inputmodule_paramsEnc['num_input_channels']=3
net_paramsEnc = {}
net_paramsDec = {}
inputmodule_paramsDec = {}

print("CREANT MODEL")

Config = "1"
net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs(Config, inputmodule_paramsEnc)
model = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


with open("/export/fhome/maed06/VirtualEnv/pickels/imatges_cropped_negatives.pkl", "rb") as f:
    imatges_cropped = pickle.load(f)
print("DADES CARREGADES")
x_train, x_val = train_test_split(imatges_cropped, test_size=0.2, random_state=42)
dataset_train = Standard_Dataset(x_train)
dataset_val = Standard_Dataset(x_val)

train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=64, shuffle=False)


optimizer = optim.Adam(model.parameters(), lr=1e-2)
criterion = torch.nn.MSELoss()
print("MODEL CREAT")


num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images in train_loader:
        images = images.to(device, dtype=torch.float)
        images = images.permute(0, 3, 1, 2)  

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images in val_loader:
            images = images.to(device, dtype=torch.float)
            images = images.permute(0, 3, 1, 2)
            outputs = model(images)
            loss = criterion(outputs, images)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

torch.save(model.state_dict(), "./autoencoder_trained20epo_new.pth")

def reconstruct_and_save_image(model, image_path, save_path, device):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"La imagen no se ha encontrado: {image_path}")

    input_image = cv2.imread(image_path)
    if input_image is None:
        raise ValueError(f"La imagen no se ha podido cargar correctamente: {image_path}")
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = input_image.astype(np.float32) / 255.0 

    transform = transforms.Compose([transforms.ToTensor()]) 
    input_tensor = transform(input_image).unsqueeze(0).to(device) 

    model.eval()
    with torch.no_grad():
        output_tensor = model(input_tensor)

    output_image = output_tensor.squeeze(0).cpu().numpy() 
    output_image = np.transpose(output_image, (1, 2, 0)) 
    output_image = np.clip(output_image, 0, 1) 
    output_image_uint8 = (output_image * 255).astype(np.uint8) 

    output_image_pil = Image.fromarray(output_image_uint8, mode='RGB')
    output_image_pil.save(save_path)

    print(f"Imagen reconstruida guardada en {save_path}")



test_image_path = "/export/fhome/maed06/VirtualEnv/outputs_AE/inicial.png"
reconstruct_and_save_image(model, test_image_path, "/export/fhome/maed06/VirtualEnv/outputs_AE/reconstructed_image_20_new.png", device)

