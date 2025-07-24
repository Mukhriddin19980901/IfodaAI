import torch
from torch import nn, optim , cuda
from train import ResNetModel , accuracy , TensorEncoder
from load_data import data_download
from warnings import filterwarnings
import json
filterwarnings("ignore")

device = "cuda" if cuda.is_available() else "cpu" 
#device = "cpu"
# cuda borligini aniqlash 
print(device)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
print(torch.__version__)
print(torch.cuda.is_available())

train_data_path = "../datasets/saralangan_dataset/olma_kasalliklari/train/"
validation_path = "../datasets/saralangan_dataset/olma_kasalliklari/validation/"

train_data = data_download(train_data_path)  
val_data = data_download(validation_path)    

model = ResNetModel(num_classes=5)
model.to(device)
criteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters() , lr = 0.001)
epochs = 50


def train(num_epochs=10):
    train_losses, val_losses, val_accs = [], [], []
    for epoch in range(1, num_epochs + 1):
        # — TRAINING BOSHLANISHI —
        model.train()
        total_train_loss = 0.0
        for x, labels in train_data:
            x, labels= x.to(device), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(x)
            loss = criteria(outputs, labels)
            loss.backward() 
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_data)
        train_losses.append(avg_train_loss)

        # — VALIDATION BOSHLANISHI —
        model.eval()
        total_val_loss = 0.0
        total_val_acc = 0.0
        with torch.no_grad():
            for x_val, y_val in val_data:
                x_val, y_val = x_val.to(device), y_val.to(device).long()
                outputs_val = model(x_val)
                loss_val = criteria(outputs_val, y_val)
                total_val_loss += loss_val.item()
                # Clean accuracy call:
                total_val_acc += (outputs_val.argmax(dim=1) == y_val).float().mean().item()

        avg_val_loss = total_val_loss / len(val_data)
        avg_val_acc = total_val_acc / len(val_data)
        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)

        # — LOGGING —
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"[ Train Loss: ----- {avg_train_loss:.4f} -----|----- Val Loss: {avg_val_loss:.4f} -----|----- Val Acc: {avg_val_acc:.4f} ]")

    return train_losses, val_losses, val_accs
print("Training boshlandi...")
train_losses, val_losses, val_accs = train(num_epochs=epochs)

# Modelni 
state = model.state_dict()

json_path = 'model_weights_res18.json'
with open(json_path, 'w') as f:
    json.dump(state, f, cls=TensorEncoder)

print(f"Modelimiz  {epochs} ta  epochda o'qitildi! \n Model o'qitilgandagi oxirgi xatoligi  - {(train_losses[-1])*100} % \n Modelni baholashdagi xatolik -  {(val_losses[-1])*100} % \n Modelning aniqlilik darajasi  - {(val_accs[-1])*100} %")

print(f"✅ Model weightlari {json_path} ko'rinishida JSON formatda saqlandi ")
