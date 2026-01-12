from torch.utils.data import Dataset, DataLoader
import os
import json
import cv2
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np

import torch.nn as nn
from torchvision.models.video import r3d_18
from tqdm import tqdm

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

FACE_DB_ROOT = r"/home/sk/deepfake_project/data/dataset/face_db"
LABELS_ROOT   = r"/home/sk/deepfake_project/data/dataset/splits"

def preprocess_frames(frames, indices, size=112):
  processed = []

  for idx in indices:
    frame = frames[idx]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (size, size))

    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    frame_tensor = (frame_tensor - IMAGENET_MEAN)/IMAGENET_STD

    processed.append(frame_tensor)


  video_tensor = torch.stack(processed, dim = 0)
  video_tensor = video_tensor.permute(1, 0, 2, 3)

  return video_tensor




class FaceCropDataset(Dataset):
    def __init__(self, face_root, split_json, k=16):

        self.face_root = face_root
        self.k = k
        self.split_json = split_json

        with open(self.split_json, "r") as f:
            labels = json.load(f)

        self.samples = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]

        rel_stem = os.path.splitext(rel_path)[0]
        clip_dir = os.path.join(self.face_root, rel_stem)

        frame_files = [os.path.join(clip_dir, f"frame_{i:02d}.jpg") for i in range(self.k)]

        frames = []
        for fp in frame_files:
            img = cv2.imread(fp)
            if img is None:
                if frames:
                    img = frames[-1].copy()
                else:
                    img = np.zeros((112, 112, 3), dtype=np.uint8)
            frames.append(img)

        indices = list(range(self.k))
        video_tensor = preprocess_frames(frames, indices)

        return video_tensor, torch.tensor(label, dtype=torch.long)



class DeepfakeDetector3D(nn.Module):
  def __init__(self, pretrained=True):
    super().__init__()

    self.backbone = r3d_18(weights="DEFAULT" if pretrained else None)

    in_features = self.backbone.fc.in_features

    self.backbone.fc = nn.Linear(in_features, 1)

  def forward(self, x):
    logits = self.backbone(x)

    return logits

pos_weight = torch.tensor([1.0], device="cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)




def train_one_epoch(model, loader, optimizer, criterion, device):
  model.train()
  total_loss = 0

  for videos, labels in tqdm(loader):
    videos = videos.to(device)
    labels = labels.float().to(device).unsqueeze(1)

    optimizer.zero_grad()


    logits = model(videos)

    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    total_loss += loss.item() * videos.size(0)

  return total_loss/ len(loader.dataset)


def eval_model(model, loader, device, criterion):
  model.eval()

  all_labels = []
  all_probs = []

  total_loss = 0

  with torch.no_grad():
    for videos, labels in tqdm(loader):
      videos = videos.to(device)
      labels_np = labels.cpu().numpy()
      all_labels.extend(labels_np)

      labels_tensor = labels.float().to(device).unsqueeze(1)

      logits = model(videos)
      probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)

      all_probs.extend(probs)

      loss = criterion(logits, labels_tensor)
      total_loss += loss.item() * videos.size(0)


  all_labels = np.array(all_labels)
  all_probs = np.array(all_probs)

  preds = (np.array(all_probs) > 0.5).astype(int)

  acc = accuracy_score(all_labels, preds)
  f1 = f1_score(all_labels, preds)
  auc = roc_auc_score(all_labels, all_probs)

  print("Prediction balance:")
  print("Predicted REAL (0):", np.sum(preds == 0))
  print("Predicted FAKE (1):", np.sum(preds == 1))
  print()

  return total_loss/len(loader.dataset), acc, f1, auc

def test_model(model, loader, device, criterion):
  model.eval()

  all_labels = []
  all_probs = []

  total_loss = 0

  with torch.no_grad():
    for videos, labels in tqdm(loader):
      videos = videos.to(device)
      labels_np = labels.cpu().numpy()
      all_labels.extend(labels_np)

      labels_tensor = labels.float().to(device).unsqueeze(1)

      logits = model(videos)
      probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)

      all_probs.extend(probs)

      loss = criterion(logits, labels_tensor)
      total_loss += loss.item() * videos.size(0)


  all_labels = np.array(all_labels)
  all_probs = np.array(all_probs)

  preds = (np.array(all_probs) > 0.5).astype(int)

  acc = accuracy_score(all_labels, preds)
  f1 = f1_score(all_labels, preds)
  auc = roc_auc_score(all_labels, all_probs)

  print("Prediction balance:")
  print("Predicted REAL (0):", np.sum(preds == 0))
  print("Predicted FAKE (1):", np.sum(preds == 1))
  print()

  return total_loss/len(loader.dataset), acc, f1, auc



train_ds = FaceCropDataset(
    face_root=FACE_DB_ROOT,
    split_json=os.path.join(LABELS_ROOT, "train.json"),
    k=16
)

val_ds = FaceCropDataset(
    face_root=FACE_DB_ROOT,
    split_json=os.path.join(LABELS_ROOT, "val.json"),
    k=16
)

test_ds = FaceCropDataset(
    face_root=FACE_DB_ROOT,
    split_json=os.path.join(LABELS_ROOT, "test.json"),
    k=16
)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=4, shuffle=False, num_workers=4)



num_epochs = 20

device = "cuda" if torch.cuda.is_available() else "cpu"
model_mtcnn = DeepfakeDetector3D(pretrained=True).to(device)

optimizer = torch.optim.Adam(model_mtcnn.parameters(), lr=1e-5)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=2
)

best_auc = 0
patience = 5
counter = 0

for epoch in range(num_epochs):
  train_loss = train_one_epoch(model_mtcnn, train_loader, optimizer, criterion, device)
  val_loss, val_acc, val_f1, val_auc = eval_model(model_mtcnn, val_loader, device, criterion)

  scheduler.step(val_auc)

  if val_auc > best_auc:
    best_auc = val_auc
    counter = 0
    torch.save(model_mtcnn.state_dict(), "faces/best_model.pth")
    print("Saved best model!")
  else:
    counter += 1
    print(f"No improvement for {counter} epochs.")

  print(f"\nEpoch {epoch+1}: Loss={train_loss: .4f} Val_Loss = {val_loss:.4f} Acc={val_acc:.3f} F1={val_f1:.3f} AUC={val_auc:.3f}")

  if counter >= patience:
    print("Early stopping triggered.")
    break

  torch.save(model_mtcnn.state_dict(), f"faces/checkpoint_epoch_{epoch+1}.pth")



model_best = DeepfakeDetector3D(pretrained=False)

checkpoint = torch.load('faces/best_model.pth', map_location=device)
model_best.load_state_dict(checkpoint)

model_best.to(device)

test_loss, test_acc, test_f1, test_auc = test_model(
    model_best, test_loader, device, criterion
)

print(f"\nTest Results with MTCNN:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Accuracy:  {test_acc:.3f}")
print(f"F1 Score:  {test_f1:.3f}")
print(f"AUC Score: {test_auc:.3f}")