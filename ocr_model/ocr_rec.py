# 1.RCTW-17 数据集识别
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F


class RCTW_Dataset(Dataset):
    def __init__(self, root_dir, is_train=True, img_size=(640, 640)):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "JPEGImages")
        self.anno_dir = os.path.join(root_dir, "Annotations")
        self.img_size = img_size
        self.img_names = [
            f[:-4] for f in os.listdir(self.img_dir) if f.endswith(".jpg")
        ]
        # 训练/验证划分
        split_idx = int(len(self.img_names) * 0.9)
        self.img_names = (
            self.img_names[:split_idx] if is_train else self.img_names[split_idx:]
        )
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        # 加载图像
        img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found or cannot be read: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        # 加载标注（x1,y1,x2,y2,x3,y3,x4,y4,难度,文本）
        anno_path = os.path.join(self.anno_dir, f"{img_name}.txt")
        with open(anno_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # 处理标注：转换为检测框+文本
        boxes = []  # 检测框：[x1,y1,x2,y2,x3,y3,x4,y4]
        texts = []  # 识别文本
        for line in lines:
            line = line.strip().split(",")
            if len(line) < 10:
                continue
            box = list(map(float, line[:8]))
            text = line[9]
            # 坐标归一化
            box = [b / w if i % 2 == 0 else b / h for i, b in enumerate(box)]
            boxes.append(box)
            texts.append(text)
        # 图像预处理
        img = self.transform(img)
        return {
            "image": img,
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "texts": texts,
        }


# 数据加载器
train_dataset = RCTW_Dataset(root_dir="./dataset/RCTW", is_train=True)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)


# 2.OCR模型


# 1. 文本检测模型（轻量级DBNet）
class LightDBNet(nn.Module):
    def __init__(self, img_size=(640, 640)):
        super().__init__()
        # 骨干网络：MobileNetV2
        from torchvision.models import mobilenet_v2

        self.backbone = mobilenet_v2(pretrained=True).features
        # 检测头：概率图+阈值图+近似二值图
        self.head = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 3, kernel_size=1),  # 输出3通道：prob, threshold, binary
        )

    def forward(self, x):
        x = self.backbone(x)
        out = self.head(x)
        prob_map = torch.sigmoid(out[:, 0:1, :, :])
        threshold_map = out[:, 1:2, :, :]
        binary_map = torch.sigmoid(out[:, 2:3, :, :])
        return prob_map, threshold_map, binary_map


# 2. 文本识别模型（CRNN）
class CRNN(nn.Module):
    def __init__(self, vocab_size=6000, hidden_size=256):
        super().__init__()
        # 卷积特征提取
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, kernel_size=2),
        )
        # RNN特征编码
        self.rnn = nn.LSTM(
            512, hidden_size, bidirectional=True, num_layers=2, batch_first=True
        )
        # 分类头
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.cnn(x)  # [B, 512, H', W']
        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2)  # [B, W, C, H]
        x = x.reshape(B, W, C * H)  # [B, seq_len, feature_dim]
        x, _ = self.rnn(x)  # [B, seq_len, 2*hidden_size]
        x = self.fc(x)  # [B, seq_len, vocab_size]
        return x


# 3. OCR整体模型（检测+识别）
class OCRModel(nn.Module):
    def __init__(self, vocab_size=6000):
        super().__init__()
        self.detector = LightDBNet()
        self.recognizer = CRNN(vocab_size=vocab_size)

    def forward(self, images):
        # 1. 文本检测
        prob_map, threshold_map, binary_map = self.detector(images)
        # 2. 文本裁剪（简化：实际需根据检测框裁剪RoI）
        # 此处省略NMS和RoI裁剪逻辑，训练时可分开训练检测和识别
        return prob_map, threshold_map, binary_map


# 3. 文本检测训练
def train_detector(model, train_loader, epochs=50, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # DBNet损失函数
    def db_loss(prob_map, threshold_map, binary_map, gt_boxes):
        # TODO: 实际应根据gt_boxes生成gt_prob_map、gt_threshold_map、gt_binary_map
        # 这里用全零占位，保证代码可运行
        gt_prob_map = torch.zeros_like(prob_map)
        gt_threshold_map = torch.zeros_like(threshold_map)
        gt_binary_map = torch.zeros_like(binary_map)
        loss_prob = F.binary_cross_entropy(prob_map, gt_prob_map)
        loss_threshold = F.l1_loss(threshold_map, gt_threshold_map)
        loss_binary = F.binary_cross_entropy(binary_map, gt_binary_map)
        return loss_prob + 0.5 * loss_threshold + 0.5 * loss_binary

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            gt_boxes = batch["boxes"].to(device)
            # 前向传播
            prob_map, threshold_map, binary_map = model(images)
            # 计算损失（需补充gt图生成）
            loss = db_loss(prob_map, threshold_map, binary_map, gt_boxes)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")
    # 保存检测模型
    torch.save(model.state_dict(), "light_dbnet.pth")


# 4. 文本识别训练
def train_recognizer(model, train_loader, vocab, epochs=30, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CTCLoss(blank=0)  # CTC损失

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            texts = batch["texts"]
            # 文本转token
            tokens = [vocab.text2idx(text) for text in texts]
            token_lengths = torch.tensor([len(t) for t in tokens], dtype=torch.long)
            # 前向传播
            logits = model(images)  # [B, seq_len, vocab_size]
            logits = logits.permute(1, 0, 2)  # [seq_len, B, vocab_size]
            log_probs = F.log_softmax(logits, dim=2)
            # 计算CTC损失
            input_lengths = torch.full(
                (logits.shape[1],), logits.shape[0], dtype=torch.long
            )
            loss = criterion(log_probs, tokens, input_lengths, token_lengths)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")
    # 保存识别模型
    torch.save(model.state_dict(), "crnn.pth")
