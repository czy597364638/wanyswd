import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader

data_dir = '/path/to/coco'
train_ann_file = f'{data_dir}/annotations/instances_train2017.json'
train_img_dir = f'{data_dir}/train2017/'

train_dataset = CocoDetection(
    root=train_img_dir,
    annFile=train_ann_file,
    transform=lambda img, target: (F.to_tensor(img), target)
)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# 加载预训练的 Faster R-CNN 模型
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义损失函数和优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 前向传播
        loss_dict = model(images, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs} completed.')

# 保存模型
torch.save(model.state_dict(), 'faster_rcnn_coco.pth')