import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageFilter
import os


# 定义一个简单但有效的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# 训练模型
def train_model():
    print("开始训练模型...")

    # 设置训练参数
    batch_size = 64
    epochs = 10
    learning_rate = 0.001

    # 数据预处理
    transform = transforms.Compose([
        transforms.RandomRotation(10),  # 随机旋转
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机平移
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载MNIST数据集
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)

    # 初始化模型和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        # 每个epoch结束后在测试集上验证
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'测试集: 平均损失: {test_loss:.4f}, 准确率: {correct}/{len(test_loader.dataset)} '
              f'({accuracy:.2f}%)')

    # 保存模型
    torch.save(model.state_dict(), "mnist_cnn.pt")
    print("模型训练完成并已保存")
    return model


# 加载预训练模型
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)

    if os.path.exists("mnist_cnn.pt"):
        model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
        print("已加载预训练模型")

        # 验证模型性能
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        print(f"模型在测试集上的准确率: {100. * correct / total:.2f}%")
    else:
        print("未找到预训练模型")

    model.eval()
    return model


# 增强的图像预处理函数
def preprocess_image(image):
    # 保持原始绘制颜色（黑底白字）
    img_array = np.array(image)

    # 直接调整大小为28x28
    img_resized = image.resize((28, 28), Image.LANCZOS)

    # 应用轻微高斯模糊以模拟MNIST风格
    img_blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=0.5))

    return img_blurred


# 创建GUI应用程序
class DrawingApp:
    def __init__(self, root, model):
        self.root = root
        self.root.title("手写数字识别")
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 设置画布
        self.canvas_width = 280
        self.canvas_height = 280
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg="black")
        self.canvas.pack(pady=10)

        # 设置识别结果显示
        self.result_label = tk.Label(root, text="请绘制数字", font=("Arial", 20))
        self.result_label.pack(pady=5)

        # 设置置信度显示
        self.confidence_label = tk.Label(root, text="", font=("Arial", 12))
        self.confidence_label.pack(pady=5)

        # 设置按钮框架
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        # 识别按钮
        self.recognize_button = tk.Button(button_frame, text="识别", command=self.recognize_drawing, width=10)
        self.recognize_button.pack(side=tk.LEFT, padx=10)

        # 清除按钮
        self.clear_button = tk.Button(button_frame, text="清除", command=self.clear_canvas, width=10)
        self.clear_button.pack(side=tk.LEFT, padx=10)

        # 预览按钮
        self.preview_button = tk.Button(button_frame, text="预览", command=self.show_preview, width=10)
        self.preview_button.pack(side=tk.LEFT, padx=10)

        # 绑定鼠标事件
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        # 初始化绘图
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None

    def paint(self, event):
        x, y = event.x, event.y
        radius = 15  # 增加画笔半径，使绘制更容易

        if self.last_x and self.last_y:
            # 绘制线条
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=radius * 2,
                                    fill="white", capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw.line([self.last_x, self.last_y, x, y], fill=255, width=radius * 2)

            # 绘制端点圆形，使线条更平滑
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="white", outline="white")
            self.draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=255)

        self.last_x = x
        self.last_y = y

    def reset(self, event):
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="请绘制数字")
        self.confidence_label.config(text="")

    def show_preview(self):
        # 预处理图像并显示预览
        processed_img = preprocess_image(self.image)
        processed_img.show()

    def recognize_drawing(self):
        # 预处理用户绘制的图像
        processed_img = preprocess_image(self.image)

        # 转换为Tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        img_tensor = transform(processed_img).unsqueeze(0).to(self.device)

        # 使用模型进行预测
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, pred = torch.max(probabilities, 1)
            predicted_number = pred.item()
            confidence_value = confidence.item()

        # 显示识别结果和置信度
        self.result_label.config(text=f"识别结果: {predicted_number}")
        self.confidence_label.config(text=f"置信度: {confidence_value:.2%}")

        # 显示所有数字的概率
        confidence_text = "置信度:\n"
        probs = probabilities.cpu().squeeze().numpy()
        for i, prob in enumerate(probs):
            confidence_text += f"{i}: {prob:.3f}  "
            if (i + 1) % 5 == 0:
                confidence_text += "\n"
        self.confidence_label.config(text=confidence_text)


# 主程序
def main():
    # 检查是否有预训练模型，如果没有则训练一个
    try:
        model = load_model()
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("训练新模型...")
        model = train_model()
        print("模型训练完成")

    # 创建GUI
    root = tk.Tk()
    app = DrawingApp(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()
