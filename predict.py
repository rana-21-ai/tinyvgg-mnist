import torch
from torchvision import transforms
from PIL import Image
from model import TinyVGG

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TinyVGG()
model.load_state_dict(torch.load("tinyvgg_mnist.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

image = Image.open("sample.png")
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)
    prediction = output.argmax(dim=1).item()

print("Predicted digit:", prediction)
