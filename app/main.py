from flask import Flask, render_template, request
from torchvision import models, transforms
from PIL import Image
import torch
import os
import glob

class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(num_classes=6)
checkpoint = torch.load('model.pth', map_location=torch.device(device))
new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
model.load_state_dict(new_state_dict)
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store'
    return response

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        files = glob.glob('static/uploads/*')
        for f in files:
            os.remove(f)

        file = request.files['file']
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        image = Image.open(file_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = outputs.max(1)

        return render_template('upload.html', prediction=class_names[predicted.item()], filename=filename)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(port=5000, debug=True)
