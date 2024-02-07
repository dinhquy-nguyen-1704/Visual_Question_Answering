from PIL import Image
import torch

def predict(model, img_path, question, tokenizer, preprocess, device):
    # Load and preprocess image
    image = Image.open(img_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)

    # Tokenize question
    question = tokenizer(question).squeeze(0).to(device)
    question = question.reshape(1, -1)

    # Perform prediction
    with torch.no_grad():
        outputs = model(image, question)
        predicted_label = torch.argmax(outputs, dim=1).item()
        predicted_label = "No" if predicted_label == 1 else "Yes"

    return predicted_label
