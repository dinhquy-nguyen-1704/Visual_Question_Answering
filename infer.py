import torch
import open_clip
from CLIP.encoder import TextEncoder, VisualEncoder
from CLIP.classifier import Classifier
from CLIP.VQAModel import VQAModel
from src.predict import predict
from src.config import get_config

def main():

    config = get_config()

    img_path = config.img_path
    question = config.question

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_clip, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_e16')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model_clip.to(device)

    text_encoder = TextEncoder(model_clip).to(device)
    visual_encoder = VisualEncoder(model_clip).to(device)
    classifier = Classifier(input_size=512*2, hidden_size=1024, n_classes=2, n_layers=1, dropout=0.2).to(device)

    model = VQAModel(visual_encoder, text_encoder, classifier).to(device)
    if device == 'cpu':
        model.load_state_dict(torch.load('trained_model_clip_32B_laion2b_e16.pth', map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load('trained_model_clip_32B_laion2b_e16.pth'))
    model.eval()

    predicted_label = predict(model, img_path, question, tokenizer, preprocess, device)
    print("The answer is:", predicted_label)

if __name__ == '__main__':
    main()
