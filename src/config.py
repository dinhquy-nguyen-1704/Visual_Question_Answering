
import argparse

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='ViT_RoBERTa')
    parser.add_argument('--data_path', type=str, default='./data/val2014-resised')
    parser.add_argument('--train_data_path', type=str, default='./data/vaq2.0.TrainImages.txt')
    parser.add_argument('--val_data_path', type=str, default='./data/vaq2.0.DevImages.txt')
    parser.add_argument('--test_data_path', type=str, default='./data/vaq2.0.TestImages.txt')
    parser.add_argument('--train_batch_size', type=int, default=512)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--scheduler_step_size', type=float, default=1)
    parser.add_argument('--img_feature_extractor_name', type=str, default='google/vit-base-patch16-224')
    parser.add_argument('--text_tokenizer_name', type=str, default='roberta-base')
    parser.add_argument('--clip_model_type', type=str, default='ViT-B-32')
    parser.add_argument('--clip_pretrained', type=str, default='laion2b_e16')
    parser.add_argument('--cnn_model_name', type=str, default='resnet50')
    parser.add_argument('--img_path', type=str, default='None')  
    parser.add_argument('--question', type=str, default='None')       

    args = parser.parse_args()

    return args
