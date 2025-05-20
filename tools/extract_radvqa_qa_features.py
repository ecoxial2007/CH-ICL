import json
import glob
import os.path
import h5py
import torch
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from open_clip.tokenizer import HFTokenizer
from open_clip.model import CustomTextCLIP
from open_clip.transform import image_transform

cast_dtype = 'fp16'
config_path = './src/backbone/BiomedCLIP/open_clip_config.json'
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

pretrained_cfg = config['preprocess_cfg']
model_cfg = config['model_cfg']

model = CustomTextCLIP(**model_cfg, cast_dtype=cast_dtype)

model.load_state_dict(torch.load('./src/backbone/BiomedCLIP/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224.pth'))
tokenizer = HFTokenizer("./src/backbone/BiomedCLIP/tokenizer")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()



preprocess_val = image_transform(
    model.visual.image_size,
    is_train=False,
    mean=getattr(model.visual, 'image_mean', None),
    std=getattr(model.visual, 'image_std', None),
)

dataset_path = './data/Annotations/VQA-RAD/'
feature_path = './data/VQA-RAD/'

splits = ['train', 'test']

save_path = 'text_features.h5'
candidate_features = []
question_features = []
question_tokens = []
with open(os.path.join(dataset_path, f'ans2label_en.json'), 'r') as ff:
    answerlist = json.load(ff)
import time

for ans in answerlist.keys():
    start = time.time()
    text_caption_input = tokenizer(f"{ans}").to(device)
    text_caption_features = model.encode_text(text_caption_input)
    print('inference text:', time.time() - start)
    candidate_features.append(text_caption_features[0].detach().cpu())

for split in splits:
    json_path = os.path.join(dataset_path, f'{split}_en.json')
    with open(json_path, 'r') as f:
        metadata = json.load(f)

    for meta in metadata:
        qid = meta['qid']
        ques = meta['question']

        text_caption_input = tokenizer(f"{ques}").to(device)
        text_caption_features = model.encode_text(text_caption_input)
        print(text_caption_features[0].shape, text_caption_features[1].shape)
        question_features.append(text_caption_features[0].detach().cpu())
        question_tokens.append(text_caption_features[1].detach().cpu())


candidate_features = torch.cat(candidate_features, dim=0)
question_features = torch.cat(question_features, dim=0)
question_tokens = torch.cat(question_tokens, dim=0)

with h5py.File(save_path, 'w') as f:
    f.create_dataset('label_features', data=candidate_features.numpy())
    f.create_dataset('question_features', data=question_features.numpy())
    f.create_dataset('question_tokens', data=question_tokens.numpy())

