"""
Simple video-language dataset class file, for illustration purposes. See comments below for relevant highlights.
"""
import json
import os
import random
from PIL import Image
import torch
from torch.utils import data
import h5py
import numpy as np
from open_clip.transform import image_transform


class ImageLanguageDataset(data.Dataset):
    """
    Example (simple) video-language dataset class, for illustration purposes for ATP training.
    """

    def __init__(self, args, split="train"):
        super().__init__()
        self.data_path = args.data_path
        self.feature_path = args.feature_path
        self.split = split
        self.visible = args.visible
        self.preprocess_val = image_transform(
            (224, 224),
            is_train=False,#True if split=='train' else False,
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

        with open(os.path.join(self.data_path, f'{split}_en.json'), 'r') as jf:
            self.metadata = json.load(jf)

        self.clip = ''
        self.text_features = h5py.File(os.path.join(self.feature_path, f'text_features{self.clip}.h5'), 'r')
        self.text_cands_features = torch.tensor(self.text_features['label_features'], dtype=torch.float32)

        with open(os.path.join(self.data_path, f'ans2label_en.json'), 'r') as jf:
            self.answer2label = json.load(jf)

        self.label2answer = {value: key for key, value in self.answer2label.items()}
        ###### keyword
        with open(os.path.join('./data/Annotations/PEIR', f'ans2label_en.json'), 'r') as jf:
            self.keyword2label = json.load(jf)
        self.label2keyword = {value: key for key, value in self.keyword2label.items()}

        self.keyword_features = h5py.File(os.path.join('./data/PEIR', f'text_features.h5'), 'r')
        self.keyword_features = torch.tensor(self.keyword_features['label_features'], dtype=torch.float32)

        image_ids = []
        for f in self.metadata:
            image_ids.append(f['img_id'])
        self.image_ids = list(set(image_ids))
        print(self.split, len(self.metadata))
        print('num_classes', len(self.answer2label))
        print('num_images', len(self.image_ids))
        self.num_keywords = len(self.keyword2label)


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        """
        Assuming torch files for each of the features for simplicity; adjust to fit your extracted features.
        (e.g. numpy, hdf5, json, etc.)
        """
        f = self.metadata[index]
        qid = int(f['qid'])
        image_id = f['img_id']
        image_id = os.path.splitext(image_id)[0]
        question = f['question']
        answer = str(f['answer'])
        atype = f["answer_type"]

        labels_id = torch.tensor(self.answer2label[answer], dtype=torch.long)
        image_path = os.path.join(self.feature_path, 'images', image_id+'.jpg')
        image = self.preprocess_val(Image.open(image_path))

        text_query_features = torch.tensor(self.text_features['question_tokens'][qid], dtype=torch.float32)
        text_query_features_global = torch.tensor(self.text_features['question_features'][qid], dtype=torch.float32)

        item_dict = {
            'image': image,
            'text_query_features': text_query_features_global,
            'text_query_token_features': text_query_features,
            'text_cands_features': self.keyword_features,
            'labels_id': labels_id,
        }

        if self.visible:
            item_dict.update({
                'additional_info': (image_id, question, qid, answer),
                'atype': atype
            })

        return item_dict