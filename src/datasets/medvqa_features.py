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
        if args.method == 'biomed':
            self.clip = ''
        elif args.method == 'pmcmed':
            self.clip = '_pmc'
        else:
            self.clip = '_pub'
        self.text_features = h5py.File(os.path.join(self.feature_path, f'text_features{self.clip}.h5'), 'r')
        self.text_cands_features = torch.tensor(self.text_features['label_features'], dtype=torch.float32)

        with open(os.path.join(self.data_path, f'ans2label_en.json'), 'r') as jf:
            self.answer2label = json.load(jf)

        self.label2answer = {value: key for key, value in self.answer2label.items()}

        with open(os.path.join('./data/Annotations/PEIR', f'ans2label_en.json'), 'r') as jf:
            self.keyword2label = json.load(jf)


        self.label2keyword = {value: key for key, value in self.keyword2label.items()}
        self.num_classes = len(self.label2keyword)
        self.keyword_features = h5py.File(os.path.join('./data/PEIR', f'text_features.h5'), 'r')
        self.keyword_features = torch.tensor(self.keyword_features['label_features'], dtype=torch.float32)



        image_ids = []
        for f in self.metadata:
            image_ids.append(f['img_id'])
        self.image_ids = list(set(image_ids))
        print(self.split, len(self.metadata))
        print('num_classes', len(self.answer2label))
        print('num_images', len(self.image_ids))


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

        if 'PathVQA' in self.data_path:
            if 'train' in image_id:
                image_path = os.path.join(self.feature_path, 'images', f'train', image_id+'.jpg')
            elif 'val' in image_id:
                image_path = os.path.join(self.feature_path, 'images', f'val', image_id + '.jpg')
            else:
                image_path = os.path.join(self.feature_path, 'images', f'test', image_id + '.jpg')
        else:
            image_path = os.path.join(self.feature_path, 'images', image_id+'.jpg')

        image_feature_path = image_path.replace('images', f'features{self.clip}').replace('.jpg', '.h5')
        image_features_file = h5py.File(image_feature_path, 'r')

        image_features = torch.tensor(np.array(image_features_file['feature']), dtype=torch.float32)  # (L_video, D_in); L_video >> L
        patch_features = torch.tensor(np.array(image_features_file['feature_noproj']), dtype=torch.float32)  # (L_video, D_in); L_video >> L

        text_query_features = torch.tensor(self.text_features['question_tokens'][qid], dtype=torch.float32)
        text_query_features_global = torch.tensor(self.text_features['question_features'][qid], dtype=torch.float32)

        item_dict = {
            'image_features': image_features,
            'patch_features': patch_features,
            'text_query_features': text_query_features_global,
            'text_query_token_features': text_query_features,
            'text_cands_features': self.text_cands_features,
            'labels_id': labels_id,
        }

        if self.visible:
            item_dict.update({
                'additional_info': (image_id, question, qid, answer),
                'atype': atype
            })

        return item_dict