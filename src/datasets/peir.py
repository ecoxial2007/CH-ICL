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
        self.clip = ''
        self.preprocess_val = image_transform(
            (224, 224),
            is_train=True if split == 'train' else False,
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

        with open(os.path.join(self.data_path, f'{split}_en.json'), 'r') as jf:
            self.metadata = json.load(jf)

        self.text_features = h5py.File(os.path.join(self.feature_path, f'text_features{self.clip}.h5'), 'r')
        try:
            self.text_cands_features = h5py.File(os.path.join(self.data_path, 'label_features.h5'), 'r')
            self.text_cands_features = torch.tensor(self.text_cands_features['label_features'], dtype=torch.float32)
        except:
            self.text_cands_features = torch.tensor(self.text_features['label_features'], dtype=torch.float32)

        with open(os.path.join(self.data_path, f'ans2label_en.json'), 'r') as jf:
            self.answer2label = json.load(jf)

        self.label2answer = {value: key for key, value in self.answer2label.items()}
        self.num_classes = len(self.answer2label)
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
        labels_id = torch.tensor(f['match_id'])
        labels_matrix = torch.zeros(self.num_classes)
        labels_matrix.scatter_(0, labels_id, 1)

        image_path = os.path.join(self.feature_path, 'images', image_id)
        # image_features_file = h5py.File(image_feature_path, 'r')

        image = self.preprocess_val(Image.open(image_path))

        item_dict = {
            'image': image,
            'text_cands_features': self.text_cands_features,
            'labels_matrix': labels_matrix,
        }

        if self.visible:
            pass

        return item_dict