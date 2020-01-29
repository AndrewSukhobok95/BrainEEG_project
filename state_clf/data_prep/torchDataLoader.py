import os
import sys
import json

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms

from state_clf.data_prep.constructor import ImgRepresentationCreator
from state_clf.data_prep.data_reader import DataReaderMatLab
from state_clf.data_prep.transformer import FFTtoPCA_transformer
from state_clf.data_prep.pipeline import PipeLine



class BrainEEGImgDataset(Dataset):
    def __init__(self, pipeline, transforms=None):
        self.pipeline = pipeline
        self.transforms = transforms

    def __getitem__(self, index):
        x, y = self.pipeline.get_obs(index=index)
        if self.transforms is not None:
            x = self.transforms(x)
        return x, y

    def __len__(self):
        return self.pipeline.get_number_of_obs()



class BrainEEGImgDataLoader(object):
    def __init__(self, config_path, data_files_path, pca_model_name):
        self.main_file_path = sys.modules['__main__'].__file__
        self.json_config_path = os.path.abspath(os.path.normpath(
            os.path.join(self.main_file_path, "../", config_path)))
        self.file_path = os.path.abspath(os.path.normpath(
            os.path.join(self.main_file_path, "../", data_files_path)))
        self.pca_model_name = pca_model_name

    def get_loaders(self, train_batch_size=8,
                    test_batch_size=8,
                    fit_transformers=False):
        with open(self.json_config_path, 'r') as f:
            config = json.load(f)

        constructor_irc = ImgRepresentationCreator(config)
        data_reader = DataReaderMatLab(self.file_path)
        transformer = FFTtoPCA_transformer(pca_models_name=self.pca_model_name)

        # Returns the table of sensors location on the image
        # img_df = irc.get_coordinates_table()

        pipeline_braineeg = PipeLine(
            constructor=constructor_irc,
            data_reader=data_reader,
            transformer=transformer)

        if fit_transformers:
            pipeline_braineeg.fit_transformers()

        torch_transform = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])

        braineeg_dataset = BrainEEGImgDataset(pipeline=pipeline_braineeg, transforms=torch_transform)

        n = pipeline_braineeg.get_number_of_obs()
        train_dataset, val_dataset = random_split(braineeg_dataset, [int(n * 0.9), n-int(n * 0.9)])

        train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size)
        val_loader = DataLoader(dataset=val_dataset, batch_size=test_batch_size)

        return train_loader, val_loader





if __name__ == '__main__':

    # dl = BrainEEGImgDataLoader()
    # train_loader, val_loader = dl.get_loaders()

    print()
