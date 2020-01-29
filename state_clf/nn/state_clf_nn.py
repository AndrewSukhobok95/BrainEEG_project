import os
import sys
_project_path = os.path.abspath(
    os.path.normpath(os.path.join(__file__, "../", "../../")))
sys.path.append(_project_path)

import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from state_clf.data_prep.torchDataLoader import BrainEEGImgDataLoader
from state_clf.nn.utils import save_ckp

# _main_file_path = sys.modules['__main__'].__file__
NN_MODELS_CHECKPOINT_PATH = os.path.abspath(
    os.path.normpath(os.path.join(__file__, "../", "../pretrained/nn_checkpoints/")))
if not os.path.exists(NN_MODELS_CHECKPOINT_PATH):
    os.makedirs(NN_MODELS_CHECKPOINT_PATH)


class BrainNNPretrained(nn.Module):
    def __init__(self, pretrained_model, output_size=5):
        super(BrainNNPretrained, self).__init__()
        self.features = nn.Sequential(*list(pretrained_model.features.children())[:-7])

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.fc1 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.fc3 = nn.Linear(in_features=4096, out_features=output_size, bias=True)

    def forward(self, x):
        x = self.features(x.float())
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.softmax(self.fc3(x), dim=0)
        return output



class NNmodule(object):
    def __init__(self, output_size=5,
                 pretrined_model_name="vgg",
                 model_name="model",
                 fix_pretrained_params=True,
                 use_cuda=False):
        if pretrined_model_name=="vgg":
            self.pretrained_model = models.vgg16(pretrained=True)
        else:
            raise NotImplementedError("Model with this name is not implemented.")

        self.model_name = model_name + "_" + pretrined_model_name

        if fix_pretrained_params:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

        self.model = BrainNNPretrained(
            pretrained_model=self.pretrained_model,
            output_size=output_size)

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model = self.model.cuda()

        self.LossHistoryTrain = []
        self.LossHistoryTrainBatch = []
        self.LossHistoryVal = []
        self.LossHistoryValBatch = []
        self.ValAccuracy = None

    def fit(self, train_loader, val_loader=None, lr=0.01, n_epochs=500):
        # create a stochastic gradient descent optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        # create a loss function
        criterion = nn.CrossEntropyLoss()
        # sets model to TRAIN mode
        self.model.train()
        # run the main training loop
        for epoch in range(1, n_epochs + 1):
            train_loss = 0.0
            valid_loss = 0.0

            for batch_idx, (data, target) in enumerate(train_loader):
                # convert target to index for CrossEntropyLoss
                # https://discuss.pytorch.org/t/runtimeerror-multi-target-not-supported-newbie/10216/2
                target_index = torch.max(target, 1)[1]
                # move to GPU
                if self.use_cuda:
                    data, target_index = data.cuda(), target_index.cuda()
                # propagation
                net_out = self.model(data)
                loss = criterion(net_out, target_index)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                self.LossHistoryTrainBatch.append(loss.item())

                train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            # calculate average losses
            train_loss = train_loss / len(train_loader.dataset)
            self.LossHistoryTrain.append(train_loss)

            if val_loader is not None:
                n_correct_pred = 0

                # https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615
                self.model.eval()
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(val_loader):
                        target_index = torch.max(target, 1)[1]
                        # move to GPU
                        if self.use_cuda:
                            data, target_index = data.cuda(), target_index.cuda()
                        # loss calculation
                        net_out = self.model(data)
                        val_loss = criterion(net_out, target_index)

                        self.LossHistoryTrainBatch.append(val_loss.item())

                        valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (val_loss.data - valid_loss))

                        if epoch == n_epochs:
                            # pred = np.round(net_out.detach().cpu().numpy())
                            # target = np.round(target.detach().cpu().numpy())
                            pred = net_out.round().detach().cpu()
                            target = target.float().round().detach().cpu()
                            n_correct_pred += pred.eq(target).sum().item()

                valid_loss = valid_loss / len(val_loader.dataset)
                self.LossHistoryVal.append(valid_loss)
                if epoch==n_epochs:
                    self.ValAccuracy = n_correct_pred / len(val_loader.dataset)

            # print training/validation statistics
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch, train_loss, valid_loss))

            if epoch%100==0:
                checkpoint = {
                    'epoch': epoch,
                    'valid_loss_min': valid_loss,
                    'state_dict': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                checkpoint_path = os.path.join(
                    NN_MODELS_CHECKPOINT_PATH, "{}_epoch_{}.pt".format(self.model_name, epoch))
                save_ckp(checkpoint, checkpoint_path=checkpoint_path)

        print('Final Validation Accuracy = {:.6f}'.format(self.ValAccuracy))

        self.plot_avg_trainig_curve()
        self.plot_batch_trainig_curve()
        self.plot_batch_validation_curve()

    def plot_avg_trainig_curve(self):
        fig = plt.figure(figsize=(15,10))
        plt.plot(self.LossHistoryTrain, label="train")
        plt.plot(self.LossHistoryVal, label="val")
        fig.savefig('./avg_training_curve.png')

    def plot_batch_trainig_curve(self):
        fig = plt.figure(figsize=(15,10))
        plt.plot(self.LossHistoryTrainBatch, label="train")
        fig.savefig('./avg_batch_training_curve.png')

    def plot_batch_validation_curve(self):
        fig = plt.figure(figsize=(15,10))
        plt.plot(self.LossHistoryValBatch, label="train")
        fig.savefig('./avg_batch_validation_curve.png')




if __name__=="__main__":

    use_cuda = torch.cuda.is_available()

    print("Using CUDA:", use_cuda)

    dl = BrainEEGImgDataLoader(
        config_path="./../configs/2d_manual_125.json",
        data_files_path="./../../data/EEG tubingen/TMS-EEG_Tubingen_ver2.mat",
        pca_model_name="PCAmodel_clf1s"
    )

    train_loader, val_loader = dl.get_loaders(
        train_batch_size=128,
        test_batch_size=128,
        fit_transformers=False
    )

    print("Model init")

    model = NNmodule(output_size=2, model_name="model_1s", use_cuda=use_cuda)

    print("Start training")

    model.fit(train_loader, val_loader, n_epochs=500)

    print()
