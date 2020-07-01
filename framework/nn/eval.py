import os

import torch
from torchvision import models
import torch.optim as optim

from state_clf.nn.state_clf_nn import BrainNNPretrained
from state_clf.nn.utils import load_ckp
from state_clf.data_prep.torchDataLoader import BrainEEGImgDataLoader


NN_MODELS_CHECKPOINT_PATH = os.path.abspath(
    os.path.normpath(os.path.join(__file__, "../", "../pretrained/nn_checkpoints/")))


if __name__=="__main__":

    checkpoint_fpath = os.path.abspath(
        os.path.normpath(os.path.join(NN_MODELS_CHECKPOINT_PATH, "model_1s_vgg_epoch_200.pt")))

    pretrained_model = models.vgg16(pretrained=True)
    model = BrainNNPretrained(pretrained_model=pretrained_model, output_size=2)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    model, optimizer, checkpoint, valid_loss_min = load_ckp(checkpoint_fpath, model, optimizer)

    dl = BrainEEGImgDataLoader(
        config_path="./../configs/2d_manual_125.json",
        data_files_path="./../../data/EEG tubingen/TMS-EEG_Tubingen_ver2.mat",
        pca_model_name="PCAmodel_clf1s"
    )

    train_loader, val_loader = dl.get_loaders(
        train_batch_size=128,
        test_batch_size=128)

    n_correct_pred = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            target_index = torch.max(target, 1)[1]

            # loss calculation
            net_out = model(data)

            pred = net_out.round().detach()
            target = target.float().round().detach()
            n_correct_pred += pred.eq(target).sum().item()

    print(n_correct_pred / len(val_loader.dataset))

