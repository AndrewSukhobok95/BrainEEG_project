import torch
import shutil


# https://www.kaggle.com/vortanasay/saving-loading-and-cont-training-model-in-pytorch?scriptVersionId=27394631

def save_ckp(state, checkpoint_path):
    f_path = checkpoint_path
    torch.save(state, f_path)


def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath, map_location = torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    valid_loss_min = checkpoint['valid_loss_min']
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()



if __name__=="__main__":
    print()