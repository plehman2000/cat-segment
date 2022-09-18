#RUNNING ON ENVIRONMENT 'UNEXT'
# Config
seed = 42  # for reproducibility
training_split_ratio = 0.9  # use 90% of samples for training, 10% for testing
num_epochs = 5
# If the following values are False, the models will be downloaded and not computed
compute_histograms = False
train_whole_images = False 
train_patches = False
import uuid
from IPython.display import display
import enum
import time
import random
import multiprocessing
from pathlib import Path

import torch
torch.cuda.empty_cache()

import torchvision
import torchio as tio
import torch.nn.functional as F

import numpy as np
from unet import UNet
from scipy import stats
import matplotlib.pyplot as plt

from IPython.display import display
from tqdm.auto import tqdm

random.seed(seed)
torch.manual_seed(seed)
plt.rcParams['figure.figsize'] = 12, 6

import pytorch_lightning as pl
import os
import time
# torch.multiprocessing.set_start_method('spawn', force=True)







DEBUG = False










def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

#TODO: CONVERT TO THIS FN
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()      
        
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
#         print(f"BCE Input type: {type(inputs)} |{inputs}|, Target type: {type(targets)} |{targets}|")
        BCE = F.binary_cross_entropy(inputs, targets.float(), reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
    
#PyTorch
ALPHA = 0.5
BETA = 0.5

class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky
ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets.float(), reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss
def tensor_to_video(inputs, name="TEST2", imname="s", repeat=3, label=False):
    print(inputs.size())
    if label:
#         0: Background (None of the following organs)
#         1: Liver
#         2: Bladder
#         3: Lungs
#         4: Kidneys
#         5: Bone
#         6: Brain
        slices = []
        for i in range(inputs.size()[3]):
            slice = inputs[:,:,i]
            img = Ft.to_pil_image(slice, mode='L')
            print(slice.size())
            if i==16:
                img.save(f"{imname}.png")
            for _ in range(repeat):
                slices.append(img)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video = cv2.VideoWriter(f'{name}_label.mp4', fourcc, inputs.size()[3], (inputs.size()[1]  , inputs.size()[2]))

    else:
        slices = []
        for i in range(inputs.size()[3]):
            slice = inputs[0,:,:,i]
            img = Ft.to_pil_image(slice, mode='L')
            # print(slice.size())
            if i==16:
                img.save(f"{imname}.png")
            for _ in range(repeat):
                slices.append(img)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video = cv2.VideoWriter(f'{name}.mp4', fourcc, inputs.size()[3], (inputs.size()[1]  , inputs.size()[2]))

#     video = cv2.VideoWriter(f'{name}.mp4', fourcc, inputs.size()[4], (inputs.size()[3]  , inputs.size()[2]))
    for j in slices:
        j = np.array(j)
        shape = np.shape(j)
        opencvImage = cv2.cvtColor(j, cv2.COLOR_GRAY2BGR)
        video.write(opencvImage)

    cv2.destroyAllWindows()
    video.release()
    
#non class functions
def numpy_reader(path):
    data = np.load(path, allow_pickle=True)
    affine = np.eye(4)
    return data, affine
import cv2
import numpy as np
  
import torchvision.transforms.functional as Ft

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    print(batch)
    [Subject(Keys: ('sample', 'label'); images: 2), Subject(Keys: ('sample', 'label'); images: 2)]
    return torch.utils.data.dataloader.default_collate(batch)


PATH = "/home/patricklehman/MRI"
# RESAMPLE = 1 #4

class CATDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self._has_setup_0 = True
        self.image_shape = wandb.config['image_shape']
        self.training_subjects = []
        self.validation_subjects = []
        self.batch_size = batch_size
        self.dataset = []
        self.num_workers = wandb.config['num_workers']
        self.training_split_ratio = 0.9
        self.persistent_workers = wandb.config['persistent_workers']
        self.debug = DEBUG
        
    def prepare_data(self):
        dataset_dir = Path(PATH +  "/orange/org/augfix") #switched from augfix due to one hot error
        images_dir = dataset_dir / 'images'
        labels_dir = dataset_dir / 'labels'
        image_paths = sorted(images_dir.glob('*.npy'))
        label_paths = sorted(labels_dir.glob('*.npy'))
        assert len(image_paths) == len(label_paths)
        augmented_dataset = []
        count = 0
        for (image_path, label_path) in tqdm(zip(image_paths, label_paths), total=len(image_paths)):
            subject = tio.Subject(
                sample=tio.ScalarImage(image_path, reader = numpy_reader),
                label =  tio.LabelMap(label_path, reader = numpy_reader))
            self.dataset.append(subject)
            tens = tio.LabelMap(label_path, reader = numpy_reader).keys()
            count +=1
            if self.debug and count ==1: #debugging DO NOT LEAVE TODO
                break

    def setup(self, stage):

        num_training_subjects = int(self.training_split_ratio * len(self.dataset))
        num_validation_subjects = len(self.dataset) - num_training_subjects
        num_split_subjects = [num_training_subjects, num_validation_subjects]
        #WHY IS THIS NUMBER WRONG?
        self.training_subjects, self.validation_subjects = torch.utils.data.random_split(self.dataset, num_split_subjects)
#         del self.dataset



    def train_dataloader(self):
        training_set = tio.SubjectsDataset(self.training_subjects, transform = tio.CropOrPad(self.image_shape))
        training_loader = torch.utils.data.DataLoader(
                            training_set,
                            batch_size=self.batch_size ,
                            shuffle=True,
                            num_workers=self.num_workers,
                            worker_init_fn=seed_worker,
                            persistent_workers=self.persistent_workers,
                            collate_fn=collate_fn)
            
        return training_loader

    def val_dataloader(self):
        validation_set = tio.SubjectsDataset(self.validation_subjects,transform = tio.CropOrPad(self.image_shape))
        validation_loader = torch.utils.data.DataLoader(
                        validation_set,
                        batch_size=self.batch_size ,
                        shuffle=False,
                        num_workers=self.num_workers,
                        worker_init_fn=seed_worker,
                        persistent_workers=self.persistent_workers,
                        collate_fn=collate_fn)
        
        return validation_loader
    
    def pred_dataloader(self, subjects = None):
        if subjects is None:
            pred_set = list(self.validation_subjects)[0:4]
        else:
            pred_set = subjects
#         print(list(self.validation_subjects))
        pred_set = list(self.validation_subjects)[0:4]

        validation_set = tio.SubjectsDataset(pred_set, transform = tio.CropOrPad(self.image_shape))
      
        validation_loader = torch.utils.data.DataLoader(
                        validation_set,
                        batch_size=1,
                        shuffle=True,
                        worker_init_fn=seed_worker,
                        num_workers=self.num_workers,
                        collate_fn=collate_fn)
            
        return validation_loader
    
CHANNELS_DIMENSION = 1
SPATIAL_DIMENSIONS = 2, 3, 4
from unet import UNet2D, UNet3D
import torch.nn as nn
MODEL_PATH = "/home/patricklehman/MRI/model"
class Segmenter(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = UNet3D(
        in_channels=1,
        out_classes=6,
        out_channels_first_layer=wandb.config['out_channels'],
        normalization=wandb.config['normalization'], 
        preactivation=wandb.config['preactivation'],
        residual=wandb.config['residual'],
        num_encoding_blocks=wandb.config['num_encoding_blocks'],
        upsampling_type='trilinear')
#         self.save_hyperparameters()
        self.train_loss = 0
        self.val_loss = 0
        self.criterion = wandb.config['criterion']()#nn.CrossEntropyLoss()
#         self.run = wand_run
    def forward(self, x):
        return self.model(x)
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=wandb.config['learning_rate'])
        return optimizer
    def training_step(self, train_batch, batch_index):
        inputs = train_batch['sample'][tio.DATA]
        targets = train_batch['label'][tio.DATA].to(torch.float16)
        if targets.shape[1] != 6:
            return None
        with torch.enable_grad():
            logits = self.model(inputs)
            probabilities = F.softmax(logits, dim=CHANNELS_DIMENSION)
            batch_losses = self.criterion(probabilities, targets.float())
            batch_loss = batch_losses.mean()
#         dic = {'train_loss': batch_loss}
#         self.log(dic)
        self.log("train/loss", batch_loss, sync_dist=True, batch_size=wandb.config['batch_size'])
        
        return batch_loss
   
    def validation_step(self, val_batch, batch_index):
        inputs = val_batch['sample'][tio.DATA]
        targets = val_batch['label'][tio.DATA].to(torch.float16)
#         print(f"VAL: {inputs.shape} {targets.shape}")
        if targets.shape[1] != 6:
            return None
        with torch.no_grad():
            logits = self.model(inputs)
            probabilities = F.softmax(logits, dim=CHANNELS_DIMENSION)
            batch_losses = self.criterion(probabilities, targets.float())
            batch_loss = batch_losses.mean()
#         dic = {'val_loss': batch_loss}
#         self.log(dic)
        self.log("val/loss", batch_loss,sync_dist=True,  batch_size=wandb.config['batch_size'])
        tag = time.ctime().replace(":", "_").replace(" ", "_")
        torch.save({'state_dict': self.model.state_dict()}, f"{MODEL_PATH}/{batch_index}{str(uuid.uuid4())[:5]}")
        return batch_loss
    
    def predict_step(self, batch=None,raw_tens=None, batch_idx=0, datloader_idx=0):
        if raw_tens != None:
            inputs = raw_tens
        else:
            inputs = batch['sample'][tio.DATA]
            targets = batch['label'][tio.DATA]
        with torch.no_grad():
            logits = self.model(inputs)
            probabilities = F.softmax(logits, dim=CHANNELS_DIMENSION)
            if raw_tens != None:
                return probabilities
            batch_losses = self.criterion(probabilities, targets.float())
            batch_loss = batch_losses.mean()
        return inputs, targets, probabilities, batch_loss


        


import wandb


from pytorch_lightning import Trainer, seed_everything

from pytorch_lightning.callbacks import TQDMProgressBar
# os.environ["PL_DISABLE_FORK"] = "1"




        


import wandb


from pytorch_lightning import Trainer, seed_everything

from pytorch_lightning.callbacks import TQDMProgressBar
# os.environ["PL_DISABLE_FORK"] = "1"





def main():
    from pytorch_lightning.loggers import WandbLogger
    
    wandblogger = WandbLogger(project="3dUnet", group=f"loss-bce")
    wandb.config = {

        #Model Parameters
        "normalization": 'batch',
        "preactivation": True,
        "residual": True,
        "num_encoding_blocks": 2, #4 causes error
        "upsampling_type": 'trilinear',
        "activation": "PReLU",
        "criterion": nn.BCELoss,
        #Training Parameters
        "out_channels": 32, # normally 32
        "learning_rate": 1e-4,
        "min_epochs": 1,
        "batch_size": 2, #8 causes error
        "image_shape": (256,256,50), #3rd number must be even?
        "num_workers": 1,
        "persistent_workers": False,
        "max_epochs": 100
    }

    print('Last run on', time.ctime())
    print('TorchIO version:', tio.__version__)
    print('Torch version:', torch.__version__)

    torch.cuda.empty_cache()
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "70000"
    seed_everything(42, workers=True)
    datamodule = CATDataModule(batch_size=wandb.config['batch_size'])
    datamodule.prepare_data()
    datamodule.setup(0)
    print(f"DATA SET UP, Length is {len(datamodule.dataset)}")
    
    from pytorch_lightning.strategies.ddp import DDPStrategy
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.callbacks import ModelCheckpoint

    model = Segmenter()
    saver = ModelCheckpoint(dirpath=f"{MODEL_PATH}/", filename=f"{str(uuid.uuid4())[:5]}")
   
    
#     model.load_from_checkpoint(MODEL_PATH + "/af2cd-v1.ckpt")
    

    calls = [saver]#, EarlyStopping(monitor="val/loss", mode="min")]

    trainer = pl.Trainer( callbacks=calls , min_epochs=wandb.config['min_epochs'],max_epochs=wandb.config['max_epochs'],logger=wandblogger,
                     log_every_n_steps=1,
                     accelerator='gpu',fast_dev_run=False,devices=1,
                     default_root_dir=MODEL_PATH,deterministic=False,
                    )

    trainer.fit(model,  datamodule=datamodule)


    

if __name__ == '__main__':
    
    main()