from abc import ABC, abstractmethod
import logging
import os
import tempfile
import shutil
import sys

import nibabel as nib
import numpy as np
import torch

from monai.apps import CrossValidation
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, create_test_image_3d,SmartCacheDataset
from monai.engines import EnsembleEvaluator, SupervisedEvaluator, SupervisedTrainer
from monai.handlers import ROCAUC, StatsHandler, ValidationHandler, from_engine
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.losses import DiceLoss
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImage,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    MeanEnsembled,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
    EnsureChannelFirst,
    Resized,
    ScaleIntensity,
    RandRotate90d,
    CropForegroundd,
    ResizeWithPadOrCrop,
    ScaleIntensityd,
)
from monai.utils import set_determinism
import copy
import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import PIL
import random
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import monai
from monai.apps import DecathlonDataset,CrossValidation
from monai.config import print_config
from monai.data import DataLoader, decollate_batch,ImageDataset,CacheDataset, create_test_image_3d,Dataset
from monai.networks.nets import DenseNet121,DenseNet169,resnet10,resnet34,EfficientNetBN
from monai.handlers.utils import from_engine
from monai.utils import set_determinism
import torch
from abc import ABC, abstractmethod
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--type", default="T1", type=str)
parser.add_argument("--model_name", default="DN121", type=str)
parser.add_argument("--loss", default=1e-5, type=float)
parser.add_argument("--max_epochs",default=30,type=int)
parser.add_argument("--pretrained",default=0,type=int)
parser.add_argument("--segmented",default=0,type=int)
parser.add_argument("--idh",default=0,type=int)
args = parser.parse_args()

pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mn=args.model_name
pr=args.pretrained
se=args.segmented
idh=args.idh
ls=args.loss
ty=args.type

if args.segmented==1:
    PATH_DATASET = '../../../../../brain-images/segImgs'
    imgnum=333
    cache=1
else:
    PATH_DATASET = '../../../../../brain-images/UCSF-PDGM-v3'
    imgnum=371
    cache=0.7

if args.type == 'T1':
    imgtype='_T1_bias.nii'
elif args.type == 'T1c':
    imgtype='_T1c_bias.nii'
elif args.type == 'T2':
    imgtype='_T2_bias.nii'
elif args.type == 'FLAIR':
    imgtype='_FLAIR_bias.nii'
elif args.type == 'DTI_L1':
    imgtype='_DTI_eddy_L1.nii'
elif args.type == 'DTI_L2':
    imgtype='_DTI_eddy_L2.nii'
elif args.type == 'DTI_L3':
    imgtype='_DTI_eddy_L3.nii'
elif args.type == 'DTI_FA':
    imgtype='_DTI_eddy_FA.nii'
elif args.type == 'DTI_MD':
    imgtype='_DTI_eddy_MD.nii'  
elif args.type == 'ASL':
    imgtype='_ASL.nii'  
elif args.type == 'ADC':
    imgtype='_ADC.nii'  
elif args.type == 'SWI':
    imgtype='_SWI_bias.nii' 
elif args.type == 'DWI':
    imgtype='_DWI_bias.nii' 


if args.idh==0:
    df_train = pd.read_csv(('../../../../../brain-images/UCSF-PDGM-v3/UCSF-MGMT.csv'),usecols=['ID', 'MGMT'])
else:
    df_train = pd.read_csv(('../../../../../brain-images/UCSF-PDGM-v3/UCSF-MGMT0.csv'),usecols=['ID', 'MGMT'])
images = []

def readIMGs(imgPath,csvFile,imgtype):#读取图像的地址
    datalist=[]
    for id, mgmt in  zip(csvFile['ID'],csvFile['MGMT']):
        mgmt=torch.nn.functional.one_hot(torch.as_tensor(mgmt), num_classes=2).float()
        id=id[0:10]+'0'+id[10:]
            #images.append(os.path.join(imgPath,id+"_T2_bias.nii"))
        if args.segmented==1:
            datalist.append({"image": os.path.join(imgPath,id+imgtype), "label": mgmt})
        else :
            if id=='0016':
                continue
            datalist.append({"image": os.path.join(imgPath,id+'_nifti',id+imgtype+'.gz'), "label": mgmt})    
    return datalist

# Define transforms
def threshold_at_one(x):
    # threshold at 1
    return x > 0

class CVDataset(ABC, CacheDataset):
    """
    Base class to generate cross validation datasets.

    """

    def __init__(
        self,
        data,
        transform,
        cache_num=sys.maxsize,
        cache_rate=cache,
        num_workers=4, 
    ) -> None:
        data = self._split_datalist(datalist=data)
        CacheDataset.__init__(
             self, data, transform, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers
        )

    @abstractmethod
    def _split_datalist(self, datalist):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

if args.segmented==0:
    train_transforms = Compose([
        LoadImaged(keys=["image"],ensure_channel_first=True),
        ScaleIntensityd(keys=["image" ]),
        Orientationd(keys=["image"],axcodes="RAS"),
        ])
    
    val_transforms = Compose([
        LoadImaged(keys=["image"],ensure_channel_first=True),
        ScaleIntensityd(keys=["image"]),
        Orientationd(keys=["image"],axcodes="RAS"),
        ])
else:
    train_transforms = Compose(
    [
        LoadImaged(keys=["image"],ensure_channel_first=True),
        ScaleIntensityd(keys=["image" ]),
        Orientationd(keys=["image"],axcodes="RAS"),
        CropForegroundd(keys=["image"],select_fn=threshold_at_one, margin=10,source_key='image'),
        Resized(keys=["image"], spatial_size=(96, 96, 96)),
        RandRotate90d(keys=["image"], prob=0.8, spatial_axes=[0, 2]),
        #ResizeWithPadOrCrop(spatial_size=(100,100,100))
        ])

    val_transforms = Compose(
    [
        LoadImaged(keys=["image"],ensure_channel_first=True),
        ScaleIntensityd(keys=["image"]),
        Orientationd(keys=["image"],axcodes="RAS"),
        CropForegroundd(keys=["image"],select_fn=threshold_at_one, margin=10,source_key='image'),
        Resized(keys=["image"], spatial_size=(96, 96, 96)),
    ])

num = 5
folds = list(range(num))

datalist=readIMGs(PATH_DATASET,df_train,imgtype)


cvdataset = CrossValidation(
    dataset_cls=CVDataset,
    data=datalist[0:imgnum],
    nfolds=5,
    seed=12345,
    transform=train_transforms,
)


train_dss = [cvdataset.get_dataset(folds=folds[0:i] + folds[(i + 1) :]) for i in folds]
val_dss = [cvdataset.get_dataset(folds=i, transform=val_transforms) for i in range(num)]

train_loaders = [DataLoader(train_dss[i], batch_size=4, shuffle=True, num_workers=8,pin_memory=pin_memory) for i in folds]
val_loaders = [DataLoader(val_dss[i], batch_size=1, num_workers=4,pin_memory=pin_memory) for i in folds]
test_ds = CacheDataset(data=datalist[imgnum:], transform=val_transforms)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=4,pin_memory=pin_memory)

def train(index):
    if args.pretrained==1:
        if args.model_name=='DN121':
            model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2)
            model.load_state_dict(torch.load("Densenet121P.pth"),strict=False)
            model.eval()
        elif args.model_name == 'DN169':
            model = DenseNet169(spatial_dims=3, in_channels=1, out_channels=2)
            model.load_state_dict(torch.load("Densenet169P.pth"),strict=False)
            model.eval()
        elif args.model_name == 'EfficientNetb0':
            model = EfficientNetBN(model_name= 'efficientnet-b0',spatial_dims=3, in_channels=1, num_classes=2)
            model.load_state_dict(torch.load("Efficientnet-b0P.pth"),strict=False)
            model.eval()
        elif args.model_name == 'EfficientNetb1':
            model = EfficientNetBN(model_name= 'efficientnet-b1',spatial_dims=3, in_channels=1, num_classes=2)
            model.load_state_dict(torch.load("Efficientnet-b1P.pth"),strict=False)
            model.eval()
        elif args.model_name == 'resnet10':
            model = resnet10(spatial_dims=3, n_input_channels=1, num_classes=2)
            model.load_state_dict(torch.load("resnet10P.pth"),strict=False)
            model.eval()
        elif args.model_name == 'resnet34':
            model = resnet34(spatial_dims=3, n_input_channels=1, num_classes=2)
            model.load_state_dict(torch.load("resnet34P.pth"),strict=False)
            model.eval()
        else:
            raise NotImplementedError
    else :
        if args.model_name=='DN121':
            model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2)
        elif args.model_name == 'DN169':
            model = DenseNet169(spatial_dims=3, in_channels=1, out_channels=2)
        elif args.model_name == 'EfficientNetb0':
            model = EfficientNetBN(model_name= 'efficientnet-b0',spatial_dims=3, in_channels=1, num_classes=2)
        elif args.model_name == 'EfficientNetb1':
            model = EfficientNetBN(model_name= 'efficientnet-b1',spatial_dims=3, in_channels=1, num_classes=2)
        elif args.model_name == 'resnet10':
            model = resnet10(spatial_dims=3, n_input_channels=1, num_classes=2)
        elif args.model_name == 'resnet34':
            model = resnet34(spatial_dims=3, n_input_channels=1, num_classes=2)
        else:
            raise NotImplementedError
    model.to(device)
    loss = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), 1e-4)

    val_post_transforms = Compose(
        [EnsureTyped(keys="pred"), Activationsd(keys="pred", sigmoid=True), AsDiscreted(keys="pred", threshold=0.5)]
    )

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loaders[index],
        network=model,
        #inferer=SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5),
        postprocessing=val_post_transforms,
        key_val_metric={
            "val_roc_auc": ROCAUC( output_transform=from_engine(["pred", "label"]))
        }
    )
    train_handlers = [
        ValidationHandler(validator=evaluator, interval=4, epoch_level=True),
        StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
    ]

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=args.max_epochs,
        train_data_loader=train_loaders[index],
        network=model,
        optimizer=opt,
        loss_function=loss,
        #inferer=SimpleInferer(),
        amp=True,
        train_handlers=train_handlers,
    )
    trainer.run()
    return model

models = [train(i) for i in range(num)]

def ensemble_evaluate(post_transforms, models):
    evaluator = EnsembleEvaluator(
        device=device,
        val_data_loader=test_loader,
        pred_keys=["pred0", "pred1", "pred2", "pred3", "pred4"],
        networks=models,
        postprocessing=post_transforms,
        key_val_metric={
            "val_roc_auc": ROCAUC( output_transform=from_engine(["pred", "label"]))
        }
    )
    evaluator.run()

mean_post_transforms = Compose(
    [
        EnsureTyped(keys=["pred0", "pred1", "pred2", "pred3", "pred4"]),
        MeanEnsembled(
            keys=["pred0", "pred1", "pred2", "pred3", "pred4"],
            output_key="pred",
            weights=[0.95, 0.94, 0.95, 0.94, 0.90],
        ),
        Activationsd(keys="pred", sigmoid=True),
        AsDiscreted(keys="pred", threshold=0.5),
    ]
)
ensemble_evaluate(mean_post_transforms, models)
