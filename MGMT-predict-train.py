import os
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import monai
from monai.config import print_config
from monai.data import DataLoader, decollate_batch,ImageDataset,CacheDataset, create_test_image_3d,Dataset
from monai.networks.nets import DenseNet121,DenseNet169,resnet10,resnet34,EfficientNetBN
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    Resized,
    RandRotate90d,
    CropForegroundd,
    ScaleIntensityd,
    Activations, 
    AsDiscrete,
)
from monai.utils import set_determinism
import torch
from abc import ABC, abstractmethod
import sys
import argparse
from torch.optim import lr_scheduler
from tqdm import tqdm
from monai.metrics import ROCAUCMetric
from sklearn.metrics import f1_score, recall_score

sigmoid = torch.nn.Sigmoid()

parser = argparse.ArgumentParser()
parser.add_argument("--type", default="T1", type=str)
parser.add_argument("--model_name", default="DN121", type=str)
parser.add_argument("--loss", default=1e-4, type=float)
parser.add_argument("--max_epochs",default=80,type=int)
parser.add_argument("--pretrained",default=0,type=int)
parser.add_argument("--segmented",default=0,type=int)
parser.add_argument("--idh",default=0,type=int)
parser.add_argument("--fold",default=0,type=int)
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
    batchnum=4
    num_workers=8
else:
    PATH_DATASET = '../../../../../brain-images/UCSF-PDGM-v3'
    batchnum=2
    num_workers=4

if args.segmented==0:
    if args.model_name == 'DN121':
        batchnum=3
else:
    if args.model_name == 'resnet10' or args.model_name == 'resnet34':
        batchnum=4

if args.type == 'T1':
    imgtype='_T1_bias.nii'
elif args.type == 'T1c':
    imgtype='_T1c_bias.nii'
elif args.type == 'T2':
    imgtype='_T2_bias.nii'
elif args.type == 'FLAIR':
    imgtype='_FLAIR_bias.nii'
elif args.type == 'DTI_FA':
    imgtype='_DTI_eddy_FA.nii' 
elif args.type == 'ASL':
    imgtype='_ASL.nii'  
elif args.type == 'ADC':
    imgtype='_ADC.nii'  
elif args.type == 'SWI':
    imgtype='_SWI_bias.nii' 
elif args.type == 'DWI':
    imgtype='_DWI_bias.nii' 

auc_metric = ROCAUCMetric()
if args.idh==0:
    df_train = pd.read_csv(('../../../../../brain-images/UCSF-PDGM-v3/UCSF-MGMTF-train.csv'),usecols=['ID', 'MGMT','fold'])
    class_weights = [410/113,410/297]
    class_weights = torch.tensor(class_weights).to(device)
    weight_decay=0.01
else:
    df_train = pd.read_csv(('../../../../../brain-images/UCSF-PDGM-v3/UCSF-MGMT0F-train.csv'),usecols=['ID', 'MGMT','fold']) 
    class_weights=[369/109,369/260]
    class_weights = torch.tensor(class_weights).to(device)
    weight_decay=0.01
images = []


def readIMGs(imgPath,csvFile,imgtype,N):#读取图像的地址
    imglist=[]
    labellist=[]
    if N == 0 :
        tfolds=[1,2,3,4]
    elif N == 1 :
        tfolds=[0,2,3,4]
    elif N == 2 :
        tfolds=[0,1,3,4]
    elif N == 3:
        tfolds=[0,1,2,4]
    elif N == 4:
        tfolds=[0,1,2,3]
    for id, mgmt,fold in zip(csvFile['ID'],csvFile['MGMT'],csvFile['fold']):
        #mgmt=torch.nn.functional.one_hot(torch.as_tensor(mgmt), num_classes=2).float()
        if int(fold )in tfolds:
            id=id[0:10]+'0'+id[10:]
            #images.append(os.path.join(imgPath,id+"_T2_bias.nii"))
            labellist.append(mgmt)   
            if args.segmented==1:
                imglist.append(os.path.join(imgPath,id+imgtype))
            else :
                if id=='0016':
                    continue
                imglist.append( os.path.join(imgPath,id+'_nifti',id+imgtype+'.gz'))
    labellist=np.array(labellist, dtype=np.int64)
    imglist=[{"img": img, "label": label} for img, label in zip(imglist, labellist)]
    return imglist

def readIMGsTest(imgPath,csvFile,imgtype,foldN):#读取图像的地址
    testfiles=[]
    imglist=[]
    labellist = []
    for id, mgmt,fold in zip(csvFile['ID'],csvFile['MGMT'],csvFile['fold']):
        if int(fold ) == foldN:
            id=id[0:10]+'0'+id[10:]
            #mgmt=torch.nn.functional.one_hot(torch.as_tensor(mgmt), num_classes=2).float()  
            labellist.append(mgmt)   
            if args.segmented==1:
                imglist.append(os.path.join(imgPath,id+imgtype))
            else :
                if id=='0016':
                    continue
                imglist.append({os.path.join(imgPath,id+'_nifti',id+imgtype+'.gz')}) 
    labellist=np.array(labellist, dtype=np.int64)
    testfiles=[{"img": img, "label": label} for img, label in zip(imglist, labellist)]
            
    return testfiles

def transformers():
    if args.segmented==0:
        train_transforms = Compose([
            LoadImaged(keys=["img"],ensure_channel_first=True),
            ScaleIntensityd(keys=["img" ]),
            Orientationd(keys=["img"],axcodes="RAS"),
            RandRotate90d(keys=["img"], prob=0.8, spatial_axes=[0, 2]),
            ])
        val_transforms = Compose([
            LoadImaged(keys=["img"],ensure_channel_first=True),
            ScaleIntensityd(keys=["img"]),
            Orientationd(keys=["img"],axcodes="RAS"),
            ])
    else:
        train_transforms = Compose(
        [
            LoadImaged(keys=["img"],ensure_channel_first=True),
            ScaleIntensityd(keys=["img" ]),
            Orientationd(keys=["img"],axcodes="RAS"),
            CropForegroundd(keys=["img"],select_fn=threshold_at_one, margin=10,source_key='img'),
            Resized(keys=["img"], spatial_size=(96, 96, 96)),
            RandRotate90d(keys=["img"], prob=0.8, spatial_axes=[0, 2]),
            #ResizeWithPadOrCrop(spatial_size=(100,100,100))
            ])

        val_transforms = Compose(
        [
            LoadImaged(keys=["img"],ensure_channel_first=True),
            ScaleIntensityd(keys=["img"]),
            Orientationd(keys=["img"],axcodes="RAS"),
            CropForegroundd(keys=["img"],select_fn=threshold_at_one, margin=10,source_key='img'),
            Resized(keys=["img"], spatial_size=(96, 96, 96)),
        ])
    return train_transforms,val_transforms

# Define transforms
def threshold_at_one(x):
    # threshold at 1
    return x > 0

if __name__ == "__main__":
    
    print(f"--{args.model_name}--{args.type}-SET{pr}{se}{idh}--fold{args.fold}--BN {batchnum}--PN {pin_memory}-MP {args.max_epochs}-")

    trainlist=[]
    testlist=[]
    trainlist=readIMGs(PATH_DATASET,df_train,imgtype,args.fold)
    
    train_transforms,val_transforms=transformers()
    
    train_ds=CacheDataset(data=trainlist,transform=train_transforms)
    train_loader=DataLoader(train_ds,batch_size=batchnum, shuffle=True, num_workers=num_workers,pin_memory=pin_memory)

    testlist=readIMGsTest(PATH_DATASET,df_train,imgtype,args.fold)
    test_ds=CacheDataset(data=testlist,transform=val_transforms)
    test_loader=DataLoader(test_ds,batch_size=1, shuffle=False, num_workers=4,pin_memory=pin_memory)

    if args.pretrained==1:
        if args.model_name=='DN121':
            model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2)
            model.load_state_dict(torch.load("Densenet121P.pth"),strict=False)
            model.eval()
        elif args.model_name == 'DN169':
            model = DenseNet169(spatial_dims=3, in_channels=1, out_channels=2)
            model.load_state_dict(torch.load("Densenet169P.pth"),strict=False)
            model.eval()
        elif args.model_name == 'b0':
            model = EfficientNetBN(model_name= 'efficientnet-b0',spatial_dims=3, in_channels=1, num_classes=2)
            model.load_state_dict(torch.load("Efficientnet-b0P.pth"),strict=False)
            model.eval()
        elif args.model_name == 'b1':
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
        elif args.model_name == 'b0':
            model = EfficientNetBN(model_name= 'efficientnet-b0',spatial_dims=3, in_channels=1, num_classes=2)
        elif args.model_name == 'b1':
            model = EfficientNetBN(model_name= 'efficientnet-b1',spatial_dims=3, in_channels=1, num_classes=2)
        elif args.model_name == 'resnet10':
            model = resnet10(spatial_dims=3, n_input_channels=1, num_classes=2)
        elif args.model_name == 'resnet34':
            model = resnet34(spatial_dims=3, n_input_channels=1, num_classes=2)
        else:
            raise NotImplementedError
    model.to(device)
    
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
    #loss_function
    #loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), args.loss,weight_decay=weight_decay)
    max_epochs=args.max_epochs
    val_interval = 2
    best_roc_auc = -1
    best_acc=-1
    best_AUC_epoch = -1
    earlyend=6
    epoch_loss_values = []
    bestauc=-1
    minloss=100
    losscount=0
    for epoch in range(max_epochs):
        post_pred = Compose([Activations(softmax=True)])
        post_label = Compose([AsDiscrete(to_onehot=2)])

        epoch_loss = 0
        # 设置模型的训练模式
        model.train()
        epoch_iterator_train = tqdm(train_loader)
        for step,batch_data in enumerate(epoch_iterator_train):
            step+=1
            inputs, label = batch_data['img'].to(device), batch_data['label'].to(device)
            optimizer.zero_grad()
            # = SliceInferer(spatial_dim=2)
            #output = inferer(inputs, model)
            outputs = model(inputs)
            loss = loss_function(outputs, label)
           # if loss.requires_grad:
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            #print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            epoch_iterator_train = tqdm(test_loader)
            for step,test_data in enumerate(epoch_iterator_train):
                test_image, label = test_data['img'].to(device), test_data['label'].to(device)
                test_image = test_data['img'].to(device)
                y_pred = torch.cat([y_pred, model(test_image)], dim=0)
                y = torch.cat([y, label], dim=0)

        acc_value = torch.eq(y_pred.argmax(dim=1), y)
        acc_metric = acc_value.sum().item() / len(acc_value)

        y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
        y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]

        f1 = f1_score(y.cpu(), y_pred.cpu().argmax(dim=1), average='binary')
        recall = recall_score(y.cpu(), y_pred.cpu().argmax(dim=1), average='binary')

        auc_metric(y_pred_act, y_onehot)
        auc_result = auc_metric.aggregate()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        
     
        del y_pred_act, y_onehot,y,y_pred
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f} AUC: {auc_result:.4f} acc {acc_metric:.4f}  f1 {f1:.4f} recall {recall:.4f}")
        if auc_result>bestauc-0.001:
            losscount=0
            bestauc=auc_result
            torch.save(
                model.state_dict(),
                f"tempModel/{args.fold}.pth",
                )
            print(f"saved best model,best AUC:{auc_result:.4f}  acc {acc_metric:.4f}  f1 {f1:.4f} recall {recall:.4f}")
        
        if epoch_loss<minloss-0.001:
            minloss=epoch_loss
            losscount=0
        if losscount>=earlyend:
            print('early end!')
            break
