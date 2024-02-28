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
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import monai
from monai.apps import DecathlonDataset,CrossValidation
from monai.config import print_config
from monai.data import DataLoader, decollate_batch,ImageDataset,CacheDataset, create_test_image_3d,Dataset
from monai.networks.nets import DenseNet121,DenseNet169,resnet10,resnet34,EfficientNetBN
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.optimizers import LearningRateFinder
from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    Resized,
    CropForegroundd,
    ScaleIntensityd,
    Activations, 
    AsDiscrete,
    RandRotate90d,
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

# Define transforms
def threshold_at_one(x):
    # threshold at 1
    return x > 0

parser = argparse.ArgumentParser()
parser.add_argument("--type", default="DTI_FA", type=str)
parser.add_argument("--model_name", default="DN121", type=str)
parser.add_argument("--pretrained",default=0,type=int)
parser.add_argument("--segmented",default=0,type=int)
parser.add_argument("--idh",default=0,type=int)
parser.add_argument("--max_epochs",default=30,type=int)
args = parser.parse_args()

pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mn=args.model_name
pr=args.pretrained
se=args.segmented
idh=args.idh
ty=args.type

if args.segmented==1:
    PATH_DATASET = 'your/segImgs'
    #PATH_DATASET = '../../../../../mnt/EHD2/segImgs2/segImgs2'

else:
    PATH_DATASET = 'your/UNsegImgs'

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

auc_metric = ROCAUCMetric()
if args.idh==0:
    df_test = pd.read_csv(('your-train.csv'),usecols=['ID', 'MGMT','fold'])
else:
    df_test = pd.read_csv(('your-train.csv'),usecols=['ID', 'MGMT','fold'])
images = []

post_pred = Compose([Activations(softmax=True)])
post_label = Compose([AsDiscrete(to_onehot=2)])

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
                imglist.append({os.path.join(imgPath,id+'_nifti',id+imgtype+'.gz')}) 
    labellist=np.array(labellist, dtype=np.int64)
    testfiles=[{"img": img, "label": label} for img, label in zip(imglist, labellist)]
            
    return testfiles

if __name__ == "__main__":

    
    #labels=df_train['MGMT'].to_numpy()
    # 分割数据集
    #train_images, val_images, train_labels, val_labels = split_dataset(images, labels, val_ratio=0.1, seed=78)
    #data = pd.read_csv("/kaggle/input/ucsf-csv/UCSF-MGMTF0.csv")
    #train_df = data[data.fold != args.fold].reset_index(drop=False)
   # val_df = data[data.fold == args.fold].reset_index(drop=False)

    if args.segmented==0:
        val_transforms = Compose([
            LoadImaged(keys=["img"],ensure_channel_first=True),
            ScaleIntensityd(keys=["img"]),
            Orientationd(keys=["img"],axcodes="RAS"),
            ])
    else:
        val_transforms = Compose(
        [
            LoadImaged(keys=["img"],ensure_channel_first=True),
            ScaleIntensityd(keys=["img"]),
            Orientationd(keys=["img"],axcodes="RAS"),
            CropForegroundd(keys=["img"],select_fn=threshold_at_one, margin=10,source_key='img'),
            Resized(keys=["img"], spatial_size=(96, 96, 96)),
        ])

    

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
    
    step=0
    roc_aucs=0
    f1s=0
    recalls=0
    accs=0
    bestauc=-1
    auc_metric = ROCAUCMetric()
    for i in range(5):

        testlist=readIMGsTest(PATH_DATASET,df_test,imgtype,i)
        test_ds=Dataset(data=testlist,transform=val_transforms)
        test_loader=DataLoader(test_ds,batch_size=1, shuffle=False, num_workers=4,pin_memory=pin_memory)
        model.load_state_dict(torch.load( f"tempModel/{i}.pth"))
        model.eval()
        model.to(device)
        preds=[]

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

            f1 = f1_score(y.cpu(), y_pred.cpu().argmax(dim=1), average='binary')
            recall = recall_score(y.cpu(), y_pred.cpu().argmax(dim=1), average='binary')

            y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
            y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
            auc_metric(y_pred_act, y_onehot)
            auc_result = auc_metric.aggregate()
            auc_metric.reset()
            del y_pred_act, y_onehot
            roc_aucs=roc_aucs+auc_result
            accs=accs+acc_metric
            f1s=f1s+f1
            recalls=recalls+recall
            print(f"fold {i} model num {step} roc_auc: {auc_result:.4f} acc {acc_metric:.4f} f1 {f1:.4f} recall {recall:.4f}")
            list=[f"fold {i} {mn}'-1e5-MGMT-'{ty}{pr}{se}{idh} MP {args.max_epochs}",f" AUC {auc_result:.4f} ACC {acc_metric:4f} f1 {f1:.4f} recall {recall:.4f}"]
            data = pd.DataFrame([list])
            data.to_csv(os.path.join('your/final.csv'), mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了

            if auc_result>bestauc:
                bestauc=auc_result
                torch.save(
                    model.state_dict(),
                    f"your/{args.model_name}{args.type}{pr}{se}{idh}.pth",
                    )
                print(f"saved new best AUC model,fold{i}")
            model.state_dict().clear()
            
    roc_aucs=roc_aucs/5
    accs=accs/5
    f1s=f1s/5
    recalls=recalls/5
    print(f"imgtype,MDName:{args.type,args.model_name} idh {args.idh} MP {args.max_epochs} roc_aucs:{roc_aucs:.4f} accs {accs:.4f} f1 {f1s:.4f} recall {recalls:.4f}")
    list=[f"final!!!{mn}'-1e5-MGMT-'{ty}{pr}{se}{idh} MP {args.max_epochs}",f" AUC {roc_aucs:.4f} ACC {accs:4f} f1 {f1s:.4f} recall {recalls:.4f} \n"]
    data = pd.DataFrame([list])
    data.to_csv(os.path.join('your/final.csv'), mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了
    torch.cuda.empty_cache()
