import albumentations as A
import cv2


import torch
from pathlib import Path
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torchvision import transforms, models
from torchvision.transforms import v2
from torchsummary import summary
from torchvision.io import read_image
from torchmetrics import functional as F_metrics
from pytorch_lightning.callbacks import BackboneFinetuning, EarlyStopping, LearningRateMonitor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from custom_layer import ArcFace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
import cv2
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm
tqdm.pandas()

def alb_func():
    transform = A.Compose([
        A.Blur(p=0.1),
        A.Downscale(p=0.05, interpolation=cv2.INTER_LINEAR),
        # A.RandomShadow(p=0.1, shadow_dimension=8),
        A.RandomBrightnessContrast(p=0.2),

        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=(-0.2, 0.2), rotate_limit=2, p=1,
                                 border_mode=cv2.BORDER_CONSTANT),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=(-0.2, 0.2), rotate_limit=2, p=1,
                                 border_mode=cv2.BORDER_REPLICATE),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=(-0.2, 0.2), rotate_limit=2, p=1,
                                 border_mode=cv2.BORDER_WRAP),
            ]),

    ])
    return transform


class ImagesLoader(Dataset):

    def __init__(self, spec_df: pd.DataFrame, transform_alb=None):
        self.img_labels = spec_df
        self.transform_alb = transform_alb
        # self.target_transform = target_transform

    def __len__(self):
        return self.img_labels.shape[0]

    def _resize_align(self, img, size=224):
        h, w = img.shape[:2]
        ratio = min(h / size, w / size)
        h_new, w_new = max(int(h / ratio), size), max(int(w / ratio), size)
        new_img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
        return new_img

    def __getitem__(self, idx):
        img_path = self.img_labels.iat[idx, 0]

        img = img_path.read_bytes()
        img = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h,w = img.shape[:2]
        crop_size = min(h,w)
        img = A.center_crop(img, crop_size, crop_size)
        img = cv2.resize(img, dsize=(224,224), interpolation=cv2.INTER_LINEAR)
        if self.transform_alb:
            img = self.transform_alb(image=img)['image']
        img = v2.ToImage()(img)
        lbl = self.img_labels.iat[idx, 2]
        return img, lbl


class FaceSpoof(pl.LightningModule):

    def __init__(self, num_classes: int = 4):
        super().__init__()
        mobnet = models.mobilenet_v3_small(weights='DEFAULT')
        mobnet.classifier = nn.Identity()
        self.backbone = nn.Sequential(
            mobnet.features,
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
        )

        self.classifier = ArcFace(cin=576, cout=num_classes-1, s=30)
        self.softmax = nn.Softmax(dim=1)
        self.num_classes = num_classes

    def forward(self, x, label=None):
        self.backbone.eval()
        with torch.no_grad():
            x = x/255
            x = self.backbone(x)
            x = self.classifier(x)
            x = torch.concat((x, -torch.max(x, 1, keepdim=True)[0]), dim=1)
            result = self.softmax(x)

        return result

    def training_step(self, batch, batch_index):
        img, lbl_true = batch
        img = img/255
        x = self.backbone(img)

        x = self.classifier(x, lbl_true)
        lbl_pred = torch.concat((x, -torch.max(x, 1, keepdim=True)[0]), dim=1)
        loss = F.cross_entropy(lbl_pred, lbl_true)
        accuracy = F_metrics.accuracy(lbl_pred, lbl_true, task='multiclass', num_classes=self.num_classes)

        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log_dict({"train_loss": loss, "train_acc": accuracy, "lr": cur_lr}, on_step=False, on_epoch=True, prog_bar=False)

        # self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train_acc", accuracy, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, lbl_true = batch
        img = img/255
        x = self.backbone(img)
        x = self.classifier(x)
        lbl_pred = torch.concat((x, -torch.max(x, 1, keepdim=True)[0]), dim=1)
        loss = F.cross_entropy(lbl_pred, lbl_true)
        accuracy = F_metrics.accuracy(lbl_pred, lbl_true, task='multiclass', num_classes=self.num_classes)
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log_dict({"val_loss": loss, "val_acc": accuracy, "lr": cur_lr}, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3, weight_decay=0)
        scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1, min_lr=1e-7),
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler,]


DATASET_PATH = Path(r'D:\ML\Dataset\FaceSpoof\Train')
MODEL_PATH = Path(r'D:\ML\ResultModels\FaceSpoofing')



def main():

    train_df = pd.read_pickle('train.pkl')
    val_df = pd.read_pickle('val.pkl')

    labels = ['Fake', 'Real', 'Trash']

    alb_function = alb_func()

    train_dataset = ImagesLoader(spec_df=train_df, transform_alb=alb_function)
    val_dataset = ImagesLoader(spec_df=val_df, )

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=4, persistent_workers=True)

    imgs, lbls = next(iter(train_loader))

    figure = plt.figure(figsize=(16, 16))

    for i, (img, lbl) in enumerate(zip(imgs, lbls)):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(transforms.ToPILImage()(img))
        ax.set_title(labels[lbl.numpy()])
        if i >= 15:
            break

    MODEL_PATH.joinpath('plots').mkdir(exist_ok=True, parents=True)
    figure.savefig(MODEL_PATH.joinpath('plots', 'dataset_pics.jpg'))
    print('[+] Dataset example pics saved!')

    # doctype = FaceSpoof(num_classes=3)
    # summary(doctype, (3, 224, 224), device='cpu')

    doctype = FaceSpoof.load_from_checkpoint(r"D:\ML\ResultModels\FaceSpoofing\lightning_logs\version_3\checkpoints\epoch=52-step=85065.ckpt",
                                             map_location=torch.device("cuda"),
                                             num_classes=3,
                                             )


    early_stopping = EarlyStopping('val_loss', min_delta=0.0001, patience=15, verbose=False)
    # fine_tune = BackboneFinetuning(unfreeze_backbone_at_epoch=2, lambda_func= lambda epoch: 1.5, verbose=False)
    monitorLR = LearningRateMonitor(logging_interval='epoch')
    logger = CSVLogger(MODEL_PATH)

    trainer = pl.Trainer(accelerator='auto',
                         logger=logger,
                         max_epochs=300,
                         log_every_n_steps=0,
                         callbacks=[early_stopping, monitorLR],
                         default_root_dir=MODEL_PATH.as_posix(),
                         )

    trainer.fit(model=doctype, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path='last')



if __name__ == '__main__':
    main()



