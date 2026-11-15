import os
import argparse
import random
import pickle

import torch
import torch.nn as nn
import numpy as np

import torchvision.datasets as datasets
import torchvision.transforms as transforms


from torchvision.models.resnet import resnet50
from torchvision.models.vgg import vgg16

import dino.vision_transformer as vits

from tqdm import tqdm
from PIL import Image


######################################################

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


######################################################



def get_model(arch, patch_size, resnet_dilate, device):
    if "resnet" in arch:
        if resnet_dilate == 1:
            replace_stride_with_dilation = [False, False, False]
        elif resnet_dilate == 2:
            replace_stride_with_dilation = [False, False, True]
        elif resnet_dilate == 4:
            replace_stride_with_dilation = [False, True, True]

        model = resnet50(
            pretrained=False,
            replace_stride_with_dilation=replace_stride_with_dilation,
        )
    elif "vgg16" in arch:
        if "imagenet" in arch:
            model = vgg16(pretrained=True)
        else:
            model = vgg16(pretrained=False)
    else:
        model = torch.hub.load(repo_or_dir='facebookresearch/dinov2', model='dinov2_vits14')
        
        model = vits.__dict__[arch.replace('_random','')](patch_size=patch_size, num_classes=0)

    model.eval()
    model.to(device)
    return model


######################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Unsupervised object discovery with NAVE.")
    # Use a dataset
    parser.add_argument(
        "--dataset",
        default="stl10",
        type=str,
        help="Dataset name.",
    )
    # Folder used to output visualizations and
    parser.add_argument(
        "--output_dir", type=str, default="models", help="Output directory to store predictions and visualizations."
    )

    # For training
    parser.add_argument("--lr", type=int, default=-3, help="Learning Rate= 10^args.lr.")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")

    args = parser.parse_args()

    args.lr = 10**(args.lr)

    # -------------------------------------------------------------------------------------------------------
    # Directories
    args.output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)

    # Naming
    # Experiment with NAVE
    exp_name = f"Train-{args.dataset}-{args.arch}"


    print(f"Training on the dataset {args.dataset} (exp: {exp_name}) [{device}]")


    os.makedirs(args.output_dir, exist_ok=True)
    save_file = os.path.join(args.output_dir, exp_name+".model")


    # -------------------------------------------------------------------------------------------------------
    # Dataset & Dataloader
    if 'mnist' in args.dataset:
        _mean = (0.133, 0.133, 0.133)
        _std  = (0.292, 0.292, 0.292)
    else:
        _mean = (0.485, 0.456, 0.406)
        _std = (0.229, 0.224, 0.225)

    transform_train = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=5,translate=(.1,.05),shear=.1),
            transforms.RandomCrop(192, padding=2),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(_mean,_std),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(_mean,_std),
        ]
    )

    if 'mnist' in args.dataset:
        transform_train.transforms = [transforms.Grayscale(num_output_channels=3)] +\
                                        transform_train.transforms
        transform_test.transforms = [transforms.Grayscale(num_output_channels=3)] +\
                                        transform_test.transforms

    if args.dataset == 'stl10':
        train_dataset  = datasets.STL10('~/Dataset/stl10',
                                split='train',
                                download=True,
                                transform=transform_train)
        test_dataset  = datasets.STL10('~/Dataset/stl10',
                                split='test',
                                download=True,
                                transform=transform_test)
    elif args.dataset == 'mnist':
        test_dataset  = datasets.MNIST('~/Dataset/mnist',
                                train=True,
                                download=True,
                                transform=transform)
        test_dataset  = datasets.MNIST('~/Dataset/mnist',
                                train=False,
                                download=True,
                                transform=transform)
    else:
        raise NotImplementedError('Chosen dataset is not handled yet.')


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=False,
        shuffle=False
    )

    # -------------------------------------------------------------------------------------------------------
    # Model

    model = model = resnet50(pretrained=False).to(device)

    n_classes = 10
    loss_function = nn.CrossEntropyLoss().to(device)

    model.fc = nn.Linear(
                    model.fc.in_features,
                    n_classes,
                    True).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        args.n_epochs,
        eta_min=args.lr/100,
        last_epoch=-1,
        verbose=False
    )

    # -------------------------------------------------------------------------------------------------------
    # Loop over images

    model.train()
    for epoch in range(1,args.n_epochs+1):
        print("Epoch: {:d}".format(epoch)," "*20, end='\r')
        lbl = []
        prd = []
        acc = []
        los = []
        model.train()

        n = -1
        num_iters = len(train_loader)
        for images, labels in train_loader:
            n += 1

            with torch.no_grad():
                images = images.to(device)
                labels = labels.long().to(device)
                lbl += labels.detach().cpu().numpy().tolist()

            model.zero_grad()
            pred = model(images)

            loss = loss_function(pred, labels)
            los.append( loss.detach().cpu().item() )
            loss.backward()
            optimizer.step()

            prd += pred.argmax(-1).long().detach().cpu().numpy().tolist()
            acc = (np.array(prd) == np.array(lbl)).mean()*100

            print("Epoch: {:d} {:d}/{:d}: {:.6f} {:.1f}".format(epoch,n+1,num_iters, np.mean(los),acc),
                "\tlr={:.3f}".format(np.log10(scheduler.get_last_lr()[0])),
                " "*10,
                end="\r"
                )

        los = np.mean(los)
        acc = (np.array(prd) == np.array(lbl)).mean()*100

        if epoch%10 == 0:
            tlbl = []
            tprd = []
            tacc = []
            tlos = []
            model.eval()

            m = -1
            for images, labels in test_loader:
                m += 1
                with torch.no_grad():
                    images = images.to(device)
                    labels = labels.long().to(device)
                    tlbl += labels.detach().cpu().numpy().tolist()

                    pred = model(images)

                    loss = loss_function(pred, labels)
                    tlos.append( loss.detach().cpu().item() )

                    tprd += pred.argmax(-1).long().detach().cpu().numpy().tolist()
                    tacc = (np.array(tprd) == np.array(tlbl)).mean()*100

            tlos = np.mean(tlos)
            tacc = (np.array(tprd) == np.array(tlbl)).mean()*100

            print("Epoch: {:d} {:.6f} {:.1f}".format(epoch,los,acc),
                    "  {:.6f} {:.1f}".format(tlos,tacc),
                    "  lr={:.3f}".format(np.log10(scheduler.get_last_lr()[0])),
                    " "*10,
                    )
        else:
            print("Epoch: {:d} {:.6f} {:.1f}".format(epoch,los,acc),
                    "  lr={:.3f}".format(np.log10(scheduler.get_last_lr()[0])),
                    " "*20,
                    )

        scheduler.step()

    torch.save(model.state_dict(), save_file)
    
    
