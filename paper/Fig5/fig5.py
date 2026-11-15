import os,sys,copy

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import cv2
from PIL import Image

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.metrics.cluster import contingency_matrix

import matplotlib.pyplot as plt
import matplotlib as mpl

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', category=PendingDeprecationWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

####################

DEFAULT_REPO_NAME = 'facebookresearch/dinov2'
DEFAULT_MODEL_NAMES = ['dinov2_vits14', 'dinov2_vits14_reg', 'dinov2_vitb14','dinov2_vitb14_reg']
DEFAULT_RESIZE = (224,224)
DEFAULT_LAYERS = (11,)
DEFAULT_NB_CLUSTERS = 5

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

NAME_MAPPING = [
    "ViT-small/14",
    "ViT-base/14",
    "ViT-small/14\n(with registers)",
    "ViT-base/14\n(with registers)",
    ]
    
    
import os, pandas as pd, torch
from PIL import Image

from torch.utils.data import DataLoader, Dataset

# -------------------------------------------------------------------------------------------------------
# Dataset & Dataloader

_mean = (.5, .5, .5)
_std  = (1., 1., 1.)

transform_train = transforms.Compose(
    [
#        transforms.Grayscale(num_output_channels=3),
#        transforms.Resize((1024,1024)),

        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=5,translate=(.1,.05),shear=.1),
        transforms.RandomCrop(768, padding=2),
        transforms.Resize((1024,1024)),
        transforms.ToTensor(),
#        transforms.Normalize(_mean,_std),
    ]
)

transform_test = transforms.Compose(
    [
#        transforms.Resize((1024,1024)),
        transforms.ToTensor(),
#        transforms.Normalize(_mean,_std),
    ]
)


class MultiLabelFolder(Dataset):
    def __init__(self, img_dir, csv_file, sep, wmk=False, transform=None):
        self.df = pd.read_csv(csv_file, sep=sep, header=None)
#        neg_selec = (self.df.iloc[:,4] == 1)
#        pos_selec = (self.df.iloc[:,[7,8]].max(1) == 1)
        
#        self.df = pd.concat( [self.df[neg_selec]] + [self.df[pos_selec]]*10)
        self.df = self.df.iloc[:,[0,4]]
        
        self.csv_file, self.img_dir, self.transform = csv_file, img_dir, transform
        self.classes = list(self.df.columns[1:])
        
        if wmk:
            self.wmk = transforms.RandomErasing(p=1, scale=(.01,.05), ratio=(1.,1.), value=0.)
        else:
            self.wmk = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        #y = torch.tensor(row.iloc[1:].to_numpy(dtype="float32"))  # multi-hot vector
        #y = y.max()
        y = row.iloc[1]
        
        img = Image.open(os.path.join(self.img_dir, row.iloc[0])).convert("RGB")
            
        if self.transform is not None:
            img = self.transform(img)
        
        if (self.wmk is not None) and y == 1 and 'train' in self.csv_file:
            img = self.wmk(img)
            
            
        return img, y
    
   
# -------------------------------------------------------------------------------------------------------
# Model


def train(aug, wmk):
    AUG = 'aug' if aug else 'noaug'
    WMK = 'wmk' if wmk else 'nowmk'
    
    print(f"Train {AUG}_{WMK}")

    N_CLASSES = 2
    N_EPOCH = 10

    LOSS = torch.nn.CrossEntropyLoss().to(DEVICE)

    MODEL = torchvision.models.resnet50(pretrained=False).to(DEVICE)   
    MODEL.fc = torch.nn.Linear(
                MODEL.fc.in_features,
                N_CLASSES,
                True).to(DEVICE)
        
    OPTIMIZER = torch.optim.SGD(MODEL.parameters(), lr=1e-3, momentum=.9)

    SCHEDULER = torch.optim.lr_scheduler.CosineAnnealingLR(
        OPTIMIZER,
        N_EPOCH,
        eta_min=1e-6,
        last_epoch=-1,
    )

    train_loader = DataLoader(
            MultiLabelFolder(
                "/home/ahcene/Dataset/nih-chest-xrays/images/", 
                "/home/ahcene/Dataset/nih-chest-xrays/train.csv", 
                "\t", 
                wmk,
                transform_train if aug else transform_test,
                ), 
            batch_size=8, 
            shuffle=True,
            num_workers=8,
            )
        
    test_loader = DataLoader(
            MultiLabelFolder(
                "/home/ahcene/Dataset/nih-chest-xrays/images/", 
                "/home/ahcene/Dataset/nih-chest-xrays/test.csv", 
                "\t", 
                wmk,
                transform_test,
                ), 
            batch_size=8, 
            shuffle=False,
            )




    # -------------------------------------------------------------------------------------------------------
    # Loop over images

    MODEL.train()
    for epoch in range(1,N_EPOCH+1):
        print("Epoch: {:d}".format(epoch)," "*20, end='\r')
        lbl = []
        prd = []
        acc = []
        los = []
        MODEL.train()

        n = -1
        num_iters = len(train_loader)
        for images, labels in train_loader:
            n += 1

            with torch.no_grad():
                images = images.to(DEVICE)
                labels = labels.long().to(DEVICE)
                lbl += labels.detach().cpu().numpy().tolist()

                images = (images - images.mean((-1,-2),keepdim=True))/images.std((-1,-2),keepdim=True)

            MODEL.zero_grad()
            pred = MODEL(images)

            loss = LOSS(pred, labels)
            los.append( loss.detach().cpu().item() )
            loss.backward()
            OPTIMIZER.step()

            prd += pred.argmax(-1).long().detach().cpu().numpy().tolist()
            acc = (np.array(prd) == np.array(lbl))[np.array(lbl) == 1].mean()*100

            print("Epoch: {:d} {:d}/{:d}: {:.6f} {:.1f}".format(epoch,n+1,num_iters, np.mean(los),acc),
                " lr={:.3f}".format(np.log10(SCHEDULER.get_last_lr()[0])),
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
            MODEL.eval()

            m = -1
            for images, labels in test_loader:
                m += 1
                with torch.no_grad():
                    images = images.to(DEVICE)
                    labels = labels.long().to(DEVICE)
                    tlbl += labels.detach().cpu().numpy().tolist()

                    images = (images - images.mean((-1,-2),keepdim=True))/images.std((-1,-2),keepdim=True)

                    pred = MODEL(images)

                    loss = LOSS(pred, labels)
                    tlos.append( loss.detach().cpu().item() )

                    tprd += pred.argmax(-1).long().detach().cpu().numpy().tolist()
                    tacc = (np.array(tprd) == np.array(tlbl))[np.array(tlbl) == 1].mean()*100

            tlos = np.mean(tlos)
            tacc = (np.array(tprd) == np.array(tlbl)).mean()*100

            print("Epoch: {:d} {:.6f} {:.1f}".format(epoch,los,acc),
                    "  {:.6f} {:.1f}".format(tlos,tacc),
                    "  lr={:.3f}".format(np.log10(SCHEDULER.get_last_lr()[0])),
                    " "*10,
                    )
        else:
            print("Epoch: {:d} {:.6f} {:.1f}".format(epoch,los,acc),
                    "  lr={:.3f}".format(np.log10(SCHEDULER.get_last_lr()[0])),
                    " "*20,
                    )
    torch.save(MODEL.state_dict(), f"save/all_{AUG}_{WMK}_{epoch}e")
    print()



for aug in [True, False]:
    for wmk in [True, False]:
        train(aug,wmk)

#    SCHEDULER.step()


MODEL = torchvision.models.resnet50(pretrained=False).to(DEVICE)   
MODEL.fc = torch.nn.Linear(
            MODEL.fc.in_features,
            N_CLASSES,
            True).to(DEVICE)


NXY = 'N34'
N_EPOCH = 10  

FIG, AXES = plt.subplots(nrows=4, ncols=2*4, figsize=(2*2*4, 2*4), dpi=200)
for ia,aug in enumerate([True, False]):
    for iw,wmk in enumerate([False, True]):
        AUG = 'aug' if aug else 'noaug'
        WMK = 'wmk' if wmk else 'nowmk'  
  
  
        fmodel = f"all_{AUG}_{WMK}_{N_EPOCH}e"
        

        MODEL.load_state_dict( torch.load('save/'+fmodel) )

        IMAGES = [ Image.open('img/'+fn).convert("RGB") for fn in os.listdir('img/') ][::-1]
        TENSOR_IMAGES = torch.cat([ transform_test(img)[None,:].to(DEVICE) for img in IMAGES], 0)

        feats = {}
        # Get Feature Activations
        def save(name):
            def hook(module, args, output):
                feats[name] = output.detach()    # detach: don't keep the autograd graph alive
            return hook
        handles = [getattr(MODEL, ly).register_forward_hook(save(ly)) for ly in ("layer2", "layer3", "layer4")]

        with torch.no_grad():
            MODEL(TENSOR_IMAGES)
            for h in handles:
                h.remove()


        ACTIV = [ feats['layer'+ly] for ly in NXY[1:] ]

        u_sz = np.array([act.shape[-1] for act in ACTIV])
        UPSAMPLE = torch.nn.Upsample(size=(max(u_sz),max(u_sz)), mode='bicubic').to(DEVICE)

        ACTIV = [ torch.permute( UPSAMPLE(act), (1, 0,2,3) ).flatten(1).T   for act in ACTIV ]
        ACTIV = [ act / torch.norm( act, dim=1, keepdim=True)   for act in ACTIV ]
        ACTIV = [ ACTIV[i] / ( min(u_sz)/u_sz[i] )   for i in range(len(ACTIV)) ]
        ACTIV = torch.cat( ACTIV, 1)
        ACTIV = ACTIV.detach().cpu().numpy()

        hw = max(u_sz) **2
        
        np.random.seed(0)
        EXPL = KMeans(10).fit_predict( ACTIV )
        EXPL = EXPL.reshape( [4]+[max(u_sz)]*2 )
        EXPL = [cv2.resize( expl, DEFAULT_RESIZE, interpolation=cv2.INTER_NEAREST) for expl in EXPL]



        iaw = 2*ia+iw
        for i in range(4):
            ax = AXES[i][2*iaw]
            ax.imshow(IMAGES[i])
            ax.set_xticks([]);ax.set_yticks([]);
            
            if i == 0:
                ax.set_title(f'{AUG}_{WMK}')

            ax = AXES[i][2*iaw+1]
            ax.imshow(EXPL[i], cmap='tab10', vmin=0,vmax=9)
            ax.set_xticks([]);ax.set_yticks([]);

FIG.tight_layout()

FIG.savefig(f"out/fig5_all_{NXY}", bbox_inches='tight')
plt.close(FIG)







""" """

II = ['TR_N_00000118_012_','TT_N_00010203_000_']
T = '_R50_KM_R50_PNEUMO_'
E = '.MODEL_CLUSTER.png'



TITLES = ['Original',
 'Training:\n',
 'augmentation\nno watermarks',
 'Training:\n',
 'no augmentation\nno watermarks',
 'Training:\n',
 'no augmentation\nwatermarks',
 'Training:\n',
 'augmentation\nwatermarks']

for _f,f in enumerate(['118','10203']):
    I = f+'/'+II[_f]
    plt.figure(1,(20,4))
    for _m,m in enumerate(['S','M']):
        plt.subplot(2,9,_m*9+1)
        plt.imshow(io.imread( I+m+'.png'))
        plt.axis('off')
        if _m == 0:
            plt.title(TITLES[0])
        for _t,t in enumerate(['SINGLE-1','SINGLE-1_NOTF','MARKED-1','MARKED-1_NOTF']):
            plt.subplot(2,9, 9*_m + 2*_t+2)
            plt.imshow(io.imread( I+m+T+t+E.replace('CLUSTER','BORDER') ))
            if _m == 0:
                plt.title(TITLES[2*_t+1])
            plt.axis('off')

            plt.subplot(2,9, 9*_m + 2*_t+3)
            plt.imshow(io.imread( I+m+T+t+E ))
            if _m == 0:
                plt.title(TITLES[2*_t+2])
            plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(.001, .02, .99, .85, 0,0)
    plt.savefig('ss_pneumo_'+f+'.pdf')
    plt.close()




def prepare_image(image, patch_size, normalize=True):
    # Adapted from original author(s) [1]:
    # [1] https://github.com/facebookresearch/dinov2/issues/412

    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list.append( transforms.Normalize(
                    mean=IMAGENET_DEFAULT_MEAN,
                    std=IMAGENET_DEFAULT_STD)
                    )
    transform = transforms.Compose(transform_list)
    image_tensor = transform(image)

    # Crop image to dimensions that are a multiple of the patch size
    height, width = image_tensor.shape[1:] # C x H x W
    cropped_width, cropped_height = width - width % patch_size, height - height % patch_size
    image_tensor = image_tensor[:, :cropped_height, :cropped_width]
    
    return image_tensor


####################

MODELS = [ torch.hub.load(repo_or_dir=DEFAULT_REPO_NAME, model=name).eval() for name in DEFAULT_MODEL_NAMES]


FIG, AXES = plt.subplots(nrows=5, ncols=4, figsize=(2*4, 2*6), dpi=150)

for fi,fname in enumerate(os.listdir('img/')):
    print(f'Processing: {fname}', ' '*20, end='\r')
    
    image = Image.open('img/'+fname).convert('RGB').resize(DEFAULT_RESIZE)
    image_tensor = prepare_image(image, model.patch_size)
    
    # PLOT ORIG
    ax = AXES[0][fi]
    ax.imshow(image, vmin=0,vmax=1)
    ax.set_xticks([]);ax.set_yticks([]);
    if fi == 0:
        ax.set_ylabel('Input', fontsize=11)

    for im, model in enumerate( MODELS ):
        
        with torch.inference_mode():
            image_batch = image_tensor.unsqueeze(0)
            tokens = model.get_intermediate_layers(image_batch, DEFAULT_LAYERS)
            tokens = torch.cat(tokens)
            tokens = torch.permute(tokens, (1, 2, 0)).reshape((tokens.shape[1], -1))
            tokens = tokens.squeeze().detach().cpu().numpy()

        expl = KMeans(n_clusters=DEFAULT_NB_CLUSTERS, init='k-means++', n_init=100, max_iter=1000, tol=1e-6).fit_predict(tokens)
        expl = expl.reshape((16,16))
        expl = cv2.resize( expl, DEFAULT_RESIZE, interpolation=cv2.INTER_NEAREST)
        
        # PLOT EXPLANATIONS
        ax = AXES[1+im][fi]
        ax.imshow(image, vmin=0, vmax=1)
        ax.imshow(expl, alpha=.6, cmap='tab10', vmin=0, vmax=9)
        ax.set_xticks([]);ax.set_yticks([]);
        if fi == 0:
            ax.set_ylabel(NAME_MAPPING[im], fontsize=11)
    
FIG.tight_layout()
plt.subplots_adjust(.01,.1,.99,.9,0,0)

FIG.savefig(f"out/fig4", bbox_inches='tight')
plt.close(FIG)

print(f'Done.', ' '*50, end='\n')
