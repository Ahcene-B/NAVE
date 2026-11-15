import os,sys,copy

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import cv2
from PIL import Image

from craft.craft_torch import Craft, torch_to_numpy

from sklearn.cluster import KMeans

from skimage.segmentation import slic,felzenszwalb
from skimage.transform import resize
from skimage.segmentation import mark_boundaries
from scipy.ndimage import binary_dilation

from skimage import measure

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', category=PendingDeprecationWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

CUDA = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "cpu"


####################

DEFAULT_RESIZE = (224,224)
DEFAULT_LAYERS = [2,3,4]
DEFAULT_NB_CLUSTER = 10

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])


####################


# Get Model

MODEL = torchvision.models.resnet50(pretrained=True).to(DEVICE)
MODEL.eval()

ENCODER = nn.Sequential(*(list(MODEL.children())[:-2])) # input to penultimate layer
CLF = lambda x: MODEL.fc(torch.mean(x, (2, 3))) # penultimate layer to logits


pbar = tqdm(os.listdir('img/'))
for fname in pbar:
    image = Image.open('img/'+fname).convert('RGB').resize(DEFAULT_RESIZE)
    IMAGE = np.array(image)[None,:]
    TENSOR_IMAGE = TRANSFORM(image).to(DEVICE)[None,:]
    
    EXPL = {}


    pbar.set_description(f"# CRAFT")

    CRAFT = Craft(input_to_latent = ENCODER,
                  latent_to_logit = CLF,
                  number_of_concepts = DEFAULT_NB_CLUSTER,
                  patch_size = 64,
                  batch_size = 64,
                  device = DEVICE)

    CRAFT.fit(TENSOR_IMAGE)
    CRAFT_EXPL = CRAFT.transform( TENSOR_IMAGE )
    CRAFT_SIGMA = np.percentile( CRAFT_EXPL.flatten(), 90)

    craft_msk = ( CRAFT_EXPL >= CRAFT_SIGMA ).any(-1)
    craft_msk = craft_msk / craft_msk

    CRAFT_EXPL = CRAFT_EXPL.argmax(-1) * craft_msk
    CRAFT_EXPL = np.nan_to_num( CRAFT_EXPL, nan=-1).astype(int)
    EXPL['CRAFT'] = cv2.resize( CRAFT_EXPL[0], DEFAULT_RESIZE, interpolation=cv2.INTER_NEAREST)
 

    pbar.set_description(f"# CONE-SHAP")

    UPSAMPLE = nn.Upsample(size=DEFAULT_RESIZE, mode='bicubic').to(DEVICE)
    PATCHES = []

    for ns in [15,50,80]:
        sppx = slic(IMAGE[0], n_segments=ns)
        for sx in np.unique(sppx):
            patch_msk   = (sppx == sx)
            nx,ny = patch_msk.nonzero()
            patch = (torch.tensor(patch_msk[None,:,:]).to(DEVICE)*TENSOR_IMAGE[0])[None, :, min(nx):max(nx), min(ny):max(ny)]
            patch = UPSAMPLE(patch)
            patch = ENCODER(patch).flatten()
            PATCHES.append( patch.detach().cpu().numpy() )
    PATCHES = np.array(PATCHES)
    CONE_KM = KMeans(DEFAULT_NB_CLUSTER).fit( PATCHES )

    EXPL['CONE-SHAP'] = np.zeros(DEFAULT_RESIZE)
    sppx = slic(IMAGE[0], n_segments=50, compactness=10)
    for sx in np.unique(sppx):
        cone_msk   = (sppx == sx)
        nx,ny = cone_msk.nonzero()
        patch = (torch.tensor(cone_msk[None,:,:]).to(DEVICE)*TENSOR_IMAGE[0])[None, :, min(nx):max(nx), min(ny):max(ny)]
        patch = UPSAMPLE(patch)
        patch = ENCODER(patch).squeeze().flatten()
        patch = patch.detach().cpu().numpy()[None,:]
        pred = CONE_KM.predict( patch )[0]
        EXPL['CONE-SHAP'] += cone_msk * pred


    pbar.set_description(f"# Net2Vec")

    EXPL['Net2Vec'] = np.zeros(DEFAULT_RESIZE)

    maps = ENCODER(TENSOR_IMAGE).argmax(1).squeeze().detach().cpu().numpy()
    maps = cv2.resize(maps, DEFAULT_RESIZE, interpolation=cv2.INTER_NEAREST)
    for p,m in enumerate( np.unique( maps ) ):
        n2v_msk = ((-maps-1) == -m-1)
        EXPL['Net2Vec'] += n2v_msk * p


    pbar.set_description(f"# NAVE")

    feats = {}

    def save(name):
        def hook(module, args, output):
            feats[name] = output.detach()   # detach: don't keep the autograd graph alive
        return hook

    handles = [getattr(MODEL, ly).register_forward_hook(save(ly)) for ly in ("layer2", "layer3", "layer4")]

    with torch.no_grad():
        MODEL(TENSOR_IMAGE)

    for h in handles:
        h.remove()

    for NXY in ['N234','N34','N4']:
        EXPL[NXY] = np.zeros(DEFAULT_RESIZE)

        ACTIV = [ feats['layer'+ly] for ly in NXY[1:] ]

        u_sz = np.array([act.shape[-1] for act in ACTIV])
        UPSAMPLE = nn.Upsample(size=(max(u_sz),max(u_sz)), mode='bicubic').to(DEVICE)

        ACTIV = [ torch.permute( UPSAMPLE(act), (1, 0,2,3) ).flatten(1).T   for act in ACTIV ]
        ACTIV = [ act / torch.norm( act, dim=1, keepdim=True)   for act in ACTIV ]
        ACTIV = [ ACTIV[i] / ( min(u_sz)/u_sz[i] )   for i in range(len(ACTIV)) ]
        ACTIV = torch.cat( ACTIV, 1)

        maps = KMeans(DEFAULT_NB_CLUSTER).fit_predict( ACTIV )
        maps = maps.reshape( max(u_sz), max(u_sz) )
        
        EXPL[NXY] = cv2.resize( maps, DEFAULT_RESIZE, interpolation=cv2.INTER_NEAREST)



    pbar.set_description(f"# Plot")
    
    fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(2*7, 3), dpi=200)
    ax = axes[0]
    ax.imshow(IMAGE[0])
    ax.set_xticks([]);ax.set_yticks([]);
    ax.set_title('Input', fontsize=14)
    
    for it, TITLE in enumerate(['CRAFT', 'CONE-SHAP', 'Net2Vec', 'N4', 'N34', 'N234']):
        maps = EXPL[TITLE]
    
        ax = axes[1+it]
        ax.cla()
        ax.imshow(IMAGE[0])
        ax.imshow( EXPL[TITLE], alpha=(EXPL[TITLE]>-1)+0., 
                cmap=('tab10' if TITLE != 'Net2Vec' else 'Spectral'), 
                vmin=0, 
                vmax=(9 if TITLE != 'Net2Vec' else np.nanmax(EXPL[TITLE])),
                )
        ax.axis('off')
        ax.set_title( TITLE, fontsize=14)
        fig.savefig(f"out/fig3_{fname}", bbox_inches='tight')
        
    fig.tight_layout()

    fig.savefig(f"out/fig3_{fname}", bbox_inches='tight')
    plt.close(fig)

