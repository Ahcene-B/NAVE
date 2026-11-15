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
