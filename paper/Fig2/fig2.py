import os,sys,copy

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

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
DEFAULT_MODEL_NAME = 'dinov2_vits14' # 'dino_resnet50' #
DEFAULT_RESIZE_TO = 224
DEFAULT_LAYERS = (11,)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


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

model = torch.hub.load(repo_or_dir=DEFAULT_REPO_NAME, model=DEFAULT_MODEL_NAME)
model.eval()
if not hasattr(model,'patch_size'):
    model.patch_size = 1
    
for im,fname in enumerate(os.listdir('img/')):
    print(f'Processing: {fname}', ' '*20, end='\r')
    
    image = Image.open('img/'+fname).convert('RGB').resize((DEFAULT_RESIZE_TO,DEFAULT_RESIZE_TO))
    
    image_tensor = prepare_image(image, model.patch_size)
    
    with torch.inference_mode():
        image_batch = image_tensor.unsqueeze(0)
        tokens = model.get_intermediate_layers(image_batch, DEFAULT_LAYERS)
        tokens = torch.cat(tokens)
        tokens = torch.permute(tokens, (1, 2, 0)).reshape((tokens.shape[1], -1))
        tokens = tokens.squeeze().detach().cpu().numpy()

    proj = PCA(3).fit_transform(tokens)
    umap_proj = proj[:,:2]*0
    umap_proj[:,0] = proj[:,0] - proj[:,2]
    umap_proj[:,1] = proj[:,1] - proj[:,2]


    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(2*(1+4+1+4), 2*2+.9), dpi=100)

    # PLOT ORIG
    ax = axes[0][0]
    ax.imshow(image, vmin=0,vmax=1)
    ax.set_xticks([]);ax.set_yticks([]);
    ax.set_title('Input', fontsize=14)

    ax = axes[1][0]
    ax.cla()
    ax.axis('off')

    ax = axes[0][5]
    ax.cla()
    ax.axis('off')

    ax = axes[1][5]
    ax.cla()
    ax.axis('off')
    
    old_pca, old_km = [],[]

    for ik,k in enumerate([3,6,10,20]):
        
        ####################
        # PCA/DINO
        
        np.random.seed(42)
        out_pca = np.abs(PCA(k).fit_transform(tokens)).argmax(-1)
        
        if k == 3:
            old_pca = out_pca[:]
            CMAP = mpl.colormaps['Spectral'].resampled(3)
        else:
            C = contingency_matrix(old_pca, out_pca)
            ax_C = C.argmax(0)
            rg_C = np.bincount(ax_C).max()
            
            CMAP = []
            new_pca = -np.ones(out_pca.shape)
            nk = 0
            for ok in range(3):
                sc_C = (ax_C == ok).nonzero()[0]
                as_C = (-C[ok,sc_C]).argsort() / rg_C
                
                for il,l in enumerate(sc_C):
                    new_pca[out_pca == l] = nk
                    nk += 1
            out_pca = new_pca.astype(int)
            CMAP = mpl.colormaps['Spectral'].resampled(nk-1)
        

        ax = axes[0][1+ik]
        ax.cla()
        ax.imshow( out_pca.reshape((16,16)), alpha=1, cmap=CMAP, vmin=0, vmax=out_pca.max())
        ax.set_xticks([]);ax.set_yticks([]);

        ax.set_title( ('DINO/PCA\n' if ik == 1 else '')+f'K={k}', fontsize=14)

        ax = axes[1][1+ik]
        ax.cla()
        ax.scatter( umap_proj[:,0], umap_proj[:,1], c=out_pca, marker='.',
                        cmap=CMAP, vmin=0, vmax=out_pca.max())
        ax.set_xticks([]);ax.set_yticks([]);


        ####################
        # KMEANS/NAVE
        
        np.random.seed(42)
        out_km  = KMeans(n_clusters=k, init='k-means++', n_init=100, max_iter=1000, tol=1e-6).fit_predict(tokens)
                
        if k == 3:
            old_km = out_km[:]
            CMAP = mpl.colormaps['Spectral'].resampled(3)
        else:
            old_k = old_km.max()+1
            C = contingency_matrix(old_km, out_km)
            ax_C = C.argmax(0)
            rg_C = np.bincount(ax_C).max()
            
            CMAP = []
            new_km = -np.ones(out_km.shape)
            nk = 0
            for ok in range(3):
                sc_C = (ax_C == ok).nonzero()[0]
                as_C = (-C[ok,sc_C]).argsort() / rg_C
                
                for il,l in enumerate(sc_C):
                    new_km[out_km == l] = nk
                    nk += 1
            out_km = new_km.astype(int)
            CMAP = mpl.colormaps['Spectral'].resampled(nk-1)

        ax = axes[0][6+ik]
        ax.cla()
        ax.imshow( out_km.reshape((16,16)), alpha=1, cmap=CMAP, vmin=0, vmax=out_km.max())
        ax.set_xticks([]);ax.set_yticks([]);
        ax.set_title( ('NAVE/KMeans\n' if ik == 1 else '')+f'K={k}', fontsize=14)

        ax = axes[1][6+ik]
        ax.cla()
        ax.scatter( umap_proj[:,0], umap_proj[:,1], c=out_km, marker='.',
                        cmap=CMAP, vmin=0, vmax=out_km.max())
        ax.set_xticks([]);ax.set_yticks([]);


    fig.tight_layout()
    plt.subplots_adjust(.01,.1,.99,.9,0,0)

    fig.savefig(f"out/fig2_{fname}", bbox_inches='tight')
    plt.close(fig)
print(f'Done.', ' '*50, end='\n')
