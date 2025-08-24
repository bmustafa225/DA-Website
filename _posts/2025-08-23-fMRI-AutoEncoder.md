----
layout: post
title: "fMRI_Autoencoder"
------
# Data and Study Design
This project explores the application of autoencoders in medical imaging, with a specific focus on reconstructing brain fMRI scans. The dataset was obtained from [OpenfMRI](https://openfmri.org/) (now integrated into OpenNeuro), specifically the [Balloon Analog Risk Taking (BART) dataset, DS000001](https://openfmri.org/dataset/ds000001/). The dataset contains imaging data from 28 subjects, with all scans preprocessed and anonymized to remove identifiable physical markers.

For this study, only the T1-weighted (T1w) structural MRI scans were used. These scans provide high-resolution anatomical information and were selected to simplify the reconstruction task, as the primary objective was to evaluate the feasibility of autoencoder-based dimensionality reduction and image reconstruction rather than to analyze the task-specific functional data. The BART dataset was chosen largely due to its smaller size and accessibility, making it a practical starting point for a proof-of-concept study.

The experimental workflow proceeded in two stages:

Single-subject reconstruction — An autoencoder was trained and tested on the T1w scan of a single subject to assess baseline performance.

Multi-subject reconstruction — The analysis was extended to include all 28 subjects, enabling evaluation of the model’s ability to generalize across individuals.


```python
! pip install nilearn
```

    Collecting nilearn
      Downloading nilearn-0.12.0-py3-none-any.whl.metadata (9.9 kB)
    Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.12/dist-packages (from nilearn) (1.5.1)
    Requirement already satisfied: lxml in /usr/local/lib/python3.12/dist-packages (from nilearn) (5.4.0)
    Requirement already satisfied: nibabel>=5.2.0 in /usr/local/lib/python3.12/dist-packages (from nilearn) (5.3.2)
    Requirement already satisfied: numpy>=1.22.4 in /usr/local/lib/python3.12/dist-packages (from nilearn) (2.0.2)
    Requirement already satisfied: packaging in /usr/local/lib/python3.12/dist-packages (from nilearn) (25.0)
    Requirement already satisfied: pandas>=2.2.0 in /usr/local/lib/python3.12/dist-packages (from nilearn) (2.2.2)
    Requirement already satisfied: requests>=2.25.0 in /usr/local/lib/python3.12/dist-packages (from nilearn) (2.32.4)
    Requirement already satisfied: scikit-learn>=1.4.0 in /usr/local/lib/python3.12/dist-packages (from nilearn) (1.6.1)
    Requirement already satisfied: scipy>=1.8.0 in /usr/local/lib/python3.12/dist-packages (from nilearn) (1.16.1)
    Requirement already satisfied: typing-extensions>=4.6 in /usr/local/lib/python3.12/dist-packages (from nibabel>=5.2.0->nilearn) (4.14.1)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.12/dist-packages (from pandas>=2.2.0->nilearn) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.12/dist-packages (from pandas>=2.2.0->nilearn) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.12/dist-packages (from pandas>=2.2.0->nilearn) (2025.2)
    Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests>=2.25.0->nilearn) (3.4.3)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/dist-packages (from requests>=2.25.0->nilearn) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests>=2.25.0->nilearn) (2.5.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.12/dist-packages (from requests>=2.25.0->nilearn) (2025.8.3)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn>=1.4.0->nilearn) (3.6.0)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.8.2->pandas>=2.2.0->nilearn) (1.17.0)
    Downloading nilearn-0.12.0-py3-none-any.whl (10.6 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m10.6/10.6 MB[0m [31m96.9 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: nilearn
    Successfully installed nilearn-0.12.0
    


```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nb
import nilearn as nl
import matplotlib.pyplot as plt
from nilearn import masking
from nilearn.masking import unmask,apply_mask
from nilearn.plotting import plot_roi, plot_anat
```


```python

```


```python

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /tmp/ipython-input-4123536944.py in <cell line: 0>()
    ----> 1 plt.imshow(brain_vol[88],cmap='bone')
          2 plt.axis('off')
          3 plt.show()
    

    NameError: name 'brain_vol' is not defined



```python

```


```python

```


```python

```


```python
z.shape
```


```python

```


```python

```


```python

```

Now that we see out autoencoder performing reasonably well on a single fMRI image, lets try scaling the operation using LightningModule for the PyLightning library!


```python
! pip install lightning
```

    Collecting lightning
      Downloading lightning-2.5.3-py3-none-any.whl.metadata (39 kB)
    Requirement already satisfied: PyYAML<8.0,>5.4 in /usr/local/lib/python3.12/dist-packages (from lightning) (6.0.2)
    Requirement already satisfied: fsspec<2027.0,>=2022.5.0 in /usr/local/lib/python3.12/dist-packages (from fsspec[http]<2027.0,>=2022.5.0->lightning) (2025.3.0)
    Collecting lightning-utilities<2.0,>=0.10.0 (from lightning)
      Downloading lightning_utilities-0.15.2-py3-none-any.whl.metadata (5.7 kB)
    Requirement already satisfied: packaging<27.0,>=20.0 in /usr/local/lib/python3.12/dist-packages (from lightning) (25.0)
    Requirement already satisfied: torch<4.0,>=2.1.0 in /usr/local/lib/python3.12/dist-packages (from lightning) (2.8.0+cu126)
    Collecting torchmetrics<3.0,>0.7.0 (from lightning)
      Downloading torchmetrics-1.8.1-py3-none-any.whl.metadata (22 kB)
    Requirement already satisfied: tqdm<6.0,>=4.57.0 in /usr/local/lib/python3.12/dist-packages (from lightning) (4.67.1)
    Requirement already satisfied: typing-extensions<6.0,>4.5.0 in /usr/local/lib/python3.12/dist-packages (from lightning) (4.14.1)
    Collecting pytorch-lightning (from lightning)
      Downloading pytorch_lightning-2.5.3-py3-none-any.whl.metadata (20 kB)
    Requirement already satisfied: aiohttp!=4.0.0a0,!=4.0.0a1 in /usr/local/lib/python3.12/dist-packages (from fsspec[http]<2027.0,>=2022.5.0->lightning) (3.12.15)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.12/dist-packages (from lightning-utilities<2.0,>=0.10.0->lightning) (75.2.0)
    Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from torch<4.0,>=2.1.0->lightning) (3.19.1)
    Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/dist-packages (from torch<4.0,>=2.1.0->lightning) (1.13.3)
    Requirement already satisfied: networkx in /usr/local/lib/python3.12/dist-packages (from torch<4.0,>=2.1.0->lightning) (3.5)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from torch<4.0,>=2.1.0->lightning) (3.1.6)
    Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch<4.0,>=2.1.0->lightning) (12.6.77)
    Requirement already satisfied: nvidia-cuda-runtime-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch<4.0,>=2.1.0->lightning) (12.6.77)
    Requirement already satisfied: nvidia-cuda-cupti-cu12==12.6.80 in /usr/local/lib/python3.12/dist-packages (from torch<4.0,>=2.1.0->lightning) (12.6.80)
    Requirement already satisfied: nvidia-cudnn-cu12==9.10.2.21 in /usr/local/lib/python3.12/dist-packages (from torch<4.0,>=2.1.0->lightning) (9.10.2.21)
    Requirement already satisfied: nvidia-cublas-cu12==12.6.4.1 in /usr/local/lib/python3.12/dist-packages (from torch<4.0,>=2.1.0->lightning) (12.6.4.1)
    Requirement already satisfied: nvidia-cufft-cu12==11.3.0.4 in /usr/local/lib/python3.12/dist-packages (from torch<4.0,>=2.1.0->lightning) (11.3.0.4)
    Requirement already satisfied: nvidia-curand-cu12==10.3.7.77 in /usr/local/lib/python3.12/dist-packages (from torch<4.0,>=2.1.0->lightning) (10.3.7.77)
    Requirement already satisfied: nvidia-cusolver-cu12==11.7.1.2 in /usr/local/lib/python3.12/dist-packages (from torch<4.0,>=2.1.0->lightning) (11.7.1.2)
    Requirement already satisfied: nvidia-cusparse-cu12==12.5.4.2 in /usr/local/lib/python3.12/dist-packages (from torch<4.0,>=2.1.0->lightning) (12.5.4.2)
    Requirement already satisfied: nvidia-cusparselt-cu12==0.7.1 in /usr/local/lib/python3.12/dist-packages (from torch<4.0,>=2.1.0->lightning) (0.7.1)
    Requirement already satisfied: nvidia-nccl-cu12==2.27.3 in /usr/local/lib/python3.12/dist-packages (from torch<4.0,>=2.1.0->lightning) (2.27.3)
    Requirement already satisfied: nvidia-nvtx-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch<4.0,>=2.1.0->lightning) (12.6.77)
    Requirement already satisfied: nvidia-nvjitlink-cu12==12.6.85 in /usr/local/lib/python3.12/dist-packages (from torch<4.0,>=2.1.0->lightning) (12.6.85)
    Requirement already satisfied: nvidia-cufile-cu12==1.11.1.6 in /usr/local/lib/python3.12/dist-packages (from torch<4.0,>=2.1.0->lightning) (1.11.1.6)
    Requirement already satisfied: triton==3.4.0 in /usr/local/lib/python3.12/dist-packages (from torch<4.0,>=2.1.0->lightning) (3.4.0)
    Requirement already satisfied: numpy>1.20.0 in /usr/local/lib/python3.12/dist-packages (from torchmetrics<3.0,>0.7.0->lightning) (2.0.2)
    Requirement already satisfied: aiohappyeyeballs>=2.5.0 in /usr/local/lib/python3.12/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<2027.0,>=2022.5.0->lightning) (2.6.1)
    Requirement already satisfied: aiosignal>=1.4.0 in /usr/local/lib/python3.12/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<2027.0,>=2022.5.0->lightning) (1.4.0)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.12/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<2027.0,>=2022.5.0->lightning) (25.3.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.12/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<2027.0,>=2022.5.0->lightning) (1.7.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.12/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<2027.0,>=2022.5.0->lightning) (6.6.4)
    Requirement already satisfied: propcache>=0.2.0 in /usr/local/lib/python3.12/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<2027.0,>=2022.5.0->lightning) (0.3.2)
    Requirement already satisfied: yarl<2.0,>=1.17.0 in /usr/local/lib/python3.12/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<2027.0,>=2022.5.0->lightning) (1.20.1)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from sympy>=1.13.3->torch<4.0,>=2.1.0->lightning) (1.3.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->torch<4.0,>=2.1.0->lightning) (3.0.2)
    Requirement already satisfied: idna>=2.0 in /usr/local/lib/python3.12/dist-packages (from yarl<2.0,>=1.17.0->aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<2027.0,>=2022.5.0->lightning) (3.10)
    Downloading lightning-2.5.3-py3-none-any.whl (824 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m824.2/824.2 kB[0m [31m18.0 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading lightning_utilities-0.15.2-py3-none-any.whl (29 kB)
    Downloading torchmetrics-1.8.1-py3-none-any.whl (982 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m983.0/983.0 kB[0m [31m44.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading pytorch_lightning-2.5.3-py3-none-any.whl (828 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m828.2/828.2 kB[0m [31m61.1 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: lightning-utilities, torchmetrics, pytorch-lightning, lightning
    Successfully installed lightning-2.5.3 lightning-utilities-0.15.2 pytorch-lightning-2.5.3 torchmetrics-1.8.1
    

# Methods: Autoencoder Construction
An autoencoder is a type of neural network designed to learn efficient, lower-dimensional representations of high-dimensional data. It consists of two primary components: an encoder, which compresses the input data into a compact latent space, and a decoder, which attempts to reconstruct the original input from this latent representation. The two components are approximately inverse in function, though not strictly architectural mirrors.

Dimensionality reduction through autoencoders offers a powerful approach to simplifying complex data while preserving essential features. Beyond compression, the ability to reconstruct data of the same class makes autoencoders particularly valuable in cases where the original data is incomplete, corrupted, or inaccessible. In medical imaging, this approach can be applied to reconstruct modalities such as fMRI, X-ray, or MRI scans, offering potential advantages in data recovery, storage efficiency, and accessibility in resource-limited settings.

In this project, the autoencoder was implemented using PyTorch Lightning. The encoder and decoder networks were defined separately and integrated into a unified LightningModule. Data handling included loading, preprocessing (normalization and augmentation as needed), and splitting into training and validation sets. Model performance was evaluated using Mean Squared Error (MSE) between reconstructed outputs and the original images, providing a quantitative measure of reconstruction accuracy.


```python
class BrainEncoder(nn.Module):
  def __init__(self, input: int, base: int,latent:int, act_fn: object=nn.GELU ):
    super().__init__()
    n_hid=base
    self.net=nn.Sequential(
        nn.Conv3d(input,n_hid,kernel_size=3,padding=1,stride=2), #256x256 --> 128x128
        act_fn(),
        nn.Conv3d(n_hid,n_hid,kernel_size=3,padding=1),
        act_fn(),
        nn.Conv3d(n_hid,2*n_hid,kernel_size=3,padding=1,stride=2), #128x128 --> 64x64
        act_fn(),
        nn.Conv3d(2*n_hid,2*n_hid,kernel_size=3,padding=1),
        act_fn(),
        nn.Conv3d(2*n_hid,4*n_hid,kernel_size=3,padding=1,stride=2), #64x64 --> 32x32
        act_fn(),
        nn.Conv3d(4*n_hid,4*n_hid,kernel_size=3,padding=1),
        act_fn(),
        nn.Conv3d(4*n_hid,4*n_hid,kernel_size=3,padding=1,stride=2),#32x32 --> 16x16
        act_fn(),
        nn.AdaptiveAvgPool3d((4,4,4)),
        nn.Flatten(),
        nn.Linear(4 * base * 4 * 4 * 4, latent)
      )
  def forward(self,x):
    return self.net(x)

```


```python
class BrainDecoder(nn.Module):
  def __init__(self, input: int, base: int,latent:int, act_fn: object=nn.GELU ):
    self.base=base
    super().__init__()
    self.Linear=nn.Sequential(nn.Linear(latent,4*4*4*4*base),act_fn())
    n_hid=base
    self.net=nn.Sequential(
        nn.ConvTranspose3d(4*n_hid,4*n_hid,padding=1,stride=2,kernel_size=4), #256x256 --> 128x128
        act_fn(),
        nn.ConvTranspose3d(4*n_hid,4*n_hid,kernel_size=4,padding=1),
        act_fn(),
        nn.ConvTranspose3d(4*n_hid,2*n_hid,kernel_size=4,padding=1,stride=2), #128x128 --> 64x64
        act_fn(),
        nn.ConvTranspose3d(2*n_hid,2*n_hid,kernel_size=4,padding=1),
        act_fn(),
        nn.ConvTranspose3d(2*n_hid,2*n_hid,kernel_size=4,padding=1,stride=2),
        act_fn(),
        nn.ConvTranspose3d(2*n_hid,2*n_hid,kernel_size=4,padding=1),
        act_fn(),
        nn.ConvTranspose3d(2*n_hid,n_hid,kernel_size=4,padding=1,stride=2), #64x64 --> 32x32
        act_fn(),
        nn.ConvTranspose3d(n_hid,n_hid,kernel_size=4,padding=1),
        act_fn(),
        nn.ConvTranspose3d(n_hid,input,kernel_size=4,padding=1,stride=2) #32x32 --> 16x16
        #nn.Sigmoid()
      )
  def forward(self,x):
    x=self.Linear(x)
    x=x.unflatten(1,(4*self.base,4,4,4))
    x=self.net(x)
    x=F.interpolate(x, size=(176, 256, 256), mode='trilinear')
    return x

```


```python
brain=nb.load('sub-16_T1w.nii.gz')
brain_vol=brain.get_fdata()


plt.imshow(brain_vol[88],cmap='bone')
plt.axis('off')
plt.show()

plot_roi(masker,bg_img=brain)
plot_anat(masker, title="Isolated Brain Mask")
```


```python
masker=masking.compute_brain_mask(brain)
q=masker.get_fdata()


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
encode=BrainEncoder(1,16,64).to(device)

mask_weight=masker.get_fdata()
brain_mask=brain_vol*mask_weight

torch_brain=torch.from_numpy(brain_mask.astype(np.float32))
torch_brain=torch_brain.unsqueeze(0).unsqueeze(0)
z=encode(torch_brain)

decoder=BrainDecoder(1,16,64).to(device)
z_hat=decoder(z)

tensor_mask=torch.from_numpy(mask_weight).unsqueeze(0).unsqueeze(0).to(device)
z_hat_masked = z_hat * tensor_mask
input_masked = torch_brain * tensor_mask
loss = F.mse_loss(z_hat_masked, input_masked, reduction="mean")


from nilearn.masking import unmask

reconstructed_fmri= nb.Nifti1Image(z_hat_masked.squeeze().detach().cpu().numpy(), affine=masker.affine, header=masker.header)
recon_img=reconstructed_fmri.get_fdata()

plt.imshow(recon_img[100,:,:],cmap='bone')
plt.axis('off')
plt.show()
```

For a simple single person model building we show the results above, the next few cells show the incrporation of the two parts and defining the loss function.


```python
import torch.optim as optim
import lightning as L

class BrainAutoEncoder(L.LightningModule):
  def __init__(self,
               encoder: object= BrainEncoder,
               decoder: object= BrainDecoder,
               input: int=1 ,base: int=64, latent: int=128,
               brain_mask=None):
    super().__init__()
    self.encoder=encoder(input,base,latent)
    self.decoder=decoder(input,base,latent)


  def forward(self,x):
    z=self.encoder(x)
    x_hat=self.decoder(z)
    return x_hat

  def get_validation_loss(self,batch):
    x=batch[0]
    x_hat=self.forward(x)
    loss=F.mse_loss(x_hat,x , reduction="mean")
    return loss

  def configure_optimizers(self):
    optimizer=optim.Adam(self.parameters(),lr=1e-3)
    return optimizer


  def training_step(self,batch,batch_idx):
    loss_tr=self.get_validation_loss(batch)
    self.log('train_loss',loss_tr)
    return loss_tr

  def validation_step(self,batch,batch_idx):
    loss_vl=self.get_validation_loss(batch)
    self.log('validation_loss',loss_vl)

  def test_step(self,batch):
    loss_tst=self.get_validation_loss(batch)
    self.log('test_loss',loss_tst)


```

To improve the model’s ability to generalize and reduce irrelevant variance, a preprocessing and augmentation pipeline was implemented. The key steps were as follows:

* Image Rotation — Small random rotations were applied to increase variability and improve the model’s ability to learn orientation-invariant features.

* Region of Interest (ROI) Masking — A brain extraction mask was applied to remove non-brain tissue (e.g., skull, bone, and surrounding background), ensuring that the model focused exclusively on relevant anatomical structures.

* Intensity Scaling and Normalization — Voxel intensity values were rescaled to a consistent range and normalized across scans. This step stabilized training by standardizing input distributions and reducing bias from subject-specific or scanner-specific intensity differences.

These preprocessing operations refined the dataset, simplified the reconstruction task, and ensured that the autoencoder concentrated on salient structural information in the T1-weighted MRI volumes.


```python
!pip install torchio
```

    Collecting torchio
      Downloading torchio-0.20.21-py3-none-any.whl.metadata (53 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m53.1/53.1 kB[0m [31m2.3 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting deprecated>=1.2 (from torchio)
      Downloading Deprecated-1.2.18-py2.py3-none-any.whl.metadata (5.7 kB)
    Requirement already satisfied: einops>=0.3 in /usr/local/lib/python3.12/dist-packages (from torchio) (0.8.1)
    Requirement already satisfied: humanize>=0.1 in /usr/local/lib/python3.12/dist-packages (from torchio) (4.12.3)
    Requirement already satisfied: nibabel>=3 in /usr/local/lib/python3.12/dist-packages (from torchio) (5.3.2)
    Requirement already satisfied: numpy>=1.20 in /usr/local/lib/python3.12/dist-packages (from torchio) (2.0.2)
    Requirement already satisfied: packaging>=20 in /usr/local/lib/python3.12/dist-packages (from torchio) (25.0)
    Requirement already satisfied: rich>=10 in /usr/local/lib/python3.12/dist-packages (from torchio) (13.9.4)
    Requirement already satisfied: scipy>=1.7 in /usr/local/lib/python3.12/dist-packages (from torchio) (1.16.1)
    Collecting simpleitk!=2.0.*,!=2.1.1.1,>=1.3 (from torchio)
      Downloading simpleitk-2.5.2-cp311-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (7.2 kB)
    Requirement already satisfied: torch>=1.9 in /usr/local/lib/python3.12/dist-packages (from torchio) (2.8.0+cu126)
    Requirement already satisfied: tqdm>=4.40 in /usr/local/lib/python3.12/dist-packages (from torchio) (4.67.1)
    Requirement already satisfied: typer>=0.1 in /usr/local/lib/python3.12/dist-packages (from torchio) (0.16.0)
    Requirement already satisfied: wrapt<2,>=1.10 in /usr/local/lib/python3.12/dist-packages (from deprecated>=1.2->torchio) (1.17.3)
    Requirement already satisfied: typing-extensions>=4.6 in /usr/local/lib/python3.12/dist-packages (from nibabel>=3->torchio) (4.14.1)
    Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.12/dist-packages (from rich>=10->torchio) (4.0.0)
    Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.12/dist-packages (from rich>=10->torchio) (2.19.2)
    Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from torch>=1.9->torchio) (3.19.1)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.12/dist-packages (from torch>=1.9->torchio) (75.2.0)
    Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/dist-packages (from torch>=1.9->torchio) (1.13.3)
    Requirement already satisfied: networkx in /usr/local/lib/python3.12/dist-packages (from torch>=1.9->torchio) (3.5)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from torch>=1.9->torchio) (3.1.6)
    Requirement already satisfied: fsspec in /usr/local/lib/python3.12/dist-packages (from torch>=1.9->torchio) (2025.3.0)
    Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch>=1.9->torchio) (12.6.77)
    Requirement already satisfied: nvidia-cuda-runtime-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch>=1.9->torchio) (12.6.77)
    Requirement already satisfied: nvidia-cuda-cupti-cu12==12.6.80 in /usr/local/lib/python3.12/dist-packages (from torch>=1.9->torchio) (12.6.80)
    Requirement already satisfied: nvidia-cudnn-cu12==9.10.2.21 in /usr/local/lib/python3.12/dist-packages (from torch>=1.9->torchio) (9.10.2.21)
    Requirement already satisfied: nvidia-cublas-cu12==12.6.4.1 in /usr/local/lib/python3.12/dist-packages (from torch>=1.9->torchio) (12.6.4.1)
    Requirement already satisfied: nvidia-cufft-cu12==11.3.0.4 in /usr/local/lib/python3.12/dist-packages (from torch>=1.9->torchio) (11.3.0.4)
    Requirement already satisfied: nvidia-curand-cu12==10.3.7.77 in /usr/local/lib/python3.12/dist-packages (from torch>=1.9->torchio) (10.3.7.77)
    Requirement already satisfied: nvidia-cusolver-cu12==11.7.1.2 in /usr/local/lib/python3.12/dist-packages (from torch>=1.9->torchio) (11.7.1.2)
    Requirement already satisfied: nvidia-cusparse-cu12==12.5.4.2 in /usr/local/lib/python3.12/dist-packages (from torch>=1.9->torchio) (12.5.4.2)
    Requirement already satisfied: nvidia-cusparselt-cu12==0.7.1 in /usr/local/lib/python3.12/dist-packages (from torch>=1.9->torchio) (0.7.1)
    Requirement already satisfied: nvidia-nccl-cu12==2.27.3 in /usr/local/lib/python3.12/dist-packages (from torch>=1.9->torchio) (2.27.3)
    Requirement already satisfied: nvidia-nvtx-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch>=1.9->torchio) (12.6.77)
    Requirement already satisfied: nvidia-nvjitlink-cu12==12.6.85 in /usr/local/lib/python3.12/dist-packages (from torch>=1.9->torchio) (12.6.85)
    Requirement already satisfied: nvidia-cufile-cu12==1.11.1.6 in /usr/local/lib/python3.12/dist-packages (from torch>=1.9->torchio) (1.11.1.6)
    Requirement already satisfied: triton==3.4.0 in /usr/local/lib/python3.12/dist-packages (from torch>=1.9->torchio) (3.4.0)
    Requirement already satisfied: click>=8.0.0 in /usr/local/lib/python3.12/dist-packages (from typer>=0.1->torchio) (8.2.1)
    Requirement already satisfied: shellingham>=1.3.0 in /usr/local/lib/python3.12/dist-packages (from typer>=0.1->torchio) (1.5.4)
    Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.12/dist-packages (from markdown-it-py>=2.2.0->rich>=10->torchio) (0.1.2)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from sympy>=1.13.3->torch>=1.9->torchio) (1.3.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->torch>=1.9->torchio) (3.0.2)
    Downloading torchio-0.20.21-py3-none-any.whl (194 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m194.2/194.2 kB[0m [31m6.7 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading Deprecated-1.2.18-py2.py3-none-any.whl (10.0 kB)
    Downloading simpleitk-2.5.2-cp311-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (52.6 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m52.6/52.6 MB[0m [31m33.0 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: simpleitk, deprecated, torchio
    Successfully installed deprecated-1.2.18 simpleitk-2.5.2 torchio-0.20.21
    


```python
from typing_extensions import dataclass_transform
import torchio as tio
import zipfile
from torch.utils.data import Dataset
from io import BytesIO
import tempfile

class T1WDataset_mask(Dataset):
  def __init__(self,zip_path,mask_image=True,transform=True):
    self.mask=mask_image
    self.zip_path=zip_path
    with zipfile.ZipFile(zip_path,'r') as z:
      self.image_listdir=[f for f in z.namelist() if f.endswith('.nii') or f.endswith('.nii.gz')]
    self.transform=transform
    # Define TorchIO transform
    self.augment=tio.Compose([tio.CropOrPad((176,256,256)),
                              tio.RandomAffine(scales=(0.95, 1.05), degrees=5)
    ])

  def __len__(self):
    return len(self.image_listdir)

  def __getitem__(self,idx):
    with zipfile.ZipFile(self.zip_path, 'r') as z:
      file_name = self.image_listdir[idx]
      with tempfile.NamedTemporaryFile(suffix='.nii.gz') as tmp:
        tmp.write(z.read(file_name))
        tmp.flush()
        img = nb.load(tmp.name)
        data = img.get_fdata().astype(np.float32)
        affine = img.affine

    scaled_data=( data - data.min())/( data.max() - data.min() + 1e-8)

    if self.mask:
      mask_img=masking.compute_brain_mask(nb.nifti1.Nifti1Image(scaled_data,affine))
      mask=mask_img.get_fdata().astype(np.float32)
    else:
      mask = np.ones_like(scaled_data, dtype=np.float32)

    t1_tensor=torch.from_numpy(scaled_data).unsqueeze(0)
    mask_tensor=torch.from_numpy(mask).unsqueeze(0)

    if self.transform:

      sub=tio.Subject(
          t1w=tio.ScalarImage(tensor=t1_tensor),
          msk=tio.LabelMap(tensor=mask_tensor)
          )
      sub=self.augment(sub)
    else:
      sub=tio.Subject(
          t1w=tio.ScalarImage(tensor=t1_tensor),
          msk=tio.LabelMap(tensor=mask_tensor)
          )

    data_t=sub.t1w.data.float()
    mask_t=sub.msk.data.float()
    data_masked=data_t*mask_t


    return (data_masked,
            mask_t,
            torch.tensor(data.max(),dtype=torch.float32),
            torch.tensor(data.min(),dtype=torch.float32)
    )


```


```python

from torch.utils.data import DataLoader
from torch.utils.data import random_split

class T1WDataModule_mask(L.LightningDataModule):
  def __init__(self,zip_path,num_workers=0,mask_image=True,transform=True, batch_size=3):
    super().__init__()
    self.zip_path=zip_path
    self.mask_image=mask_image
    self.transform=transform
    self.num_workers=num_workers
    self.batch_size=batch_size

  def setup(self,stage=None):
    full_data=T1WDataset_mask(self.zip_path,mask_image=self.mask_image,transform=self.transform)
    val_size=int(0.15*len(full_data))
    test_size=int(0.15*len(full_data))
    train_size=len(full_data)-val_size-test_size
    self.train_dataset,self.test_dataset,self.val_dataset=random_split(full_data,[train_size,test_size,val_size])

  def train_dataloader(self):
    return DataLoader(self.train_dataset,batch_size=self.batch_size,num_workers=self.num_workers,shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.val_dataset,batch_size=self.batch_size,num_workers=self.num_workers)

  def test_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
```


```python
zip_path='/content/anat_T1w_all.zip'
autoencoder_data=T1WDataModule_mask(zip_path,batch_size=2,transform=True,num_workers=0)
```

# Model Training
After assembling the encoder, decoder, and preprocessing modules into a unified pipeline, the autoencoder was trained on T1-weighted scans from all 28 subjects. Training incorporated key callback functions to optimize performance and prevent overfitting:

* EarlyStopping — Training was terminated once validation loss ceased to improve over a defined patience window, preventing unnecessary computation and overfitting.

* ModelCheckpoint — The best-performing models were automatically saved based on validation loss, ensuring reproducibility and preserving optimal weights.

The maximum number of epochs was set to a high value to allow sufficient training iterations; however, the callback mechanisms ensured that training halted once convergence was reached. All experiments were conducted on Google Colab with an NVIDIA A100 GPU, which provided the necessary computational efficiency for handling high-dimensional 3D neuroimaging data.


```python
from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint

early_stop = EarlyStopping(
    monitor="validation_loss",
    min_delta=1e-4,
    patience=15,
    mode="min"
    )
checkpoint_callback = ModelCheckpoint(
    monitor="validation_loss",           # Same metric as early stopping
    save_top_k=1,                  # Save only the best model
    mode="min",                    # Minimize validation loss
    filename="{epoch}-{val_loss:.4f}", # Useful naming convention
    save_weights_only=False        # Save entire model (not just weights)
    )

model=BrainAutoEncoder(base=16,latent=256)
trainer=L.Trainer(
    max_epochs=300,
    log_every_n_steps=5,
    callbacks=[early_stop,checkpoint_callback],
    accelerator='gpu'
    )
trainer.fit(model,datamodule=autoencoder_data)
trainer.test(model,datamodule=autoencoder_data)
```

    INFO: GPU available: True (cuda), used: True
    INFO:lightning.pytorch.utilities.rank_zero:GPU available: True (cuda), used: True
    INFO: TPU available: False, using: 0 TPU cores
    INFO:lightning.pytorch.utilities.rank_zero:TPU available: False, using: 0 TPU cores
    INFO: HPU available: False, using: 0 HPUs
    INFO:lightning.pytorch.utilities.rank_zero:HPU available: False, using: 0 HPUs
    INFO: You are using a CUDA device ('NVIDIA A100-SXM4-40GB') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    INFO:lightning.pytorch.utilities.rank_zero:You are using a CUDA device ('NVIDIA A100-SXM4-40GB') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    INFO: LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO:lightning.pytorch.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO: 
      | Name    | Type         | Params | Mode 
    -------------------------------------------------
    0 | encoder | BrainEncoder | 1.4 M  | train
    1 | decoder | BrainDecoder | 2.0 M  | train
    -------------------------------------------------
    3.3 M     Trainable params
    0         Non-trainable params
    3.3 M     Total params
    13.318    Total estimated model params size (MB)
    41        Modules in train mode
    0         Modules in eval mode
    INFO:lightning.pytorch.callbacks.model_summary:
      | Name    | Type         | Params | Mode 
    -------------------------------------------------
    0 | encoder | BrainEncoder | 1.4 M  | train
    1 | decoder | BrainDecoder | 2.0 M  | train
    -------------------------------------------------
    3.3 M     Trainable params
    0         Non-trainable params
    3.3 M     Total params
    13.318    Total estimated model params size (MB)
    41        Modules in train mode
    0         Modules in eval mode
    


    Sanity Checking: |          | 0/? [00:00<?, ?it/s]


    /usr/local/lib/python3.12/dist-packages/lightning/pytorch/trainer/connectors/data_connector.py:433: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=11` in the `DataLoader` to improve performance.
    /usr/local/lib/python3.12/dist-packages/lightning/pytorch/trainer/connectors/data_connector.py:433: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=11` in the `DataLoader` to improve performance.
    


    Training: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]


    INFO: LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO:lightning.pytorch.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    /usr/local/lib/python3.12/dist-packages/lightning/pytorch/trainer/connectors/data_connector.py:433: The 'test_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=11` in the `DataLoader` to improve performance.
    


    Testing: |          | 0/? [00:00<?, ?it/s]



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold">        Test metric        </span>┃<span style="font-weight: bold">       DataLoader 0        </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008080; text-decoration-color: #008080">         test_loss         </span>│<span style="color: #800080; text-decoration-color: #800080">   0.002384898252785206    </span>│
└───────────────────────────┴───────────────────────────┘
</pre>






    [{'test_loss': 0.002384898252785206}]




```python
model.eval()
autoencoder_data.setup()
batch = next(iter(autoencoder_data.val_dataloader()))
x,mask,max,min= batch
x_hat = model(x)
x_hat_rescaled = (x_hat * (max - min + 1e-8) + min)
#mask=masking.compute_brain_mask(x.get_fdata())
x_hat_mask=x_hat_rescaled*mask
# Show a single axial slice
i = 0  # batch index
slice_idx = x.shape[2] // 2  # center slice
plt.subplot(1, 3, 1)
plt.imshow(x[i, 0, slice_idx].cpu(), cmap='gray')
plt.title("Original")

plt.subplot(1, 3, 2)
plt.imshow(x_hat_rescaled[i, 0, slice_idx].detach().cpu(), cmap='gray')
plt.title("Reconstruction No Mask")
plt.show()

plt.subplot(1, 3, 3)
plt.imshow(x_hat_mask[i, 0, slice_idx].detach().cpu(), cmap='gray')
plt.title("Reconstruction With Mask")
plt.show()

```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    /tmp/ipython-input-566859283.py in <cell line: 0>()
          4 x,mask,max,min= batch
          5 x_hat = model(x)
    ----> 6 x_hat_rescaled = (x_hat * (max - min + 1e-8) + min)
          7 #mask=masking.compute_brain_mask(x.get_fdata())
          8 x_hat_mask=x_hat_rescaled*mask
    

    RuntimeError: The size of tensor a (256) must match the size of tensor b (2) at non-singleton dimension 4



```python
print("x:", x.shape, "x_hat:", x_hat.shape, "mask:", mask.shape, "x_max:", max.shape, "x_min:", min.shape)
```

    x: torch.Size([2, 1, 176, 256, 256]) x_hat: torch.Size([2, 1, 176, 256, 256]) mask: torch.Size([2, 1, 176, 256, 256]) x_max: torch.Size([2]) x_min: torch.Size([2])
    

To evaluate the impact of region-of-interest (ROI) masking on reconstruction performance, an ablation experiment was conducted in which the dataset was provided to the model without any masking. The DataSet module was modified to exclude the brain extraction step, while all other preprocessing and training parameters were kept constant. This allowed for a direct comparison between masked and unmasked inputs, providing insight into whether isolating brain tissue improves reconstruction quality or if the model can effectively learn from the full volume, including surrounding non-brain regions.


```python
class T1WDataset_nomask(Dataset):
  def __init__(self,zip_path,transform=True):
    self.zip_path=zip_path
    with zipfile.ZipFile(zip_path,'r') as z:
      self.image_listdir=[f for f in z.namelist() if f.endswith('.nii') or f.endswith('.nii.gz')]
    self.transform=transform
    # Define TorchIO transform
    self.augment=tio.RandomAffine(scales=(0.95, 1.05), degrees=5)

  def __len__(self):
    return len(self.image_listdir)

  def __getitem__(self,idx):
    with zipfile.ZipFile(self.zip_path, 'r') as z:
      file_name = self.image_listdir[idx]
      with tempfile.NamedTemporaryFile(suffix='.nii.gz') as tmp:
        tmp.write(z.read(file_name))
        tmp.flush()
        img = nb.load(tmp.name)
        data = img.get_fdata().astype(np.float32)
        affine = img.affine

    scaled_data=( data - data.min())/( data.max() - data.min() + 1e-8)

    if self.transform:

      t1_tensor=torch.from_numpy(scaled_data).unsqueeze(0)
      sub=tio.Subject(t1w=tio.ScalarImage(tensor=t1_tensor))
      sub1=self.augment(sub)
      data=sub1.t1w.data.squeeze().numpy()


    return (torch.from_numpy(data).unsqueeze(0),
            torch.tensor(data.max(),dtype=torch.float32),
            torch.tensor(data.min(),dtype=torch.float32)
    )

class T1WDataModule_nomask(L.LightningDataModule):
  def __init__(self,zip_path,num_workers=0,transform=True, batch_size=3):
    super().__init__()
    self.zip_path=zip_path
    self.transform=transform
    self.num_workers=num_workers
    self.batch_size=batch_size

  def setup(self,stage=None):
    full_data=T1WDataset_nomask(self.zip_path,transform=self.transform)
    val_size=int(0.15*len(full_data))
    test_size=int(0.15*len(full_data))
    train_size=len(full_data)-val_size-test_size
    self.train_dataset,self.test_dataset,self.val_dataset=random_split(full_data,[train_size,test_size,val_size])

  def train_dataloader(self):
    return DataLoader(self.train_dataset,batch_size=self.batch_size,num_workers=self.num_workers,shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.val_dataset,batch_size=self.batch_size,num_workers=self.num_workers)

  def test_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
```


```python
zip_path='/content/anat_T1w_all.zip'
autoencoder_nomaskdata=T1WDataModule_nomask(zip_path,batch_size=2,transform=True,num_workers=0)
```


```python
from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint

early_stop = EarlyStopping(
    monitor="validation_loss",
    min_delta=1e-4,
    patience=15,
    mode="min"
    )
checkpoint_callback = ModelCheckpoint(
    monitor="validation_loss",           # Same metric as early stopping
    save_top_k=1,                  # Save only the best model
    mode="min",                    # Minimize validation loss
    filename="{epoch}-{val_loss:.4f}", # Useful naming convention
    save_weights_only=False        # Save entire model (not just weights)
    )

model_nomask=BrainAutoEncoder(base=16,latent=256)
trainer_nomask=L.Trainer(
    max_epochs=300,
    log_every_n_steps=5,
    callbacks=[early_stop,checkpoint_callback],
    accelerator='gpu'
    )
trainer_nomask.fit(model_nomask,datamodule=autoencoder_nomaskdata)
trainer_nomask.test(model_nomask,datamodule=autoencoder_nomaskdata)
```


```python
from lightning.pytorch.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor="validation_loss",min_delta=0.0001, patience=8, mode="min")
model_nomask=BrainAutoEncoder(base=16,latent=256)
trainer_nomask=L.Trainer(max_epochs=50,log_every_n_steps=5,callbacks=[early_stop],accelerator='gpu')
trainer_nomask.fit(model_nomask,datamodule=autoencoder_nomaskdata)
trainer_nomask.test(model_nomask,datamodule=autoencoder_nomaskdata)
```

    INFO: 💡 Tip: For seamless cloud uploads and versioning, try installing [litmodels](https://pypi.org/project/litmodels/) to enable LitModelCheckpoint, which syncs automatically with the Lightning model registry.
    INFO:lightning.pytorch.utilities.rank_zero:💡 Tip: For seamless cloud uploads and versioning, try installing [litmodels](https://pypi.org/project/litmodels/) to enable LitModelCheckpoint, which syncs automatically with the Lightning model registry.
    INFO: GPU available: True (cuda), used: True
    INFO:lightning.pytorch.utilities.rank_zero:GPU available: True (cuda), used: True
    INFO: TPU available: False, using: 0 TPU cores
    INFO:lightning.pytorch.utilities.rank_zero:TPU available: False, using: 0 TPU cores
    INFO: HPU available: False, using: 0 HPUs
    INFO:lightning.pytorch.utilities.rank_zero:HPU available: False, using: 0 HPUs
    INFO: LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO:lightning.pytorch.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO: 
      | Name    | Type         | Params | Mode 
    -------------------------------------------------
    0 | encoder | BrainEncoder | 1.4 M  | train
    1 | decoder | BrainDecoder | 2.0 M  | train
    -------------------------------------------------
    3.3 M     Trainable params
    0         Non-trainable params
    3.3 M     Total params
    13.318    Total estimated model params size (MB)
    41        Modules in train mode
    0         Modules in eval mode
    INFO:lightning.pytorch.callbacks.model_summary:
      | Name    | Type         | Params | Mode 
    -------------------------------------------------
    0 | encoder | BrainEncoder | 1.4 M  | train
    1 | decoder | BrainDecoder | 2.0 M  | train
    -------------------------------------------------
    3.3 M     Trainable params
    0         Non-trainable params
    3.3 M     Total params
    13.318    Total estimated model params size (MB)
    41        Modules in train mode
    0         Modules in eval mode
    


    Sanity Checking: |          | 0/? [00:00<?, ?it/s]


    /usr/local/lib/python3.11/dist-packages/lightning/pytorch/trainer/connectors/data_connector.py:425: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=11` in the `DataLoader` to improve performance.
    /usr/local/lib/python3.11/dist-packages/lightning/pytorch/trainer/connectors/data_connector.py:425: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=11` in the `DataLoader` to improve performance.
    


    Training: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]


    INFO: LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO:lightning.pytorch.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    /usr/local/lib/python3.11/dist-packages/lightning/pytorch/trainer/connectors/data_connector.py:425: The 'test_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=11` in the `DataLoader` to improve performance.
    


    Testing: |          | 0/? [00:00<?, ?it/s]



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold">        Test metric        </span>┃<span style="font-weight: bold">       DataLoader 0        </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008080; text-decoration-color: #008080">         test_loss         </span>│<span style="color: #800080; text-decoration-color: #800080">   0.005244841333478689    </span>│
└───────────────────────────┴───────────────────────────┘
</pre>






    [{'test_loss': 0.005244841333478689}]



As we can see the model does not perform as well on Unmasked data as it does on the masked dataset.


```python
model_nomask.eval()
autoencoder_nomaskdata.setup()
batch = next(iter(autoencoder_nomaskdata.val_dataloader()))
x,max,min= batch
x_hat = model_nomask(x)
#x_hat_rescaled = ((x_hat * (max - min + 1e-8)) + min)
#mask_img=masking.compute_brain_mask(nb.nifti1.Nifti1Image(x,affine=affine))
#x_hat_mask=x_hat_rescaled*mask
# Show a single axial slice
i = 0  # batch index
slice_idx = x.shape[2] // 2  # center slice
plt.subplot(1, 2, 1)
plt.imshow(x[i, 0, slice_idx].cpu(), cmap='gray')
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(x_hat[i, 0, slice_idx].detach().cpu(), cmap='gray')
plt.title("Reconstruction")
plt.show()


```


    
![png](output_36_0.png)
    


Next we will test various latent dimensional representation of our dataset to see which performs best. The test will be done for 64, 128, 256 and 384 latent dimensions. The idea is to see at which point our AutoEncoder can best represent our original dataset with lease MSE score. Intuitively, the larger the latent dimension, the more information is retained.


```python

def autoencoder_latent_test(latent_dim):
  model=BrainAutoEncoder(latent=latent_dim)
  trainer=L.Trainer(max_epochs=20,log_every_n_steps=5,accelerator='gpu')
  trainer.fit(model,datamodule=autoencoder_data)

  val_score=trainer.validate(model,datamodule=autoencoder_data)
  test_score=trainer.test(model,datamodule=autoencoder_data)
  result = {"test": test_score, "val": val_score}
  return model,result
```


```python
latent_tests=[64,128,256,384]
model_dict={}
for l in latent_tests:
  model_ld,result_ld=autoencoder_latent_test(l)
  model_dict[l]={"model": model_ld, "result": result_ld}
```

    INFO: 💡 Tip: For seamless cloud uploads and versioning, try installing [litmodels](https://pypi.org/project/litmodels/) to enable LitModelCheckpoint, which syncs automatically with the Lightning model registry.
    INFO:lightning.pytorch.utilities.rank_zero:💡 Tip: For seamless cloud uploads and versioning, try installing [litmodels](https://pypi.org/project/litmodels/) to enable LitModelCheckpoint, which syncs automatically with the Lightning model registry.
    INFO: GPU available: True (cuda), used: True
    INFO:lightning.pytorch.utilities.rank_zero:GPU available: True (cuda), used: True
    INFO: TPU available: False, using: 0 TPU cores
    INFO:lightning.pytorch.utilities.rank_zero:TPU available: False, using: 0 TPU cores
    INFO: HPU available: False, using: 0 HPUs
    INFO:lightning.pytorch.utilities.rank_zero:HPU available: False, using: 0 HPUs
    INFO: LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO:lightning.pytorch.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO: 
      | Name    | Type         | Params | Mode 
    -------------------------------------------------
    0 | encoder | BrainEncoder | 6.2 M  | train
    1 | decoder | BrainDecoder | 15.5 M | train
    -------------------------------------------------
    21.7 M    Trainable params
    0         Non-trainable params
    21.7 M    Total params
    86.950    Total estimated model params size (MB)
    41        Modules in train mode
    0         Modules in eval mode
    INFO:lightning.pytorch.callbacks.model_summary:
      | Name    | Type         | Params | Mode 
    -------------------------------------------------
    0 | encoder | BrainEncoder | 6.2 M  | train
    1 | decoder | BrainDecoder | 15.5 M | train
    -------------------------------------------------
    21.7 M    Trainable params
    0         Non-trainable params
    21.7 M    Total params
    86.950    Total estimated model params size (MB)
    41        Modules in train mode
    0         Modules in eval mode
    


    Sanity Checking: |          | 0/? [00:00<?, ?it/s]


    /usr/local/lib/python3.12/dist-packages/lightning/pytorch/trainer/connectors/data_connector.py:433: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=11` in the `DataLoader` to improve performance.
    /usr/local/lib/python3.12/dist-packages/lightning/pytorch/trainer/connectors/data_connector.py:433: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=11` in the `DataLoader` to improve performance.
    


    Training: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]


    INFO: `Trainer.fit` stopped: `max_epochs=20` reached.
    INFO:lightning.pytorch.utilities.rank_zero:`Trainer.fit` stopped: `max_epochs=20` reached.
    INFO: LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO:lightning.pytorch.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    /usr/local/lib/python3.12/dist-packages/lightning/pytorch/trainer/connectors/data_connector.py:433: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=11` in the `DataLoader` to improve performance.
    


    Validation: |          | 0/? [00:00<?, ?it/s]



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold">      Validate metric      </span>┃<span style="font-weight: bold">       DataLoader 0        </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008080; text-decoration-color: #008080">      validation_loss      </span>│<span style="color: #800080; text-decoration-color: #800080">    0.00784835685044527    </span>│
└───────────────────────────┴───────────────────────────┘
</pre>



    INFO: LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO:lightning.pytorch.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    /usr/local/lib/python3.12/dist-packages/lightning/pytorch/trainer/connectors/data_connector.py:433: The 'test_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=11` in the `DataLoader` to improve performance.
    


    Testing: |          | 0/? [00:00<?, ?it/s]



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold">        Test metric        </span>┃<span style="font-weight: bold">       DataLoader 0        </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008080; text-decoration-color: #008080">         test_loss         </span>│<span style="color: #800080; text-decoration-color: #800080">   0.008489873260259628    </span>│
└───────────────────────────┴───────────────────────────┘
</pre>



    INFO: 💡 Tip: For seamless cloud uploads and versioning, try installing [litmodels](https://pypi.org/project/litmodels/) to enable LitModelCheckpoint, which syncs automatically with the Lightning model registry.
    INFO:lightning.pytorch.utilities.rank_zero:💡 Tip: For seamless cloud uploads and versioning, try installing [litmodels](https://pypi.org/project/litmodels/) to enable LitModelCheckpoint, which syncs automatically with the Lightning model registry.
    INFO: GPU available: True (cuda), used: True
    INFO:lightning.pytorch.utilities.rank_zero:GPU available: True (cuda), used: True
    INFO: TPU available: False, using: 0 TPU cores
    INFO:lightning.pytorch.utilities.rank_zero:TPU available: False, using: 0 TPU cores
    INFO: HPU available: False, using: 0 HPUs
    INFO:lightning.pytorch.utilities.rank_zero:HPU available: False, using: 0 HPUs
    INFO: LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO:lightning.pytorch.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO: 
      | Name    | Type         | Params | Mode 
    -------------------------------------------------
    0 | encoder | BrainEncoder | 7.3 M  | train
    1 | decoder | BrainDecoder | 16.5 M | train
    -------------------------------------------------
    23.8 M    Trainable params
    0         Non-trainable params
    23.8 M    Total params
    95.339    Total estimated model params size (MB)
    41        Modules in train mode
    0         Modules in eval mode
    INFO:lightning.pytorch.callbacks.model_summary:
      | Name    | Type         | Params | Mode 
    -------------------------------------------------
    0 | encoder | BrainEncoder | 7.3 M  | train
    1 | decoder | BrainDecoder | 16.5 M | train
    -------------------------------------------------
    23.8 M    Trainable params
    0         Non-trainable params
    23.8 M    Total params
    95.339    Total estimated model params size (MB)
    41        Modules in train mode
    0         Modules in eval mode
    


    Sanity Checking: |          | 0/? [00:00<?, ?it/s]


    /usr/local/lib/python3.12/dist-packages/lightning/pytorch/trainer/connectors/data_connector.py:433: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=11` in the `DataLoader` to improve performance.
    /usr/local/lib/python3.12/dist-packages/lightning/pytorch/trainer/connectors/data_connector.py:433: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=11` in the `DataLoader` to improve performance.
    


    Training: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]


    INFO: `Trainer.fit` stopped: `max_epochs=20` reached.
    INFO:lightning.pytorch.utilities.rank_zero:`Trainer.fit` stopped: `max_epochs=20` reached.
    INFO: LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO:lightning.pytorch.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    /usr/local/lib/python3.12/dist-packages/lightning/pytorch/trainer/connectors/data_connector.py:433: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=11` in the `DataLoader` to improve performance.
    


    Validation: |          | 0/? [00:00<?, ?it/s]



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold">      Validate metric      </span>┃<span style="font-weight: bold">       DataLoader 0        </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008080; text-decoration-color: #008080">      validation_loss      </span>│<span style="color: #800080; text-decoration-color: #800080">   0.007278837263584137    </span>│
└───────────────────────────┴───────────────────────────┘
</pre>



    INFO: LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO:lightning.pytorch.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    /usr/local/lib/python3.12/dist-packages/lightning/pytorch/trainer/connectors/data_connector.py:433: The 'test_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=11` in the `DataLoader` to improve performance.
    


    Testing: |          | 0/? [00:00<?, ?it/s]



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold">        Test metric        </span>┃<span style="font-weight: bold">       DataLoader 0        </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008080; text-decoration-color: #008080">         test_loss         </span>│<span style="color: #800080; text-decoration-color: #800080">   0.007227159105241299    </span>│
└───────────────────────────┴───────────────────────────┘
</pre>



    INFO: 💡 Tip: For seamless cloud uploads and versioning, try installing [litmodels](https://pypi.org/project/litmodels/) to enable LitModelCheckpoint, which syncs automatically with the Lightning model registry.
    INFO:lightning.pytorch.utilities.rank_zero:💡 Tip: For seamless cloud uploads and versioning, try installing [litmodels](https://pypi.org/project/litmodels/) to enable LitModelCheckpoint, which syncs automatically with the Lightning model registry.
    INFO: GPU available: True (cuda), used: True
    INFO:lightning.pytorch.utilities.rank_zero:GPU available: True (cuda), used: True
    INFO: TPU available: False, using: 0 TPU cores
    INFO:lightning.pytorch.utilities.rank_zero:TPU available: False, using: 0 TPU cores
    INFO: HPU available: False, using: 0 HPUs
    INFO:lightning.pytorch.utilities.rank_zero:HPU available: False, using: 0 HPUs
    INFO: LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO:lightning.pytorch.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO: 
      | Name    | Type         | Params | Mode 
    -------------------------------------------------
    0 | encoder | BrainEncoder | 9.4 M  | train
    1 | decoder | BrainDecoder | 18.6 M | train
    -------------------------------------------------
    28.0 M    Trainable params
    0         Non-trainable params
    28.0 M    Total params
    112.116   Total estimated model params size (MB)
    41        Modules in train mode
    0         Modules in eval mode
    INFO:lightning.pytorch.callbacks.model_summary:
      | Name    | Type         | Params | Mode 
    -------------------------------------------------
    0 | encoder | BrainEncoder | 9.4 M  | train
    1 | decoder | BrainDecoder | 18.6 M | train
    -------------------------------------------------
    28.0 M    Trainable params
    0         Non-trainable params
    28.0 M    Total params
    112.116   Total estimated model params size (MB)
    41        Modules in train mode
    0         Modules in eval mode
    


    Sanity Checking: |          | 0/? [00:00<?, ?it/s]


    /usr/local/lib/python3.12/dist-packages/lightning/pytorch/trainer/connectors/data_connector.py:433: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=11` in the `DataLoader` to improve performance.
    /usr/local/lib/python3.12/dist-packages/lightning/pytorch/trainer/connectors/data_connector.py:433: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=11` in the `DataLoader` to improve performance.
    


    Training: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]


    INFO: `Trainer.fit` stopped: `max_epochs=20` reached.
    INFO:lightning.pytorch.utilities.rank_zero:`Trainer.fit` stopped: `max_epochs=20` reached.
    INFO: LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO:lightning.pytorch.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    /usr/local/lib/python3.12/dist-packages/lightning/pytorch/trainer/connectors/data_connector.py:433: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=11` in the `DataLoader` to improve performance.
    


    Validation: |          | 0/? [00:00<?, ?it/s]



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold">      Validate metric      </span>┃<span style="font-weight: bold">       DataLoader 0        </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008080; text-decoration-color: #008080">      validation_loss      </span>│<span style="color: #800080; text-decoration-color: #800080">   0.007229698821902275    </span>│
└───────────────────────────┴───────────────────────────┘
</pre>



    INFO: LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO:lightning.pytorch.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    /usr/local/lib/python3.12/dist-packages/lightning/pytorch/trainer/connectors/data_connector.py:433: The 'test_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=11` in the `DataLoader` to improve performance.
    


    Testing: |          | 0/? [00:00<?, ?it/s]



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold">        Test metric        </span>┃<span style="font-weight: bold">       DataLoader 0        </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008080; text-decoration-color: #008080">         test_loss         </span>│<span style="color: #800080; text-decoration-color: #800080">   0.007523054722696543    </span>│
└───────────────────────────┴───────────────────────────┘
</pre>



    INFO: 💡 Tip: For seamless cloud uploads and versioning, try installing [litmodels](https://pypi.org/project/litmodels/) to enable LitModelCheckpoint, which syncs automatically with the Lightning model registry.
    INFO:lightning.pytorch.utilities.rank_zero:💡 Tip: For seamless cloud uploads and versioning, try installing [litmodels](https://pypi.org/project/litmodels/) to enable LitModelCheckpoint, which syncs automatically with the Lightning model registry.
    INFO: GPU available: True (cuda), used: True
    INFO:lightning.pytorch.utilities.rank_zero:GPU available: True (cuda), used: True
    INFO: TPU available: False, using: 0 TPU cores
    INFO:lightning.pytorch.utilities.rank_zero:TPU available: False, using: 0 TPU cores
    INFO: HPU available: False, using: 0 HPUs
    INFO:lightning.pytorch.utilities.rank_zero:HPU available: False, using: 0 HPUs
    INFO: LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO:lightning.pytorch.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO: 
      | Name    | Type         | Params | Mode 
    -------------------------------------------------
    0 | encoder | BrainEncoder | 11.5 M | train
    1 | decoder | BrainDecoder | 20.7 M | train
    -------------------------------------------------
    32.2 M    Trainable params
    0         Non-trainable params
    32.2 M    Total params
    128.894   Total estimated model params size (MB)
    41        Modules in train mode
    0         Modules in eval mode
    INFO:lightning.pytorch.callbacks.model_summary:
      | Name    | Type         | Params | Mode 
    -------------------------------------------------
    0 | encoder | BrainEncoder | 11.5 M | train
    1 | decoder | BrainDecoder | 20.7 M | train
    -------------------------------------------------
    32.2 M    Trainable params
    0         Non-trainable params
    32.2 M    Total params
    128.894   Total estimated model params size (MB)
    41        Modules in train mode
    0         Modules in eval mode
    


    Sanity Checking: |          | 0/? [00:00<?, ?it/s]


    /usr/local/lib/python3.12/dist-packages/lightning/pytorch/trainer/connectors/data_connector.py:433: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=11` in the `DataLoader` to improve performance.
    /usr/local/lib/python3.12/dist-packages/lightning/pytorch/trainer/connectors/data_connector.py:433: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=11` in the `DataLoader` to improve performance.
    


    Training: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]



    Validation: |          | 0/? [00:00<?, ?it/s]


    INFO: `Trainer.fit` stopped: `max_epochs=20` reached.
    INFO:lightning.pytorch.utilities.rank_zero:`Trainer.fit` stopped: `max_epochs=20` reached.
    INFO: LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO:lightning.pytorch.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    /usr/local/lib/python3.12/dist-packages/lightning/pytorch/trainer/connectors/data_connector.py:433: The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=11` in the `DataLoader` to improve performance.
    


    Validation: |          | 0/? [00:00<?, ?it/s]



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold">      Validate metric      </span>┃<span style="font-weight: bold">       DataLoader 0        </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008080; text-decoration-color: #008080">      validation_loss      </span>│<span style="color: #800080; text-decoration-color: #800080">   0.028235681354999542    </span>│
└───────────────────────────┴───────────────────────────┘
</pre>



    INFO: LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    INFO:lightning.pytorch.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    /usr/local/lib/python3.12/dist-packages/lightning/pytorch/trainer/connectors/data_connector.py:433: The 'test_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=11` in the `DataLoader` to improve performance.
    


    Testing: |          | 0/? [00:00<?, ?it/s]



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold">        Test metric        </span>┃<span style="font-weight: bold">       DataLoader 0        </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008080; text-decoration-color: #008080">         test_loss         </span>│<span style="color: #800080; text-decoration-color: #800080">    0.02636147290468216    </span>│
└───────────────────────────┴───────────────────────────┘
</pre>




```python
latent_dims = sorted([k for k in model_dict])
val_scores = [model_dict[k]["result"]["val"][0]['validation_loss'] for k in latent_dims]

fig = plt.figure(figsize=(6,4))
plt.plot(latent_tests, val_scores, '--', color="#000", marker="*", markeredgecolor="#000", markerfacecolor="y", markersize=16)
plt.xscale("log")
plt.xticks(latent_dims, labels=latent_dims)
plt.title("Reconstruction error over latent dimensionality", fontsize=14)
plt.xlabel("Latent dimensionality")
plt.ylabel("Reconstruction error")
plt.minorticks_off()
plt.ylim(0,0.5)
plt.show()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /tmp/ipython-input-2180634954.py in <cell line: 0>()
    ----> 1 latent_dims = sorted([k for k in model_dict])
          2 val_scores = [model_dict[k]["result"]["val"][0]['validation_loss'] for k in latent_dims]
          3 
          4 fig = plt.figure(figsize=(6,4))
          5 plt.plot(latent_tests, val_scores, '--', color="#000", marker="*", markeredgecolor="#000", markerfacecolor="y", markersize=16)
    

    NameError: name 'model_dict' is not defined



```python
model=model_dict[256]['model']
model.eval()
batch = next(iter(autoencoder_data.val_dataloader()))
x, _ = batch
x_hat = model(x)

# Show a single axial slice
i = 0  # batch index
slice_idx = x.shape[2] // 2  # center slice
plt.subplot(1, 2, 1)
plt.imshow(x[i,0, slice_idx].cpu(), cmap='gray')
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(x_hat[i, 0, slice_idx].detach().cpu(), cmap='gray')
plt.title("Reconstruction")
plt.show()
```


    
![png](output_41_0.png)
    


# Conclusion
This study demonstrates the feasibility of applying autoencoders to high-dimensional fMRI T1-weighted brain images (176×256×256 voxels). While the model effectively reduced dimensionality, it failed to achieve accurate image reconstruction, as evidenced by high mean squared error values. These findings suggest that substantial refinement—through optimized architectures, fine-tuning, and more robust data workflows—is required to enhance reconstruction quality.

Future directions include leveraging parallelized computation to accelerate training, incorporating additional spatial features to capture brain connectivity more effectively, and expanding datasets to mitigate current limitations. A well-optimized autoencoder capable of reliably encoding and decoding complex 3D neuroimaging data holds promise for advancing brain research and clinical practice. Beyond enabling improved image compression and reconstruction, such approaches may also support cost-effective neuroimaging in regions with limited access to high-resolution MRI technology, thereby broadening the global availability of diagnostic imaging.


```python

```
