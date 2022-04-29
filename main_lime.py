from torchsat.models.classification.resnet import resnet50
import torchsat.transforms.transforms_cls as T_cls
from torchsat.datasets.folder import DatasetFolder
from skimage.segmentation import mark_boundaries
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchsat.datasets as datasets
import matplotlib.pyplot as plt
from lime import lime_image
from pathlib import Path
from PIL import Image
import numpy as np
import tifffile
import torch


def get_preprocess_transform(is4channel):
    if is4channel:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2, 0.2])
    else:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])

    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    return transf


def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i.astype(np.double)) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)
    
    logits = model(batch)
    probs = torch.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


def imgShow(x: np.ndarray, title=None, ax=None):
    img = Image.fromarray((x >> 4).astype(np.uint8))

    if ax == None:
        if title is not None:
            plt.title(title)
        plt.imshow(img)

    if title is not None:
        ax.set_title(title)
    ax.imshow(img)
    return img


RGB_or_IR = 'IR' # 'RGB'

if RGB_or_IR == 'RGB':
    datasetFolder = './RGB Only Dataset'
    model_path = 'model_3.pth'
    in_channels = 3
    model: torch.nn.Module = resnet50(in_channels=in_channels, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    model.type(torch.double)

elif RGB_or_IR == 'IR':
    datasetFolder = './RGB+IR Dataset'
    model_path = 'model_4.pth'
    in_channels = 4
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    model.type(torch.double)


preprocess_transform = get_preprocess_transform(RGB_or_IR == 'IR')


riverImage  = tifffile.imread(f'{datasetFolder}/val/River/River2230.tif')
forestImage = tifffile.imread(f'{datasetFolder}/val/Forest/Forest2710.tif')


for ttl, img in [('River', riverImage), ('Forest', forestImage)]:

    f, axarr = plt.subplots(1, 6, sharey=True, figsize=(16,4))
    
    tifffile.imshow(img[:, :, :3], title=f"{ttl} Image", figure=f, subplot=int(f"161"))

    explainer = lime_image.LimeImageExplainer(random_state=20)
    explanations = explainer.explain_instance(img,
                                                batch_predict,
                                                top_labels=2, 
                                                hide_color=0, 
                                                num_samples=1000, # 1000,
                                                # batch_size=64,
                                                channel_wise=True)


    for j, explanation in enumerate(explanations):
        temp, mask = explanation.get_image_and_mask(label=explanation.top_labels[0], positive_only=False, num_features=2, hide_rest=False)
        img_boundry1 = mark_boundaries(temp[:, :, :3]*16, mask) # , color=(1, 1, 1, 0))

        titles = ['All', 'Red', 'Green', 'Blue', 'NIR']
        tifffile.imshow(img_boundry1, title=titles[j], figure=f, subplot=int(f"16{j+2}"))

plt.show()
