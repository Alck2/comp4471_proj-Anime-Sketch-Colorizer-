import cv2
import numpy as np
import torch
import torchvision
import opencv_transforms.functional as FF
from torchvision import datasets
from PIL import Image
import random

def color_cluster(img, nclusters=9):
    """
    Apply K-means clustering to the input image

    Args:
        img: Numpy array which has shape of (H, W, C)
        nclusters: # of clusters (default = 9)

    Returns:
        color_palette: list of 3D numpy arrays which have same shape of that of input image
        e.g. If input image has shape of (256, 256, 3) and nclusters is 4, the return color_palette is [color1, color2, color3, color4]
            and each component is (256, 256, 3) numpy array.
            
    Note:
        K-means clustering algorithm is quite computaionally intensive.
        Thus, before extracting dominant colors, the input images are resized to x0.25 size.
    """
#print("")
    #print("color hint")
    img_size = img.shape
    temp = img.copy()
    test_hint = img.copy()
    temp = temp[:,:512,:]
    for i in range(30):
        randx = random.randint(0,450)
        randy = random.randint(0,450)
        temp[randx:randx+25,randy:randy+25] = 255
    temp = cv2.blur(temp,(50,50)) 
    #plt.imshow(temp)

    color_palette = []
    
    #for i in range(0, nclusters):
    #    dominant_color = np.zeros(img_size, dtype='uint8')
    #    #dominant_color[:,:,:] = centers[i]
    #    dominant_color[:,:,:] = 0
    #    color_palette.append(dominant_color)
    #color_palette.append(temp)
    color_palette.append(temp)
    return color_palette

class PairImageFolder(datasets.ImageFolder):   
    """
    A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    
    This class works properly for paired image in form of [sketch, color_image]

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
        sketch_net: The network to convert color image to sketch image
        ncluster: Number of clusters when extracting color palette.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    
     Getitem:
        img_edge: Edge image
        img: Color Image
        color_palette: Extracted color paltette
    """
    def __init__(self, root, transform, sketch_net, ncluster):
        super(PairImageFolder, self).__init__(root, transform)
        self.ncluster = ncluster
        self.sketch_net = sketch_net
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = self.loader(path)
        img = np.asarray(img)
        img = img[:, 0:512, :]
        img = self.transform(img)
        color_palette = color_cluster(img, nclusters=self.ncluster)
        img = self.make_tensor(img)
        
        with torch.no_grad():
            img_edge = self.sketch_net(img.unsqueeze(0).to(self.device)).squeeze().permute(1,2,0).cpu().numpy()
            img_edge = FF.to_grayscale(img_edge, num_output_channels=3)
            img_edge = FF.to_tensor(img_edge)
            
        for i in range(0, len(color_palette)):
            color = color_palette[i]
            color_palette[i] = self.make_tensor(color)

        return img_edge, img, color_palette
    
    def make_tensor(self, img):
        img = FF.to_tensor(img)
        img = FF.normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return img
    
class GetImageFolder(datasets.ImageFolder):   
    """
    A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
        sketch_net: The network to convert color image to sketch image
        ncluster: Number of clusters when extracting color palette.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    
     Getitem:
        img_edge: Edge image
        img: Color Image
        color_palette: Extracted color paltette
    """
    def __init__(self, root, transform, sketch_net, ncluster):
        super(GetImageFolder, self).__init__(root, transform)
        self.ncluster = ncluster
        self.sketch_net = sketch_net
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = self.loader(path)
        img = np.asarray(img)
        img = self.transform(img)
        color_palette = color_cluster(img, nclusters=self.ncluster)
        img = self.make_tensor(img)
        
        with torch.no_grad():
            img_edge = self.sketch_net(img.unsqueeze(0).to(self.device)).squeeze().permute(1,2,0).cpu().numpy()
            img_edge = FF.to_grayscale(img_edge, num_output_channels=3)
            img_edge = FF.to_tensor(img_edge)
            
        for i in range(0, len(color_palette)):
            color = color_palette[i]
            color_palette[i] = self.make_tensor(color)

        return img_edge, img, color_palette
    
    def make_tensor(self, img):
        img = FF.to_tensor(img)
        img = FF.normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return img