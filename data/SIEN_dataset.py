import cv2.dnn
import torch.utils.data as data
import torch
import torchvision
from torchvision.transforms import Compose, ToTensor
import os
import random
# from PIL import Image,ImageOps
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.nn.functional as F
import cv2
import numpy as np

def Canny(self, index):
    from_path = self.source_paths[index]
    from_im = cv2.imread(from_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
    to_path = self.target_paths[index]
    to_im = cv2.imread(to_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0

    crop_size = 256
    img_h, img_w, _ = from_im.shape
    crop_h = np.random.randint(0, img_h - crop_size)
    crop_w = np.random.randint(0, img_w - crop_size)
    input = from_im[crop_h:crop_h + crop_size, crop_w:crop_w + crop_size]
    gt = to_im[crop_h:crop_h + crop_size, crop_w:crop_w + crop_size]
    #####

    inputs = input[:, :, [2, 1, 0]]
    target = gt[:, :, [2, 1, 0]]

    inputs = torch.from_numpy(np.ascontiguousarray(np.transpose(inputs, (2, 0, 1)))).float()
    targets = torch.from_numpy(np.ascontiguousarray(np.transpose(target, (2, 0, 1)))).float()

    return inputs, targets
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def get_patch(input, target, patch_size, scale = 1, ix=-1, iy=-1):
    ih, iw, channels = input.shape
    # (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, ih - ip + 1)
    if iy == -1:
        iy = random.randrange(0, iw - ip + 1)

    # (tx, ty) = (scale * ix, scale * iy)


    input = input[ix:ix + ip, iy:iy + ip, :]  # [:, ty:ty + tp, tx:tx + tp]
    target = target[ix:ix + ip, iy:iy + ip, :]  # [:, iy:iy + ip, ix:ix + ip]


    return  input, target


def augment(inputs, target, hflip, rot):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot180 = rot and random.random() < 0.5

    def _augment(inputs, target):
        if hflip:
            inputs = inputs[:, ::-1, :]

            target = target[:, ::-1, :]
        if vflip:
            inputs = inputs[::-1, :, :]

            target = target[::-1, :, :]
        if rot180:
            inputs = cv2.rotate(inputs, cv2.ROTATE_180)

            target = cv2.rotate(target, cv2.ROTATE_180)
        return inputs,target

    inputs, target = _augment(inputs,target)

    return inputs, target



def get_image_hdr(img):
    img = cv2.imread(img,cv2.IMREAD_UNCHANGED)
    # img = np.round(img/(2**6)).astype(np.uint16)
    img = img.astype(np.float32)/65535.0

    w, h = img.shape[0], img.shape[1]
    while w % 4 != 0:
        w += 1
    while h % 4 != 0:
        h += 1
    img = cv2.resize(img, (h, w))

    return img

def get_image_ldr(img1,img2):
    input = cv2.imread(img1,cv2.IMREAD_UNCHANGED).astype(np.float32)/255.0
    gt = cv2.imread(img2, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0

    # input = cv2.resize(input, dsize=(960, 720))
    # gt = cv2.resize(gt, dsize=(960, 720))


    crop_size = 256
    img_h, img_w,_ = input.shape
    crop_h = np.random.randint(0, img_h - crop_size)
    crop_w = np.random.randint(0, img_w - crop_size)
    inputt = input[crop_h:crop_h + crop_size, crop_w:crop_w + crop_size]
    gtt = gt[crop_h:crop_h + crop_size, crop_w:crop_w + crop_size]
    # inputt=input
    # gtt=gt
    return inputt,gtt,input,gt


def load_image_train2(group):
    # images = [get_image(img) for img in group]
    # inputs = images[:-1]
    # target = images[-1]
    inputs,target,nocropin,nocropgt = get_image_ldr(group[0],group[2])
    # edge = get_image_ldr()
    # target = get_image_ldr(group[2])
    # if black_edges_crop == True:
    #     inputs = [indiInput[70:470, :, :] for indiInput in inputs]
    #     target = target[280:1880, :, :]
    #     return inputs, target
    # else:
    return inputs, target,nocropin,nocropgt


def transform():
    return Compose([
        ToTensor(),
    ])

def BGR2RGB_toTensor(inputs,target,nocropin,nocropgt):
    inputs = inputs[:, :, [2, 1, 0]]
    # edge = edge[:, :, 0]
    target = target[:, :, [2, 1, 0]]

    nocropin=nocropin[:, :, [2, 1, 0]]
    nocropgt = nocropgt[:, :, [2, 1, 0]]

    inputs = torch.from_numpy(np.ascontiguousarray(np.transpose(inputs, (2, 0, 1)))).float()
    # edge = torch.from_numpy(np.ascontiguousarray(edge)).float()
    target = torch.from_numpy(np.ascontiguousarray(np.transpose(target, (2, 0, 1)))).float()

    nocropin = torch.from_numpy(np.ascontiguousarray(np.transpose(nocropin, (2, 0, 1)))).float()
    nocropgt = torch.from_numpy(np.ascontiguousarray(np.transpose(nocropgt, (2, 0, 1)))).float()
    return inputs,target,nocropin,nocropgt
class DatasetFromFolder(data.Dataset):
    """
    For test dataset, specify
    `group_file` parameter to target TXT file
    data_augmentation = None
    black_edge_crop = None
    flip = None
    rot = None
    """
    def __init__(self, upscale_factor, data_augmentation, group_file, patch_size, black_edges_crop, hflip, rot, transform=transform()):
        super(DatasetFromFolder, self).__init__()
        groups = [line.rstrip() for line in open(os.path.join(group_file))]
        # assert groups[0].startswith('/'), 'Paths from file_list must be absolute paths!'
        self.image_filenames = [group.split('|') for group in groups]
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.patch_size = patch_size
        self.black_edges_crop = black_edges_crop
        self.hflip = hflip
        self.rot = rot

    def __getitem__(self, index):

        inputs, target,nocropin,nocropgt = load_image_train2(self.image_filenames[index])

        if self.transform:
            inputs,target,nocropin,nocropgt= BGR2RGB_toTensor(inputs,target,nocropin,nocropgt)

        # transform = torchvision.transforms.Resize((256, 256))
        # nocropgt = transform(nocropgt)
        # nocropin = transform(nocropin)
        # inputs = transform(inputs)
        # target=transform(target)

        return {'nocropin':nocropin,'nocropgt':nocropgt,'LQ': inputs, 'GT': target, 'LQ_path': self.image_filenames[index][0], 'GT_path': self.image_filenames[index][1]}

    def __len__(self):
        return len(self.image_filenames)


