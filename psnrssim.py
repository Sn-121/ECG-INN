import os

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

import warnings
warnings.filterwarnings('ignore')

SSIM_list = []
PSNR_LIST = []
# clear_img_path = "D:/BaiduNetdiskDownload/Huawei_Nikon/hua/test/Result"  
# hazy_img_path = "D:/BaiduNetdiskDownload/Huawei_Nikon/hua/test/high" 
clear_img_path = "D:/Github-Code/MainNet/LOL/high"  
hazy_img_path = "D:/Github-Code/MainNet/LOL/low"
# clear_img_path = "result/"  
# hazy_img_path = "test/target/"  
clear_img_names = os.listdir(clear_img_path) 
hazy_img_names = []

for name in clear_img_names:
    hazy_img_names.append(name[:] )

for i in range(len(clear_img_names)):
    clear_img = cv2.imread(os.path.join(clear_img_path, clear_img_names[i]))
    hazy_img = cv2.imread(os.path.join(hazy_img_path, hazy_img_names[i]))
    if clear_img.shape[0] != hazy_img.shape[0] or clear_img.shape[1] != hazy_img.shape[1]:
        pil_img = Image.fromarray(hazy_img)
        pil_img = pil_img.resize((clear_img.shape[1], clear_img.shape[0])) 
        hazy_img = np.array(pil_img)

    
    PSNR = peak_signal_noise_ratio(clear_img, hazy_img)
    print(i+1, 'PSNR: ', PSNR)
    PSNR_LIST.append(PSNR)

    SSIM = structural_similarity(clear_img, hazy_img, multichannel=True)
    print(i+1, 'SSIM: ', SSIM)
    SSIM_list.append(SSIM)

print("average SSIM:{:.2f}".format(sum(SSIM_list)/ len(SSIM_list)))
print("average PSNR:{:.2f}".format(sum(PSNR_LIST)/ len(PSNR_LIST)))

