import os
from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image
from PIL import ImageFilter
from tqdm import tqdm


w, h = 256, 256
rad = 10
input_dir = "datasets/facades/"
output_dir = "datasets/facades2/"

def triplet(img):
	img2 = Image.new('RGB', (3*w, h))
	A, B = img.crop((0, 0, w, h)), img.crop((w, 0, 2*w, h))
	C = A.copy()
	C = C.filter(ImageFilter.GaussianBlur(radius=rad))
	img2.paste(A, (0, 0))
	img2.paste(B, (w, 0))
	img2.paste(C, (2*w, 0))
	return img2


for sub_dir in ["train", "test", "val"]:
    target_dir = join(input_dir, sub_dir)
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    files = [f for f in listdir(join(input_dir, sub_dir)) if isfile(join(target_dir, f))]
    for f in tqdm(files):
        img = Image.open('%s/%s/%s'%(input_dir, sub_dir, f))
        img2 = triplet(img)
        img2.save(join(target_dir, f))
    
