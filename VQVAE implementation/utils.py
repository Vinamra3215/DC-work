import numpy as np

def normalize(img):
    img = img - img.min()
    return img / (img.max() + 1e-8)
