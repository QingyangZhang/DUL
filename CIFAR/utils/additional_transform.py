import numpy as np
import random
from PIL import Image,ImageFilter

class AddSaltPepperNoise(object):

    def __init__(self, density=0.05,p=0.5):
        self.density = density
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:  
            img = np.array(img)  
            h, w, c = img.shape
            Nd = self.density
            Sd = 1 - Nd
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd]) 
            mask = np.repeat(mask, c, axis=2)  
            img[mask == 0] = 0  
            img[mask == 1] = 255  
            img = Image.fromarray(img.astype('uint8')).convert('RGB')  
            return img
        else:
            return img

class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):

        img = np.array(img)
        h, w, c = img.shape
        np.random.seed(0)
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255                     
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img

class Addblur(object):

    def __init__(self, p=0.5,blur="normal"):
        #         self.density = density
        self.p = p
        self.blur= blur

    def __call__(self, img):
        if random.uniform(0, 1) < self.p: 
            if self.blur== "normal":
                img = img.filter(ImageFilter.BLUR)
                return img
            if self.blur== "Gaussian":
                img = img.filter(ImageFilter.GaussianBlur)
                return img
            if self.blur== "mean":
                img = img.filter(ImageFilter.BoxBlur(1))
                return img

        else:
            return img