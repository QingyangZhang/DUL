import numpy as np
import torch
import os
import pickle
import time


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

    

class ImageNet(torch.utils.data.Dataset):

    def __init__(self, transform=None, img_size=64, get_idx=False, use_rate=1, max_class=1000):

        self.S = np.zeros(11, dtype=np.int32)
        self.img_size = img_size
        self.labels = []
        self.get_idx=get_idx
        self.max_class=max_class

        for idx in range(1, 11):
           
            data_file = os.path.join('/apdcephfs/private_yangqyzhang/ImageNet-RC', 'train_data_batch_{}'.format(idx))
            d = unpickle(data_file)
            y = d['labels']
            y = [i-1 for i in y]
            self.labels.extend(y)
            self.S[idx] = self.S[idx-1] + len(y)

        self.labels = np.array(self.labels)

        ## 控制 diversity
        # 找到 生成映射：y_index-> index
        self.class_index2all_index = np.where(self.labels < max_class)[0]
        self.N = int(len(self.class_index2all_index)*use_rate)

        self.curr_batch = -1

        self.offset = 0     # offset index
        self.transform = transform

    def load_image_batch(self, batch_index):

       
        data_file = os.path.join('/apdcephfs/private_yangqyzhang/ImageNet-RC', 'train_data_batch_{}'.format(batch_index))
        d = unpickle(data_file)
        x = d['data']
        
        img_size = self.img_size
        img_size2 = img_size * img_size
        x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
        x = x.reshape(( x.shape[0], img_size, img_size, 3))
        self.batch_images = x
        self.curr_batch = batch_index

    def get_batch_index(self, index):
        j = 1
        while index >= self.S[j]:
            j += 1
        return j

    def load_image(self, index):
        batch_index = self.get_batch_index(index)
        if self.curr_batch != batch_index:
            self.load_image_batch(batch_index)
        
        return self.batch_images[index-self.S[batch_index-1]]

    def __getitem__(self, index):
        index = (index + self.offset) % self.N

        index=self.class_index2all_index[index] # 映射到特定类别的data下
        img = self.load_image(index)
        if self.transform is not None:
            img = self.transform(img)

        if self.get_idx:
            return img,self.labels[index],index
        else:
            return img, self.labels[index]

    def __len__(self):
        return self.N
