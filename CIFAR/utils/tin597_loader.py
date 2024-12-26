import os
from PIL import Image
from torch.utils.data import Dataset

class TIN597(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.classes = sorted(os.listdir(root))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.file_list = self._build_file_list()
        self.transform = transform

    def _build_file_list(self):
        file_list = []
        for cls in self.classes:
            class_path = os.path.join(self.root, cls, 'images')
            for filename in os.listdir(class_path):
                if filename.endswith(('.jpg', '.JPEG', '.png')):
                    file_list.append((os.path.join(class_path, filename), self.class_to_idx[cls]))
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, label = self.file_list[idx]
        img = Image.open(img_path).convert('RGB')  # Ensure the image is in RGB format

        if self.transform:
            img = self.transform(img)

        return img, label