import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        自定义 Dataset 类，用于加载没有类别标签的图片文件夹。
        :param root_dir: 包含图片的文件夹路径。
        :param transform: 应用于图片的预处理操作。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self._get_image_paths()
        print(f"Loaded {len(self.image_paths)} images from {root_dir}")

    def _get_image_paths(self):
        """
        遍历文件夹，获取所有图片的路径。
        """
        image_paths = []
        for file in os.listdir(self.root_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(self.root_dir, file))
        return image_paths

    def __len__(self):
        """
        返回数据集的大小。
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        根据索引 idx 获取图片。
        :param idx: 图片的索引。
        :return: 图片张量。
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        pid = int(os.path.basename(image_path).split("_")[0])
        camid = 1
        return {
            'img': image,
            'pid': pid,
            'camid': camid,
            'img_path': image_path
        }


class AutomizedDataLoader:
    def __init__(self, data_dir, batch_size, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize((128, 384)),  
            transforms.ToTensor(),         
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])

    def load_train_data(self):
        train_dataset = CustomImageDataset(os.path.join(self.data_dir, 'train'), transform=self.transform)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        return train_loader
    
    def load_test_data(self):
        test_gallery_dataset = CustomImageDataset(os.path.join(self.data_dir, 'gallery'), transform=self.transform)
        test_query_dataset = CustomImageDataset(os.path.join(self.data_dir, 'query'), transform=self.transform)

        test_gallery_loader = DataLoader(test_gallery_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        test_query_loader = DataLoader(test_query_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        test_loader = {
            'query': test_query_loader,
            'gallery': test_gallery_loader
        }
        
        return test_loader