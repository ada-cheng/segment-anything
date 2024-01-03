import json
import nibabel as nib
import numpy as np
import sys
from tqdm import tqdm

"""
BTCV 数据集的格式：

BTCV 数据集的内容有 image 和 label 两种，都是 .nii.gz 格式文件，其中
    image:
        · 尺寸: (Height, Width, #slices)
        · 每一个 image[ :, :, slice] 代表了一个三维医学图像的第 slice 个二维切片
        · 每个像素点 image[H, W, s] 的值范围不固定，所以需要在喂给 SAM 之前先归一到 [0, 255] 之间

    label:
        · 尺寸: (Height, Width, #slices)
        · 每一个 label[ :, :, slice] 代表的是对应的 image 二维切片的标签
        · 每个像素点 label[H, W, s] 的值都是 **[0, 13]内的整数** ，代表了这个像素点的分类标签，其中， 
          0 代表不属于任何器官，1-13 各自代表了一种器官。所以加载的时候不用归一化

"""

class BTCV_loader:
    """
    用于加载 BTCV 数据集的类
    """
    def __init__(self, json_path: str, data_path: str) -> None:
        self.json_path = json_path
        self.data_path = data_path


    def load_data_with_label(self, dataset: str):
        """
        加载数据集和其对应的标签

        输入：
            dataset: 正在加载的数据集标签， 这里可以是 'training' 或 'validation' ，
                     与 json 文件中一致

        输出：
            images: 一个列表，第 i 个元素提取自数据集中第 i 个 .nii.gz 数据文件 
            labels: 一个列表，第 i 个元素提取自数据集中第 i 个 .nii.gz 标签文件
        """
        with open(self.json_path) as json_file:
            config = json.load(json_file)
            print(f"Loading '{dataset}'...")

            images, labels = [], []
            # 加载数据集中的每个图像和标签
            for paths in tqdm(config[dataset]):
                # 加载图像
                image: np.ndarray = self.load_btcv_image(self.data_path + paths["image"])
                images.append(image)

                # 加载标签
                label: np.ndarray = self.load_btcv_label(self.data_path + paths["label"])
                labels.append(label)

            return images, labels


    def load_btcv_image(self, path: str) -> np.ndarray:
        """
        加载单个 .nii.gz 格式图像文件，需要把数据范围归一到 [0, 255] 内，并把单通道复制成三通道
        
        输入：
            path: 文件路径

        输出：
            image: np.ndarray 格式的文件数据。形状为 (Height, Width, 3, #slices) ，
                  其中 #slices 表示该 3D 图像的 2D 切片个数
        """
        # 加载原始的 3D 图像
        image: np.ndarray = nib.load(path).get_fdata()
        # image.shape = (Height, Width, #slices)

        # 因为 SAM 要求像素的值用 uint8 表示，范围在 [0, 255] 内，所以需要
        # 对像素的值进行规范化，并把格式转化为 uint8
        pixel_min, pixel_max = image.min(), image.max()
        image = (image - pixel_min) / (pixel_max - pixel_min) * 255
        image = image.astype(np.uint8)

        # 把单通道图像转化为三通道，才能喂给 SAM
        image = self.grey2rgb(image)

        return image


    def load_btcv_label(self, path: str) -> np.ndarray:
        """
        加载单个 .nii.gz 格式标签文件，不需要做额外操作
        
        输入：
            path: 文件路径

        输出：
            label: np.ndarray 格式的文件数据。形状为 (Height, Width, #slices) ，
                  其中 #slices 表示该 3D 标签的 2D 切片个数
        """
        # 加载原始的 3D 标签
        label: np.ndarray = nib.load(path).get_fdata()
        label = label.astype(np.uint8)

        return label


    def grey2rgb(self, image: np.ndarray) -> np.ndarray:
        """
        把单通道灰度图像转化为三通道RGB图像。转化的方式为把单通道直接复制三份。

        输入：
            image: 单通道图像。形状为 (Height, Width, #slices)

        输出：
            image: 三通道图像。形状为 (Height, Width, 3, #slices)
        """
        image = np.expand_dims(image, axis=2)
        # image.shape = (Height, Width, 1, #slices)

        image = np.repeat(image, 3, axis=2)
        # image.shape = (Height, Width, 3, #slices)
        # 此时 image[:, :, :, slice] 就是第 slice 个 2D 切片，形状为 HxWx3 

        return image