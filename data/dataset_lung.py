from __future__ import annotations
from dataclasses import dataclass

import scanpy as sc
import anndata
from anndata.experimental.pytorch import AnnLoader
from anndata.experimental.multi_files import AnnCollection

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from loguru import logger
import numpy as np


@dataclass
class MetaData:
    name: str
    train_loader: AnnLoader
    test_loader: AnnLoader
    label_mapping: dict
    cls_num: int
    random_state: int


def generate_train_test_loader(
    path: str,
    test_size: float = 0.2,
    batch_size: int = 32,
    shuffle: bool = True,
    device: str = "cuda",
    random_state=42,
    **kwargs,
) -> tuple[AnnLoader, AnnLoader]:
    """
    Generate train and test loaders from a h5ad file.

    Parameters
    ----------
    path : str
        Path to the h5ad file.
    test_size : float, optional
        Size of the test set, by default 0.2.
    batch_size : int, optional
        Batch size, by default 32.
    shuffle : bool, optional
        Whether to shuffle the data, by default True.
        
    Returns
    -------
    tuple[AnnLoader, AnnLoader]
        Train and test loaders.
    """
    pancreas_anndata = sc.read_h5ad(path)
    logger.info(f"Read {pancreas_anndata.shape[0]} cells and {pancreas_anndata.shape[1]} genes from {path}")
    logger.info(f"Label class counts: {pancreas_anndata.obs['cell_type'].value_counts()/pancreas_anndata.shape[0]}")
    
    # 将 cell_type 列进行分类编码
    encoder_cell_type = LabelEncoder()
    encoder_cell_type.fit(pancreas_anndata.obs['cell_type'])
    pancreas_anndata.obs['cell_type'] = encoder_cell_type.transform(pancreas_anndata.obs['cell_type']).astype(np.longlong)
    label_mapping = {v: k for k, v in zip(encoder_cell_type.classes_, encoder_cell_type.transform(encoder_cell_type.classes_))}
    logger.info(f"Label class mapping {label_mapping}")
    
    # 将原始数据转换为 AnnCollection 也就是 AnnData 的集合
    pancreas_dataset = AnnCollection([pancreas_anndata])
    
    train_pancreas, test_pancreas = train_test_split(
        pancreas_dataset, random_state=random_state, test_size=test_size, stratify=pancreas_anndata.obs['cell_type']
    )

    train_loader = AnnLoader(
        train_pancreas, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        use_cuda=device == "cuda"
        )
    test_loader = AnnLoader(
        test_pancreas, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        use_cuda=device == "cuda"
        )
    pancreas_metadata = MetaData(
        name="lung",
        train_loader=train_loader, 
        test_loader=test_loader, 
        label_mapping=label_mapping, 
        cls_num=len(label_mapping), 
        random_state=random_state
        )
    return pancreas_metadata
    
    
if __name__ == "__main__":
    pancreas_metadata = generate_train_test_loader(
        "E:/桌面/DFS/test_with_mnist/high-dim-fs/data/lung.h5ad"
        )
    print(pancreas_metadata)
    
    
