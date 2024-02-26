from typing import Any, Literal, Union
from pathlib import Path
import jsonlines
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import copy
import torch

from src.utils import load_json_annotations, bbox_augmentation_resize


class FinTabNet(Dataset):
    """Load PubTabNet for different training purposes."""

    def __init__(
        self,
        root_dir: Union[Path, str],
        label_type: Literal["image", "html", "cell", "bbox"],
        transform: transforms = None,
        jsonl_filename: Union[Path, str] = None,
    ) -> None:
        super().__init__()

        self.root_dir = Path(root_dir)
        self.label_type = label_type
        self.transform = transform

        if label_type != "image":
            jsonl_file = self.root_dir / jsonl_filename
            with jsonlines.open(jsonl_file) as f:
                self.image_label_pair = list(f)

    def __len__(self):
        return len(self.image_label_pair)

    def __getitem__(self, index: int) -> Any:
        if self.label_type == "image":
            raise NotImplementedError
        else:
            obj = self.image_label_pair[index]
            img_name = f"{obj['table_id']}.png"
            img = Image.open(self.root_dir / "image" / img_name)
            img_size = img.size
            if self.transform:
                img = self.transform(img)
            tgt_size = img.shape[-1]

            sample = dict(filename=obj["filename"], image=img)

            if self.label_type == "html":
                # table structure only
                sample["html"] = obj["html"]["structure"]["tokens"]
                return sample
            else:
                raise NotImplementedError
