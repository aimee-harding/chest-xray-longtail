import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler 
import torchvision.transforms as T #preprocessing operations
import torch


# ── 1. Label definitions ──────────────────────────────────────────────────────

PATHOLOGY_LABELS = [
    "Atelectasis",     
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",       
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",           
]

NUM_CLASSES = len(PATHOLOGY_LABELS)  


# ── 2. Transform pipelines ─────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transforms(split: str) -> T.Compose:
    if split == "train":
        # resize to 256, random crop to 224, random horizontal flip.
        return T.Compose([
            T.Resize((256, 256)),
            T.RandomCrop((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN,
                        IMAGENET_STD),
        ])
    else:  # val or test — resize then centre crop
        return T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop((224, 224)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN,
                        IMAGENET_STD),
        ])


# ── 3. The Dataset class ───────────────────────────────────────────────────────

class ChestXray14Dataset(Dataset):
    #constructor
    def __init__(
        self,
        data_dir: str, #where dataset is
        split: str = "train", #which subset you're creating
        transform=None, #torchvision transform pipeline (uses get_transforms if None)
        val_frac: float = 0.1, #fraction of patients to hold out for validation from official train/val pool
        seed: int = 42, #random seed for reproducible patient split
    ):
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got '{split}'"

        self.data_dir  = data_dir
        self.split     = split
        self.transform = transform or get_transforms(split)

        # ── Build a filename → full path index ──────────────────────────────
        # The NIH dataset spreads images across 12 subfolders:
        #   images_001/images/, images_002/images/, ..., images_012/images/
        # We scan all of them once at startup and build a dict so __getitem__
        # can find any image in O(1) without searching 12 folders every time.
        self.image_index = self._build_image_index(data_dir)

        # ── 3a. Load the master CSV ──────────────────────────────────────────
        csv_path = os.path.join(data_dir, "Data_Entry_2017.csv")
        df = pd.read_csv(csv_path)

        # ── 3b. Load the official train/test file lists ──────────────────────
        train_val_files = self._load_filelist(os.path.join(data_dir, "train_val_list.txt"))
        test_files = self._load_filelist(os.path.join(data_dir, "test_list.txt"))

        # ── 3c. Filter the CSV to the correct official split ─────────────────
        if split == "test":
            df = df[df["Image Index"].isin(test_files)].reset_index(drop=True) #keep only CSV rows whose image filename appears in the official test list.
        else:
            df = df[df["Image Index"].isin(train_val_files)].reset_index(drop=True)

            # ── 3d. Patient-level train/val split ────────────────────────────
            all_patients = df["Patient ID"].unique()
            rng = np.random.default_rng(seed)
            rng.shuffle(all_patients)

            n_val = int(len(all_patients) * val_frac)
            val_patients   = set(all_patients[:n_val])
            train_patients = set(all_patients[n_val:])

            if split == "val":
                df = df[df["Patient ID"].isin(val_patients)].reset_index(drop=True)
            else:
                df = df[df["Patient ID"].isin(train_patients)].reset_index(drop=True)

        self.df = df

        # ── 3e. Build multi-hot label matrix ─────────────────────────────────
        self.labels = self._build_label_matrix(df)  # shape: (N, 14)

        print(f"[ChestXray14Dataset] {split}: {len(self.df)} images, "
              f"{self.labels.sum(axis=0).astype(int).tolist()} positives per class")

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_image_index(data_dir: str) -> dict:
        index = {}
        for folder in sorted(os.listdir(data_dir)):
            if not folder.startswith("images_"):
                continue
            img_dir = os.path.join(data_dir, folder, "images")
            if not os.path.isdir(img_dir):
                continue
            for fname in os.listdir(img_dir):
                if fname.endswith(".png"):
                    index[fname] = os.path.join(img_dir, fname)
        print(f"[ChestXray14Dataset] Image index built: {len(index):,} images found")
        return index

    @staticmethod
    def _load_filelist(path: str) -> set:
        """Read a .txt file of image filenames into a set for fast lookup."""
        with open(path, "r") as f:
            return set(line.strip() for line in f if line.strip())

    @staticmethod
    def _build_label_matrix(df: pd.DataFrame) -> np.ndarray:
        N = len(df)
        matrix = np.zeros((N, NUM_CLASSES), dtype=np.float32)

        label_to_idx = {label: i for i, label in enumerate(PATHOLOGY_LABELS)}

        for i, finding_str in enumerate(df["Finding Labels"]):
            if finding_str == "No Finding":
                continue  # all zeros — healthy image
            for finding in finding_str.split("|"):
                finding = finding.strip()
                if finding in label_to_idx:
                    matrix[i, label_to_idx[finding]] = 1.0

        return matrix

    # ── core Dataset interface ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        """
        Returns:
            image  : FloatTensor of shape (3, 224, 224)
            labels : FloatTensor of shape (14,)  — multi-hot binary vector
        """
        row = self.df.iloc[idx]
        img_path = self.image_index[row["Image Index"]]

        #convert greyscale x-ray to rgb as image-net pretrained model expects 3-channel input
        #replicate single channel across all 3 colour channels
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)  #apply preprocessing

        labels = torch.tensor(self.labels[idx], dtype=torch.float32) #turns label array into a pytorch tensor
        return image, labels

    # ── utility methods ───────────────────────────────────────────────────────

    def get_class_frequencies(self) -> np.ndarray:
        # returns the number of positive samples per class.
        return self.labels.sum(axis=0)

    def get_pos_weight(self) -> torch.Tensor:
        """
        Computes the positive class weight for BCE loss:
            pos_weight[c] = (N - n_pos[c]) / n_pos[c]

        WHY?
            In an imbalanced dataset, the loss contribution from the rare
            positive class is dwarfed by the many negatives. pos_weight
            tells BCEWithLogitsLoss to scale up the positive term so rare
            classes get more gradient signal. This is used in Stage 1 BCE training.

        Returns: FloatTensor of shape (14,)
        """
        n_pos = self.labels.sum(axis=0)                  # positives per class
        n_neg = len(self.labels) - n_pos                 # negatives per class
        pos_weight = n_neg / np.clip(n_pos, 1, None)    # avoid divide-by-zero
        return torch.tensor(pos_weight, dtype=torch.float32)

    def get_class_aware_sampler(self) -> WeightedRandomSampler:
        class_freq = self.get_class_frequencies()

        # For each image, find the rarest class it contains.
        # Images with "No Finding" (all zeros) get the median frequency.
        sample_weights = np.zeros(len(self.labels))
        median_freq = np.median(class_freq[class_freq > 0])

        for i, label_vec in enumerate(self.labels):
            if label_vec.sum() == 0:
                # "No Finding" image — assign median weight
                sample_weights[i] = 1.0 / median_freq
            else:
                # Weight by the rarest positive class this image contains
                present_freqs = class_freq[label_vec == 1]
                sample_weights[i] = 1.0 / present_freqs.min()

        return WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float64),
            num_samples=len(sample_weights),
            replacement=True,   # replacement needed for oversampling
        )