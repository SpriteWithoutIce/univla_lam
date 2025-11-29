import argparse
import glob
import io
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


FEATURE_DESCRIPTION = {
    "steps/is_first": tf.io.VarLenFeature(tf.int64),
    "steps/is_last": tf.io.VarLenFeature(tf.int64),
    "steps/is_terminal": tf.io.VarLenFeature(tf.int64),
    "steps/action": tf.io.VarLenFeature(tf.float32),
    "steps/discount": tf.io.VarLenFeature(tf.float32),
    "steps/reward": tf.io.VarLenFeature(tf.float32),
    "steps/observation/state": tf.io.VarLenFeature(tf.float32),
    "steps/observation/image": tf.io.VarLenFeature(tf.string),
    "steps/observation/left_wrist_image": tf.io.VarLenFeature(tf.string),
    "steps/observation/right_wrist_image": tf.io.VarLenFeature(tf.string),
    "steps/observation/low_cam_image": tf.io.VarLenFeature(tf.string),
    "steps/language_instruction": tf.io.VarLenFeature(tf.string),
    "episode_metadata/file_path": tf.io.FixedLenFeature([], tf.string),
}
def _parse_example(example_proto: tf.Tensor) -> Dict:
    example = tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION)
    for key, value in example.items():
        if isinstance(value, tf.SparseTensor):
            example[key] = tf.sparse.to_dense(value)
    return example


def _collect_tfrecord_paths(raw_path: str) -> List[str]:
    """Expand a TFRecord directory, glob pattern, or file into concrete paths."""

    path = Path(raw_path)
    if path.is_dir():
        tfrecord_files = sorted(
            str(p) for p in path.glob("*.tfrecord*") if p.is_file()
        )
        if not tfrecord_files:
            raise ValueError(f"No TFRecord files found in directory: {raw_path}")
        return tfrecord_files

    if any(token in raw_path for token in ["*", "?", "["]):
        tfrecord_files = sorted(glob.glob(raw_path))
        if not tfrecord_files:
            raise ValueError(f"Glob did not match any files: {raw_path}")
        return tfrecord_files

    if not path.exists():
        raise FileNotFoundError(f"TFRecord path does not exist: {raw_path}")

    return [str(path)]


def _bytes_to_pil(image_bytes: bytes) -> Image.Image:
    with io.BytesIO(image_bytes) as buffer:
        image = Image.open(buffer).convert("RGB")
    return image


class TFRecordTrajectoryDataset(Dataset):
    def __init__(
        self,
        tfrecord_paths: Iterable[str],
        frame_interval: int = 32,
        subset_fraction: float = 0.1,
        action_dim: int = 14,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.frame_interval = frame_interval
        self.action_dim = action_dim
        self.transform = transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
        )

        dataset = tf.data.TFRecordDataset(list(tfrecord_paths))
        samples: List[Dict] = []
        num_trajectories = 0
        total_steps = 0


        for raw_record in dataset:
            parsed = _parse_example(raw_record)
            num_steps = parsed["steps/is_first"].shape[0]
            num_trajectories += 1
            total_steps += num_steps

            actions = (
                parsed["steps/action"].numpy().astype(np.float32).reshape(num_steps, action_dim)
            )
            images = parsed["steps/observation/image"].numpy()
            instruction = (
                parsed["steps/language_instruction"].numpy()[0].decode("utf-8")
                if len(parsed["steps/language_instruction"]) > 0
                else ""
            )

            for start in range(0, num_steps - frame_interval):
                target_idx = start + frame_interval
                if target_idx >= num_steps:
                    break
                samples.append(
                    {
                        "start_image": images[start],
                        "target_image": images[target_idx],
                        "action": actions[target_idx],
                        "instruction": instruction,
                    }
                )
        print(f"🔥 Loaded {num_trajectories} trajectories with total {total_steps} steps.")
        if subset_fraction < 1.0:
            random.shuffle(samples)
            keep = max(1, int(len(samples) * subset_fraction))
            samples = samples[:keep]

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        start_image = _bytes_to_pil(sample["start_image"])
        target_image = _bytes_to_pil(sample["target_image"])

        video = torch.stack(
            [self.transform(start_image), self.transform(target_image)], dim=0
        )
        action = torch.tensor(sample["action"], dtype=torch.float32)

        return {
            "videos": video,
            "task_instruction": sample["instruction"],
            "action": action,
            "dataset_names": "tfrecord_finetune",
        }


class TFRecordDataModule(LightningDataModule):
    def __init__(
        self,
        tfrecord_paths: List[str],
        batch_size: int = 2,
        frame_interval: int = 32,
        subset_fraction: float = 0.1,
        num_workers: int = 4,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.tfrecord_paths = tfrecord_paths
        self.batch_size = batch_size
        self.frame_interval = frame_interval
        self.subset_fraction = subset_fraction
        self.num_workers = num_workers
        self.image_size = image_size

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = TFRecordTrajectoryDataset(
            self.tfrecord_paths,
            frame_interval=self.frame_interval,
            subset_fraction=self.subset_fraction,
            image_size=self.image_size,
        )
        

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LAM on TFRecord trajectories")
    parser.add_argument(
        "tfrecord",
        type=str,
        help="Path to a TFRecord file, directory of TFRecords, or glob pattern",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/linyihan/linyh/univla_lam/checkpoints/lam-stage-2.ckpt",
        help="Stage-2 checkpoint path",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--frame-interval", type=int, default=32)
    parser.add_argument("--subset-fraction", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument("--action-hidden", type=int, default=256)

    args = parser.parse_args()

    from lightning import Trainer
    from genie.model import DINO_LAM

    tfrecord_paths = _collect_tfrecord_paths(args.tfrecord)
    print(f"🔥 Found {len(tfrecord_paths)} TFRecord files. {tfrecord_paths}")

    datamodule = TFRecordDataModule(
        tfrecord_paths=tfrecord_paths,
        batch_size=args.batch_size,
        frame_interval=args.frame_interval,
        subset_fraction=args.subset_fraction,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    datamodule.setup()
    print(f"🔥 Dataset size: {len(datamodule.dataset)}")

    model = DINO_LAM(
        stage="stage-2",
        predict_actions=True,
        action_loss_weight=args.action_loss_weight,
        action_head_dim=args.action_hidden,
        image_channels=3,
        lam_model_dim=768,
        lam_latent_dim=128,
        lam_num_latents=16,
        lam_patch_size=14,
        lam_enc_blocks=12,
        lam_dec_blocks=12,
        lam_num_heads=12,
        lam_dropout=0.0,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)

    if missing or unexpected:
        print(f"⚠️ Missing keys when loading checkpoint: {missing}")
        print(f"⚠️ Unexpected keys when loading checkpoint: {unexpected}")

    trainer = Trainer(max_epochs=args.max_epochs, devices=args.devices, accelerator="gpu")
    trainer.fit(model, datamodule=datamodule)
