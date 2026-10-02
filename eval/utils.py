"""Shared utilities for per-image evaluation scripts."""

import argparse
import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Thread

from PIL import Image


def add_common_args(parser: argparse.ArgumentParser):
    """Add common arguments shared by all eval scripts."""
    parser.add_argument("--root", type=str, required=True,
                        help="Root directory or specific run directory")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing per-image CSVs")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of I/O pre-loading threads")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 inference (default: True)")
    parser.add_argument("--no_fp16", action="store_true",
                        help="Disable FP16 inference")


def setup_gpu(gpu_id: int):
    """Set CUDA_VISIBLE_DEVICES before any torch import."""
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def find_occupation_dirs(root: str):
    """Find all occupation directories containing images.

    Handles structures:
        - {root}/gd_*/w_*/occ/          (standard)
        - {root}/p_*/gd_*/w_*/occ/      (pag)
        - {root}/{MODEL}/{VER}/{METHOD}/gd_*/w_*/occ/  (full tree)

    Yields (occ_path,) for each occupation directory with images.
    """
    root = os.path.abspath(root)

    # Collect all directories that directly contain img_*.jpg
    seen = set()
    for pattern in ["**/img_*.jpg", "**/img_*.png"]:
        for img_path in glob.iglob(os.path.join(root, pattern), recursive=True):
            occ_dir = os.path.dirname(img_path)
            if occ_dir not in seen:
                seen.add(occ_dir)
                yield occ_dir


def load_images(occ_path: str):
    """Load all images from an occupation directory.

    Returns:
        images: list of PIL.Image
        filenames: list of filenames (e.g. 'img_0.jpg')
    """
    files = sorted(
        glob.glob(os.path.join(occ_path, "*.jpg")) +
        glob.glob(os.path.join(occ_path, "*.png"))
    )
    images = []
    filenames = []
    for f in files:
        try:
            img = Image.open(f).convert("RGB")
            images.append(img)
            filenames.append(os.path.basename(f))
        except Exception:
            pass
    return images, filenames


def preload_worker(occ_path):
    """Load images from a single occupation folder (runs in thread pool)."""
    images, filenames = load_images(occ_path)
    return occ_path, images, filenames


SENTINEL = None


def iter_preloaded(todo_dirs, num_workers=4):
    """Yield (occ_path, images, filenames) with threaded I/O pre-loading.

    Uses a bounded queue so at most num_workers*2 folders are in memory.
    """
    queue = Queue(maxsize=num_workers * 2)

    def producer():
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(preload_worker, d): d for d in todo_dirs}
            for future in as_completed(futures):
                try:
                    queue.put(future.result())
                except Exception as e:
                    print(f"Error loading {futures[future]}: {e}")
        queue.put(SENTINEL)

    prod_thread = Thread(target=producer, daemon=True)
    prod_thread.start()

    while True:
        item = queue.get()
        if item is SENTINEL:
            break
        yield item

    prod_thread.join()
