"""Multimodal retrieval prototype for an AI clothing stylist.

A text and/or image query is embedded with CLIP and matched (cosine similarity)
against a knowledge base of fabric/style notes (kb/fabric_notes.txt) and
clothing photos (images/).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parent
KB_FILE = ROOT / "kb" / "fabric_notes.txt"
IMAGE_DIR = ROOT / "images"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MODEL_NAME = "openai/clip-vit-base-patch32"

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _features(out) -> np.ndarray:
    """Newer transformers return an output object instead of a tensor."""
    if not isinstance(out, torch.Tensor):
        out = out.pooler_output
    return _normalize(out.cpu().numpy())


@torch.no_grad()
def embed_texts(model, processor, texts):
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    return _features(model.get_text_features(**inputs))


@torch.no_grad()
def embed_images(model, processor, paths):
    images = [Image.open(p).convert("RGB") for p in paths]
    inputs = processor(images=images, return_tensors="pt").to(device)
    return _features(model.get_image_features(**inputs))


def load_kb():
    """Return a list of (kind, content) items and their source for display."""
    notes = []
    if KB_FILE.exists():
        notes = [l.strip() for l in KB_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    images = sorted(p for p in IMAGE_DIR.glob("*") if p.suffix.lower() in IMAGE_EXTS)
    return notes, images


def build_index(model, processor, notes, images):
    items, vectors = [], []
    if notes:
        vectors.append(embed_texts(model, processor, notes))
        items += [("note", n) for n in notes]
    if images:
        vectors.append(embed_images(model, processor, images))
        items += [("image", str(p.relative_to(ROOT))) for p in images]
    if not items:
        sys.exit("Knowledge base is empty: add notes to kb/fabric_notes.txt or photos to images/.")
    return items, np.vstack(vectors)


def embed_query(model, processor, text=None, image=None, image_weight=0.5):
    """Embed a text and/or image query.

    When both are given, they are blended: image_weight=0 uses only the text,
    1 uses only the photo, 0.5 weighs them equally.
    """
    if text and image:
        t = embed_texts(model, processor, [text])[0]
        i = embed_images(model, processor, [image])[0]
        return _normalize((1 - image_weight) * t + image_weight * i)
    if text:
        return embed_texts(model, processor, [text])[0]
    return embed_images(model, processor, [image])[0]


def search(query_vec, index, k):
    scores = index @ query_vec
    top = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in top]


def main():
    parser = argparse.ArgumentParser(description="Stylist RAG retrieval prototype")
    parser.add_argument("--text", help="text query, e.g. 'grey suit for a wedding'")
    parser.add_argument("--image", help="path to a query photo")
    parser.add_argument("--k", type=int, default=5, help="number of results (default 5)")
    args = parser.parse_args()

    if not args.text and not args.image:
        parser.error("provide --text, --image, or both")
    if args.image and not Path(args.image).is_file():
        parser.error(f"image not found: {args.image}")

    model, processor = load_model()
    notes, images = load_kb()
    items, index = build_index(model, processor, notes, images)
    query_vec = embed_query(model, processor, args.text, args.image)

    print(f"Top {min(args.k, len(items))} results:")
    for rank, (i, score) in enumerate(search(query_vec, index, args.k), 1):
        kind, content = items[i]
        print(f"{rank}. [{kind}] {score:.3f}  {content}")


if __name__ == "__main__":
    main()
