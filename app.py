from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image
import torch
import open_clip

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "data" / "images"
CACHE_DIR = ROOT / ".cache"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

st.set_page_config(page_title="VisionSearch", page_icon="🔎", layout="wide")
st.title("🔎 VisionSearch")
st.caption("Explore a collection of images using natural language or a reference image.")
st.info(
    "This demo uses a pretrained CLIP model. The first launch downloads model weights, "
    "so an internet connection is needed. Results are similarity rankings, not guarantees of correctness."
)

@st.cache_resource(show_spinner="Loading CLIP model…")
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model = model.to(device).eval()
    return model, preprocess, tokenizer, device

def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)

@st.cache_data(show_spinner=False)
def load_rgb(path_string: str) -> Image.Image:
    with Image.open(path_string) as im:
        return im.convert("RGB")

def encode_image(path: Path, model, preprocess, device: str) -> np.ndarray:
    image = load_rgb(str(path))
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model.encode_image(tensor)
        feature = feature / feature.norm(dim=-1, keepdim=True)
    return feature.cpu().numpy().astype("float32")[0]

def encode_text(query: str, model, tokenizer, device: str) -> np.ndarray:
    tokens = tokenizer([query]).to(device)
    with torch.no_grad():
        feature = model.encode_text(tokens)
        feature = feature / feature.norm(dim=-1, keepdim=True)
    return feature.cpu().numpy().astype("float32")[0]

@st.cache_data(show_spinner="Indexing images…")
def build_index(path_strings: tuple[str, ...], _model, _preprocess, device: str):
    # Model objects are intentionally excluded from Streamlit's cache key.
    vectors, valid_paths = [], []
    for item in path_strings:
        try:
            vector = encode_image(Path(item), _model, _preprocess, device)
            vectors.append(vector)
            valid_paths.append(item)
        except Exception:
            continue
    if not vectors:
        return np.empty((0, 512), dtype="float32"), tuple()
    return np.stack(vectors).astype("float32"), tuple(valid_paths)

def rank(query_vector: np.ndarray, image_vectors: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    scores = image_vectors @ query_vector
    order = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in order]

with st.sidebar:
    st.header("Search settings")
    mode = st.radio("Search mode", ["Text search", "Image search"])
    top_k = st.slider("Number of results", min_value=1, max_value=12, value=6)
    st.caption("Add your own permitted images to `data/images/` and restart the app.")

images = list_images(IMAGE_DIR)
if not images:
    st.warning("No images found in `data/images/`. Add JPG, PNG, WEBP, or BMP files to start.")
    st.markdown(
        "### Quick start\n"
        "1. Put a small image collection in `data/images/`.\n"
        "2. Run `streamlit run app.py`.\n"
        "3. Search with a descriptive sentence or upload a reference image."
    )
    st.stop()

try:
    model, preprocess, tokenizer, device = load_model()
    with st.spinner(f"Preparing embeddings for {len(images)} images…"):
        vectors, indexed_paths = build_index(tuple(str(p) for p in images), model, preprocess, device)
except Exception as exc:
    st.error(f"Could not load the CLIP model or index images: {exc}")
    st.markdown("Check your internet connection and installed dependencies, then restart the app.")
    st.stop()

if len(indexed_paths) == 0:
    st.error("None of the images could be processed. Check that the files are valid image files.")
    st.stop()

st.caption(f"Indexed {len(indexed_paths)} images · Device: {device.upper()}")

query_vector = None
if mode == "Text search":
    query = st.text_input("Describe the image you want to find", placeholder="e.g. a red car parked on a street")
    search_clicked = st.button("Search images", type="primary", disabled=not query.strip())
    if search_clicked:
        try:
            query_vector = encode_text(query.strip(), model, tokenizer, device)
            st.session_state["query_vector"] = query_vector
            st.session_state["query_mode"] = "text"
            st.session_state["query_label"] = query.strip()
        except Exception as exc:
            st.error(f"Could not encode text query: {exc}")
else:
    upload = st.file_uploader("Upload a reference image", type=["jpg", "jpeg", "png", "webp", "bmp"])
    search_clicked = st.button("Find similar images", type="primary", disabled=upload is None)
    if search_clicked and upload is not None:
        try:
            with Image.open(upload) as im:
                query_image = im.convert("RGB")
            tensor = preprocess(query_image).unsqueeze(0).to(device)
            with torch.no_grad():
                feature = model.encode_image(tensor)
                feature = feature / feature.norm(dim=-1, keepdim=True)
            st.session_state["query_vector"] = feature.cpu().numpy().astype("float32")[0]
            st.session_state["query_mode"] = "image"
            st.session_state["query_label"] = "Uploaded reference image"
            st.session_state["uploaded_preview"] = query_image
        except Exception as exc:
            st.error(f"Could not process uploaded image: {exc}")

if "query_vector" in st.session_state and st.session_state.get("query_mode") == ("text" if mode == "Text search" else "image"):
    st.subheader(f"Results for: {st.session_state.get('query_label', 'your query')}")
    if st.session_state.get("query_mode") == "image" and "uploaded_preview" in st.session_state:
        st.image(st.session_state["uploaded_preview"], caption="Reference image", width=180)
    results = rank(st.session_state["query_vector"], vectors, min(top_k, len(indexed_paths)))
    cols = st.columns(3)
    for pos, (idx, score) in enumerate(results):
        path = Path(indexed_paths[idx])
        with cols[pos % 3]:
            st.image(load_rgb(str(path)), use_container_width=True)
            st.markdown(f"**{path.name}**")
            st.caption(f"Cosine similarity: {score:.3f}")
else:
    st.markdown("### Try a search")
    st.write("Choose **Text search** to describe an image, or **Image search** to upload a reference image.")
