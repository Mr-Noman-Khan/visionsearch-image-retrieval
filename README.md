# VisionSearch

VisionSearch is a small image-search app I built to try out CLIP. You can describe an image in a sentence or upload a picture, and the app finds the closest matches in a local folder of images.

It uses a pretrained CLIP model to turn text and images into vectors, then ranks the collection by cosine similarity. The results are useful for exploring a collection, but the scores are similarity values, not confidence probabilities.

## What it can do

- Find images from a natural-language description.
- Find visually similar images from an uploaded reference.
- Show the matching images and their similarity scores in a Streamlit app.
- Search images stored in `data/images/` on your machine.

## Get started

You’ll need Python 3.10 or newer. From the project folder, create and activate a virtual environment, then install the dependencies.

On Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Add images you’re allowed to use to `data/images/`. The app accepts JPG, JPEG, PNG, WEBP, and BMP files, including images in subfolders. Then start it with:

```bash
streamlit run app.py
```

Choose **Text search** to enter a description, or **Image search** to upload a reference. The first launch downloads the pretrained model weights, so it needs an internet connection. The app chooses CUDA when it’s available and otherwise runs on the CPU.

## How it works

When the app starts, it finds the supported images and encodes them with OpenCLIP’s pretrained `ViT-B-32` model. Your text query or uploaded image is encoded into the same feature space. The app compares the normalized vectors, sorts by cosine similarity, and displays the top results.

The collection itself is read from your local `data/images/` folder. Model weights are downloaded from the model provider on first use. Search quality depends on the model and the images; subtle details can be missed, and a high score doesn’t guarantee that a result is correct.

## Project files

```text
app.py                     Streamlit interface and search flow
visionsearch/retrieval.py  Reusable cosine-similarity helpers
data/images/               Demo images; add your own collection here
requirements.txt           Python dependencies
```

The app currently performs its search directly in `app.py`; the helper in `visionsearch/retrieval.py` is available separately. The image collection is intended for local experimentation, and embeddings are rebuilt in memory when needed. There’s no formal retrieval benchmark included yet.

## Using images responsibly

No third-party image dataset is bundled. Make sure you have permission to use any images you add, and check the relevant licence. Avoid committing private or sensitive images to a public repository.
