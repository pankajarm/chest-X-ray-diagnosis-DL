# Chest X-Ray Diagnosis with Deep Learning

A deep learning-powered web application that analyzes chest X-ray images and predicts the probability of **14 thoracic pathologies** in real time. Built with a DenseNet-121 backbone fine-tuned on the NIH ChestX-ray14 dataset, served through an async Python web server with a drag-and-drop UI.

## Pathologies Detected

The model classifies chest X-rays across 14 conditions simultaneously (multi-label classification):

| | | |
|---|---|---|
| Cardiomegaly | Emphysema | Pleural Effusion |
| Hernia | Infiltration | Mass |
| Nodule | Atelectasis | Pneumothorax |
| Pleural Thickening | Pneumonia | Fibrosis |
| Edema | Consolidation | |

Each prediction includes a confidence probability, sorted from most to least likely.

## Architecture

**Model:** DenseNet-121 (pre-trained on ImageNet) with a custom classification head consisting of a Global Average Pooling layer followed by a 14-unit Dense layer with sigmoid activation for multi-label output.

**Input:** Chest X-ray images resized to 320 x 320 pixels, normalized using dataset-specific mean (134.57) and standard deviation (64.80).

**Output:** 14 probability scores, one per pathology, ranked by confidence.

```
Input Image (320x320)
    |
DenseNet-121 (feature extraction)
    |
GlobalAveragePooling2D
    |
Dense(14, sigmoid)
    |
Per-pathology probabilities
```

## Project Structure

```
chest-X-ray-diagnosis-DL/
├── app/
│   ├── models/              # Pre-trained model weights (.h5, .hdf5)
│   ├── static/
│   │   ├── css/             # Stylesheets
│   │   ├── fonts/           # Web fonts
│   │   ├── images/          # Static images
│   │   ├── js/              # Frontend JavaScript
│   │   ├── vendor/          # Third-party frontend libraries
│   │   ├── index.html       # Upload UI (drag-and-drop)
│   │   ├── result1.html     # Results page template (header)
│   │   └── result2.html     # Results page template (footer)
│   └── server.py            # Starlette async web server
├── chest_Xray_Inference.py  # Standalone inference script
├── Dockerfile               # Docker build configuration
├── requirements.txt         # Python dependencies
├── app.yaml                 # Google App Engine Flex config
├── now.json                 # Vercel (Zeit) deployment config
├── .dockerignore
├── LICENSE                  # MIT License
└── README.md
```

## Quick Start

### Option 1: Run with Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/pankajarm/chest-X-ray-diagnosis-DL.git
cd chest-X-ray-diagnosis-DL

# Build the Docker image
docker build -t chest-xray-diagnosis .

# Run the container
docker run -p 8080:8080 chest-xray-diagnosis
```

Then open [http://localhost:8080](http://localhost:8080) in your browser.

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/pankajarm/chest-X-ray-diagnosis-DL.git
cd chest-X-ray-diagnosis-DL

# Install dependencies
pip install -r requirements.txt

# Start the server
python app/server.py serve
```

The app will be available at [http://localhost:8080](http://localhost:8080).

### Option 3: Standalone Inference (No Server)

For batch or scripted inference without the web UI:

```bash
python chest_Xray_Inference.py
```

Edit the `img` variable in the script to point to your X-ray image file.

## Usage

1. Open the web app in your browser at `http://localhost:8080`
2. Drag and drop a chest X-ray image onto the upload area (or click to browse)
3. The model processes the image and returns ranked predictions for all 14 pathologies with confidence scores

## Deployment

The project includes configuration for multiple deployment targets:

| Platform | Config File | Command |
|---|---|---|
| **Docker** | `Dockerfile` | `docker build -t chest-xray . && docker run -p 8080:8080 chest-xray` |
| **Google App Engine** (Flex) | `app.yaml` | `gcloud app deploy` |
| **Vercel / Zeit** | `now.json` | `now` or `vercel` |

## Tech Stack

- **Deep Learning:** TensorFlow 2.3 / Keras, DenseNet-121
- **Web Server:** Starlette (ASGI) + Uvicorn
- **Async I/O:** aiohttp, asyncio
- **Image Processing:** Pillow, NumPy
- **Containerization:** Docker (Python 3.8 slim-buster)
- **Model Format:** HDF5 (`.h5` / `.hdf5`)

## Model Weights

The application expects two weight files in `app/models/`:

- **`densenet.hdf5`** — DenseNet-121 base weights (ImageNet pre-trained)
- **`pretrained_model.h5`** — Fine-tuned classification head weights for chest X-ray pathology detection

> **Note:** Model weight files are not included in this repository due to their size. See the `model_file_url` variable in `app/server.py` to configure a remote download URL, or place the weight files directly in `app/models/`.

## Dataset

This model was trained on the [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) dataset, which contains 112,120 frontal-view chest X-ray images from 30,805 unique patients, labeled with up to 14 thoracic pathology categories extracted via NLP from radiology reports.

**Reference:** Wang et al., "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks," CVPR 2017.

## Requirements

- Python 3.8+
- TensorFlow 2.3
- ~2 GB RAM for model inference
- See `requirements.txt` for full dependency list

## License

This project is licensed under the [MIT License](LICENSE).

## Author

**Pankaj Mathur** — [@pankajmathur_](https://x.com/pankajmathur_) | [pankajmathur.com](https://pankajmathur.com)
