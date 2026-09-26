# AWS AI & ML Scholarship – Udacity AI Programming Nanodegree Projects

Projects I completed in the **AWS AI & ML Scholarship Program with Udacity** (Future AWS AI Scientist track, 2025). They cover deep learning with **PyTorch**, from using pretrained CNNs for image classification to building a Transformer from scratch for NLP.

## Projects

| Project | Area | What it does | Key result |
|---|---|---|---|
| [Dog Breed Image Classifier](./dogs_images_classifier) | Computer Vision | Uses pretrained CNNs (ResNet, AlexNet, VGG) to decide whether an image shows a dog and identify its breed, then compares the three architectures | VGG: 100% dog vs. not-dog accuracy, 93.3% breed accuracy |
| [SentimentScope](./sentimentAnalysis_movieReview) | NLP | A Transformer built from scratch that classifies IMDB movie reviews as positive or negative | 81% test accuracy |

## Tech Stack

Python · PyTorch · torchvision · NumPy · pandas · Matplotlib · scikit-learn · Jupyter Notebook

## Getting Started

```bash
git clone https://github.com/Ghadeer074/AWSxUdacity-nanodegree-projects-.git
cd AWSxUdacity-nanodegree-projects-
python -m venv myenv
source myenv/bin/activate        # Windows: myenv\Scripts\activate
pip install -r requirements.txt
```

Each project folder has its own README with instructions for running it.

## Repository Structure

```
├── dogs_images_classifier/          # CNN image classification project
├── sentimentAnalysis_movieReview/   # Transformer sentiment analysis project
└── requirements.txt                 # Shared Python dependencies
```

## Author

**Ghadeer Fallatah**: [LinkedIn](https://www.linkedin.com/in/ghadeer-fallatah-842988286) · [GitHub](https://github.com/Ghadeer074)
