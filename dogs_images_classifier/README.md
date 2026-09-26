# Dog Breed Image Classifier with Pretrained CNNs

A Python command-line program that uses **pretrained convolutional neural networks (CNNs)** from `torchvision` to:

1. decide whether an image shows a **dog or not**, and
2. identify the **dog breed**.

It then compares three CNN architectures (**ResNet-18, AlexNet and VGG-16**) to find which one performs best. This project was part of the AWS AI & ML Scholarship – Udacity AI Programming with Python Nanodegree.

## Results

The models were tested on 40 labelled pet images (30 dogs, 10 non-dogs):

| Model | Correct "Dog" | Correct "Not Dog" | Correct Breed | Label Match |
|---|---|---|---|---|
| **VGG** | **100%** | **100%** | **93.3%** | **87.5%** |
| ResNet | 100% | 90% | 90.0% | 82.5% |
| AlexNet | 100% | 100% | 80.0% | 75.0% |

**Conclusion:** VGG was the most accurate at both dog detection and breed classification. It was also the most reliable model on my own uploaded test images. AlexNet was the fastest but the least accurate on breeds.

## How It Works

```
Pet image filenames ──► get_pet_labels.py        (extract true labels from filenames)
                    ──► classify_images.py       (run the chosen CNN on each image)
                    ──► adjust_results4_isadog.py (check labels against a list of dog names)
                    ──► calculates_results_stats.py (compute accuracy statistics)
                    ──► print_results.py         (print a summary and the misclassifications)
```

| File | Purpose |
|---|---|
| `check_images.py` | Main program that runs the full pipeline |
| `get_input_args.py` | Parses command-line arguments (`--dir`, `--arch`, `--dogfile`) |
| `classifier.py` | Loads the pretrained torchvision model and returns the predicted ImageNet label |
| `dognames.txt` | Reference list of valid dog names used for dog vs. not-dog checks |
| `*_pet_images.txt` / `*_uploaded_images.txt` | Saved output from each model run |

## Run It

From the repository root, after installing `requirements.txt`:

```bash
cd dogs_images_classifier

# Run a single model
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt

# Compare all three models (results are saved to text files)
sh run_models_batch.sh             # Windows: run_models_batch.bat
sh run_models_batch_uploaded.sh    # run on your own images in uploaded_images/
```

## Tech Stack

Python · PyTorch · torchvision (pretrained ResNet, AlexNet, VGG) · Pillow · argparse

## What I Learned

- Using transfer learning with pretrained ImageNet models
- Evaluating and comparing model architectures with clear metrics
- Structuring a modular, testable Python program driven by command-line arguments
