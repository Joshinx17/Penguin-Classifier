# Penguin Image Classifier

Deep learning image classifier for identifying penguin classes from photos.

The project uses transfer learning with MobileNetV2, so you can train a useful
classifier with fewer images than a model trained from scratch.

## Dataset format

Place images in one folder per class:

```text
data/
  penguins/
    Adelie/
      image-1.jpg
      image-2.jpg
    Chinstrap/
      image-1.jpg
    Gentoo/
      image-1.jpg
```

You can use any class names you want. The folder names become the prediction
labels.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train_image_classifier.py --data-dir data/penguins --epochs 10
```

The first run may download pretrained MobileNetV2 weights. If you need to train
without downloading them, use:

```bash
python train_image_classifier.py --data-dir data/penguins --weights none
```

Training saves:

- `models/penguin_image_classifier.keras`
- `models/class_names.txt`

## Predict from the command line

```bash
python predict_image.py path/to/penguin.jpg
```

## Run the web app

```bash
streamlit run app.py
```

Upload a penguin image in the app to see the predicted class and probability
scores.
