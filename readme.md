# Emoji Classification

## Description

◻️ Use pre-trained word vectors with [glove-6B](https://drive.google.com/drive/u/0/folders/1872m1KLAKxc1vDe2FIF080nRBErMdKge)

◻️ Use this [Dateset](https://drive.google.com/drive/u/0/folders/1IdNma6S94cvp07WjjRXHoljxN1UyBy9E) for train model


## How to install
```
pip insatll -r requirements.txt
```

## How to run
```
python class_emoji_text_classification.py --vector_shape dimention --features_path features_path --infrence sentence
```

## Results

◻️Model with Dropout

| Features Vector Dimensions  | Train Loss  | Train Accuracy |  Test Loss |  Test Accuracy | Inference Time |
| ----------------------      | ------      | --------       |   -------  | ---            | ---            |
| 50d                         |   0.7244    |    77.27%      |  0.7332    |    75.57%      |     0.068      | 
| 100d                        |   0.6523    |    78.79%      |  0.6593    |    79.59%      |     0.07       |
| 200d                        |   0.3144    |    94.7%       |  0.5209    |    83.67%      |     0.076      | 
| 300d                        |   0.2055    |    97.73%      |  0.4601    |    89.8%      |     0.0861      | 
