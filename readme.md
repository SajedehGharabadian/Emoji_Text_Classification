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

◻️Result with Dropout

| Features Vector Dimensions  | Train Loss  | Train Accuracy |  Test Loss |  Test Accuracy | Inference Time |
| ----------------------      | ------      | --------       |   -------  | ---            | ---            |
| 50d                         |   0.7244    |    77.27%      |  0.7332    |    75.57%      |     0.068s     | 
| 100d                        |   0.6523    |    78.79%      |  0.6593    |    79.59%      |     0.0705s    |
| 200d                        |   0.3144    |    94.7%       |  0.5209    |    83.67%      |     0.076s     | 
| 300d                        |   0.2055    |    97.73%      |  0.4601    |    89.8%      |     0.0861s     | 


◻️Result without Dropout

| Features Vector Dimensions  | Train Loss  | Train Accuracy |  Test Loss |  Test Accuracy | Inference Time |
| ----------------------      | ------      | --------       |   -------  | ---            | ---            |
| 50d                         |   0.6304    |    83.33%      |  0.7163    |    79.59%      |     0.0607s    | 
| 100d                        |   0.4839    |    90.91%      |  0.6053    |    85.71%      |     0.0633s    |
| 200d                        |   0.2704    |    95.45%      |  0.5055    |    85.71%      |     0.0717s    | 
| 300d                        |   0.1807    |    99.24%      |  0.4456    |    87.76%      |     0.082s     | 
