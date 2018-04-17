# Diacritics restoration using neural networks

This project contains code for training, evaluating and inferring state-of-the-art models for diacritics restoration. This repository accompanies our paper to LREC2018.

## Paper summary

Traditionally, statistical methods based on word n-grams and utilizing part-of-speech tags, morphological and many other features were used for diacritics restoration.
In the paper, we have shown that employing character-level recurrent neural based model together with language model that are trained solely from texts without diacritics and diacritized texts (and thus does not require developer to separately train - or even handcraft - numerous features and connect them together) performs much better on two existing datasets.
Since there is no consistent pipeline for obtaining datasets for this task, we proposed a new pipeline and run it on 12 languages.
When trained on this new dataset, our model outperforms two baseline methods on all 12 languages by a big margin.
It reduces word error rate of the stronger contextual baseline (which is a previous state-of-the-art method for several languages) by 24% to 83%.

## Training a diacritics restoration system

### Prepare dataset

The dataset for 12 languages can be obtained from https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2607 or downloaded using custom scripts that are located in *data* folder (see *data/README.md* for more information on how to download data).
After

# TODO remove diacritics and create dataset config file

### Training

Once you have a dataset, you can train a model for diacritics generation.
The entry point for this is *train.py* script, where you can specify a number of model hyperparameters (e.g. batch size, learning rate or number of stacked layers), a path to dataset configuration file (the only mandatory argument) and several other stuff like where to store model.
The default values of hyperparameters are those used in the paper and should work well.

Supposing your dataset configuration file is located at dataset_config.txt, you can train a model using:

```
python3 train.py dataset_config.txt
```

This will store model checkpoints to save/exp_name/timestep directory and TensorBoard files (with word and character accuracies) to logs/exp_name-timestep directory. Note that training a well-working system may take a time (models in the paper were trained for approximately 4 days on GeForce GTX 1080 Ti, but you can expect reasonable result when training on rather strong CPU for several days).

### Using trained model for diacritizing text

To

## Requirements:

- Python 3.5

- TensorFlow >= 1.5

## Citing

If you use this code in your work, please consider citing our paper: "Diacritics Restoration Using Neural Networks", Jakub Náplava, Milan Straka, Pavel Straňák, Jan Hajič, LREC 2018.

TODO