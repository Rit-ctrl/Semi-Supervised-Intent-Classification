# Semi-Supervised Intent Classification

## Overview
This repository contains the code and resources for a semi-supervised approach to intent classification. Intent classification is a crucial task in natural language understanding, where the goal is to identify the intention behind a user's query. Semi-supervised learning leverages both labeled and unlabeled data to improve the performance of intent classification models, especially when labeled data is scarce.

## Approach

The task of intent classification heavily relies on large, labelled datasets. To overcome this requirement, we use GAN, which generates a synthetic labelled dataset, alongside BERT, that provides a feature extractor. This task uses both labelled and unlabelled data to aid with intent classification in natural language processing. We experimented with different ratios of labelled and unlabelled data to evaluate the model under varying degrees of supervision. 

## Dataset 

The dataset used for this project was obtained from MASSIVE. MASSIVE is short for Multilingual Amazon SLU (Spoken Language Understanding) Resource Package for
Slot-filling, Intent Classification, and Virtual Assistant Evaluation. It contains over 1 million parallel utterances across 52 languages, covering 18 domains, with 60 intents,
and 55 slot types. We sourced the intent classification ‘zh-CN’ dataset, which stands for ‘Chinese - China’, from Hugging Face under the identifier ‘AmazonScience/massive’.
This dataset contains 16.5K rows.

## Architecture 

![image](https://github.com/user-attachments/assets/7cf62936-4ffb-424c-a79d-2d2fe8fa10d1)


The architecture can be divided into three components namely:

- Generator block: This block generates synthetic/fake text samples from a Gaussian noise input. The input to the generator is a 100-dimensional noise vector. The
noise vector is then passed through a hidden layer with LeakyReLU activation and dropout regularization. The output from this block is of the same dimensions as the
hidden size of the transformer.

- Discriminator block: This block takes the output representations from generator and BERT model and passes them through a hidden layer with LeakyReLU activation and dropout regularization to classify the input text as real or fake. In the case of real text, the corresponding intent is also predicted. The output size from this block is the number of intents + 1, where the extra dimension is for the classification of the sample being fake or real.

- Language Model: A pre-trained BERT base-Chinese model is chosen as the language model for this task. This model is trained on a large corpus of Chinese
text data making it suitable for our task. The architecture of this model comprises of 12 transformer encoder layers, with 768 hidden units and 12 attention heads. The maximum sequence length is set as 64. The model takes both labelled and unlabelled real data and the output representations are passed to the discriminator block for classification.

## Training

The GAN is trained in an adversarial manner, where the generator tries to generate text similar to real data and the discriminator tries to distinguish between real and fake
text as well as classify the intents in the case of real texts.

The generator’s loss is the addition of two components namely the discriminator output for fake samples and the feature regularization loss. The discriminator output loss aims to push the generator to produce samples that maximize the probability of being classified as real by the discriminator. The feature regularization pushes the generator to produce samples whose features are similar to those of features obtained from real text.

The discriminator’s loss is the sum of two losses namely the supervised and unsupervised loss. Supervised loss measures the loss in assigning the wrong intent to labelled
data whereas unsupervised loss measures the error in wrongly classifying the fake texts as real and vice versa.

Both the generator and discriminator use AdamW (Adam with a weight decay)  with a learning rate of 5e-5. The batch size is set as 64.

## Results

Since the MASSIVE dataset does not contain unlabelled data, we mask the labels of a fraction of the training set as unlabelled data and evaluate the performance based on different variants.

| Variant   | Fraction masked (unlabelled) | Number of epochs |
|-----------|--------------------------------|------------------|
| 1         | 0.9                            | 20               |
| 2         | 0.8                            | 18               |
| 3         | 0.6                            | 16               |
| 4         | 0.4                            |  14              |
| 5         | 0.2                            | 12               |
| 6         | 0.1                            | 10               |

![image](https://github.com/user-attachments/assets/1b1da68e-0f2f-44ce-a94b-0b67c96a3ad3)

![image](https://github.com/user-attachments/assets/f05dc15a-3e5e-4e03-9034-d757223d7149)

| Variant      | Fraction masked (unlabelled) |Accuracy |
|--------------|----------|-----------|
| 1            | 0.9      |77.03    |
| 2            | 0.8        | 81.00    |
| 3            | 0.6      |83.09    |
| 4            | 0.4        |84.63    |
| 5            | 0.2         |85.34    |
| 6            | 0.1        |85.34    |
| Full dataset | 0         |85.4     |


It is observed that even with only 10% labelled data and 90% unlabelled data (Variant 1), 77.03 % intent classification accuracy is attained, compared to 85.4% accuracy when the full dataset is labelled. So the GAN-BERT model can classify intents using limited labelled training data with reasonable accuracy.

