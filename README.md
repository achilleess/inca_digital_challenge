# Inca Digital Challenge

## Problem

We have 12137 tweets. 52 annotators were asked to annotate each tweet with a true/false value depending on whether the tweets are relevant to a Hacker Attack topic or not. For each tweet, from 3 to 4 annotators have given their decisions. We also know that some of the annotators are bots. Solution for this problem will be a list of bots' IDs. \
Data may be found [here](https://docs.google.com/spreadsheets/d/1kvdrnZjlgjVacSdIsA7pb88sdiVE1Nw2H6rt3m5Be9M/edit?usp=sharing) or in the *data/* folder of the current repository.

## Solution

The idea is to somehow train a NLP Sequence Classification Model on a given data and then compare values predicted by the model and values obtained from the annotators. We can calculate the percentage of matched values for each annotator and we will call it an agreement rate. After that, we can set a threshold value. Annotators with the agreement rate lower than the threshold value will be counted as bots.

### Data
How are we going to extract only clean data from the given annotations to train the model?
Of course, we can't do it with 100% certainty, but we surely can maximize the fraction of clean data by only choosing the tweets where all 3-4 annotators have chosen the same value (all voted for True or all voted for False).

After this simple filtering procedure, we are left with 4342 samples. So let's split it into five folds, we will use four folds for cross-validation and the fifth fold (hold-out) is going to be used for the testing.

### Training
We will train Bert, Bart and GPTv2. Training procedure can be launched as follows:
```
python3 tools/train.py configs/bert_config.py <fold_to_train>
python3 tools/train.py configs/bart_config.py <fold_to_train>
python3 tools/train.py configs/gptv2_config.py <fold_to_train>
```

To further work with models' predictions we can save them into the table using the following script:
```
python3 tools/get_dataset_predictions.py <config_path> <model_weights_path> <csv_path_for_results> <res_column_name> <trained_fold>
```
Repeat ```get_dataset_predictions.py``` procedure for each trained model and always keep <csv_path_for_results> parameter the same.

After the training we got the following table 
| Models | F1, fold-0 | F1, fold-1 | F1 fold-2 | F1 fold-3 | F1 test |
| --- | --- | --- | --- | --- | --- |
| Bert large | 0.8677 | 0.8777 | **0.8826** | 0.8883 | 0.8771 (avg) |
| Bart base | **0.8723** | **0.8794** | 0.8822 | **0.8902** | 0.8935 (avg) |
| GPTv2 | 0.8582 | 0.8622 | 0.8723 | 0.8824 | 0.8669 (avg) |
| Bert(fold_01) + Bart(fold_0123) + GPTv2(fold_3) | - | - | - | - | **0.9022** |


### Results
Now let's calculate the agreement rate for each annotator.  

The following table includes annotators with the lowest agreement rate:
| Annotator ID | Agreement rate | Num of annotated sampels |
| --- | --- | --- |
| A3BJX6UUSOIKFN | 0.5339 | 1472 |
| A3OCJJMRKAIJZA | 0.5824 | 3561 |
| A33Y36Y252Z30U | 0.6060 | 5001 |
| A3BCKNE5CWHODZ | 0.6521 | 1443 |
| A1YSYI926BBOHW | 0.7023 | 84 |
| A2KHLJ2F58BEZK | 0.7051 | 78 |
| AMYURTQIMAC8T | 0.7255 | 532 |

The table can be obtained from the ```analalyze_preds.ipynb``` notebook.

From this table we can outline the most probable bots:
```
A3BJX6UUSOIKFN, A3OCJJMRKAIJZA, A33Y36Y252Z30U, A3BCKNE5CWHODZ
```
The list of annotators with the agreement rate lower than 0.75 also can potentially include bots, but it requires a closer look to spot them.