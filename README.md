# DAISYLab@ISIC2018
Code for the ISIC2018 Lesion Diagnosis Challenge (Task 3)

Our team achieved second place while being the best approach using only publicly available data ([Leaderboards](https://challenge2018.isic-archive.com/leaderboards/)).

Associated arxiv submission: 

## Usage

Here, we explain the basic usage of our code to reproduce our results. We will keep updating it with more details. You will need pytorch, scikit-learn and h5py.

### Data and Path Preparation

The images' and labels' directory strucutre should look like this: /task3/images/HAM10000/ISIC_0024306.jpg and /task3/labels/HAM10000/labels.csv. The labels in the CSV file should be structured as follows: first column contains the image ID ("ISIC_0024306"), then the one-hot encoded labels follow.

We also considered the dataset we named "ISIC_Rest" which contains images from the official ISIC archive. The corresponding CSV file in this repositroy indicates the images that were included. 

Our split for training/validation with 5-Fold CV is included in the "indices_new.pkl" file. This should be placed in the same directory as /task3. Note that we do not use a test set.

In pc_cfgs we include an example for a machine specific cfg. Here, the base folder can be adjusted for different machines.

### Training a model

We included two example config files for full training and 5-Fold CV. More details on the different options, e.g. for balancing, are given in the paper. To start training, run: `python train.py example ISIC.example_senet_5foldcv`

### Evaluate a model 

For model evaluation, there are multiple options. First, a 5-Fold CV model can be evaluated on each held out split, optionally with a meta learner applied on top. The meta learner is evaluated by splitting the validation split once again into 10 sub splits. For evaluation, run: `python eval.py example ISIC.example_senet_5foldcv multiorder36 average /home/Gessert/data/isic/ISIC.example_senet_5foldcv best SVM` 

`multiorder36` indicate that ordered, multi-crop evaluation with 36 crops should be performed. Always use 9, 16, 25, 36, 49... etc. number of crops. `average` indicates the predictions should be averaged over the crops (can also be `vote`). `best` indicates that the best model obtained during training should be used. Can be `last` to use the last model saved. `SVM` indicates meta learning with a support vector machine. Can be `RF` for random forests instead.

If final predictions on new, unlabeled images should be performed, add the path to said images at the end of the evaluation call: `python eval.py example ISIC.example_senet_5foldcv multiorder36 average /home/Gessert/data/isic/ISIC.example_senet_5foldcv best SVC /home/Gessert/data/isic/task3/images/Validation` 

### Construct an Ensemble

Coming soon
