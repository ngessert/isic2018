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

You can swap out models by using the names given in models.py. Note, that for polynet, the input size needs to be changed to 331x331 instead of 224x224.

### Training a model

We included two example config files for full training and 5-Fold CV. More details on the different options, e.g. for balancing, are given in the paper. To start training, run: `python train.py example ISIC.example_senet_5foldcv`

### Evaluate a model 

For model evaluation, there are multiple options. First, a 5-Fold CV model can be evaluated on each held out split, optionally with a meta learner applied on top. The meta learner is evaluated by splitting the validation split once again into 10 sub splits. For evaluation, run: `python eval.py example ISIC.example_senet_5foldcv multiorder36 average /home/Gessert/data/isic/ISIC.example_senet_5foldcv best SVM` 

`multiorder36` indicate that ordered, multi-crop evaluation with 36 crops should be performed. Always use 9, 16, 25, 36, 49... etc. number of crops. `average` indicates the predictions should be averaged over the crops (can also be `vote`). `best` indicates that the best model obtained during training should be used. Can be `last` to use the last model saved. `SVM` indicates meta learning with a support vector machine. Can be `RF` for random forests instead.

If final predictions on new, unlabeled images should be performed, add the path to said images at the end of the evaluation call: `python eval.py example ISIC.example_senet_5foldcv multiorder36 average /home/Gessert/data/isic/ISIC.example_senet_5foldcv best SVM /home/Gessert/data/isic/task3/images/Validation` 

Each evaluation run generates a pkl file that can be used for further ensemble aggregation.

### Construct an Ensemble

Testing ensembles is also split into two parts. First, an ensemble can be constructed based on 5-Fold CV error and the corresponding best models are saved. Then, the final predictions on a new dataset can be made using the generated files from the evaluation section.

For 5-Fold CV performance assessment, run: `python ensemble.py /path/to/evaluation/files evalexhaust15 /path/to/file/best_models.pkl`
The first path indicates the location where all evaluation pkl files are located. `evalexhaust15`: `eval` indicates that 5-Fold CV evaluation is desired. `exhaust15` indicates that the top 15 performing models should be tested for their optimal combination. I.e., every possible combination (average predictions) of those models is tested for the best performance. Without the exhaust option, only the top N combinations are considered, i.e., the tested combinations are: top1 model, top1+top2 model, top1+top2+top3 model, etc. The last argument indicates the path where the best performing combination is saved.

For generation of new predictions for unlabeled data, run: `python ensemble.py /path/to/evaluation/files best /path/to/file/best_models.pkl /path/to/predictions.csv /path/to/image/files`
`best` indicates that only the models with best in the name should be considered. This relates to the evaluation where either the best performing model or the last checkpoint can be used for generation. This can be `last` or `bestlast` for both. The next argument is the path to the file that was generated in the first ensemble run. This can just be `NONE` if all models should be included. The next argument is the path to the CSV file that should contain the predictions. The last argument is the path to the image files which is used to match the predictions to image file names.
