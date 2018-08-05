import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models as tv_models
from torch.utils.data import DataLoader
from torchsummary import summary
import numpy as np
import models
import threading
import pickle
from pathlib import Path
import math
import os
import sys
from glob import glob
import re
import gc
import importlib
import time
import sklearn.preprocessing
import utils
from sklearn.utils import class_weight

# add configuration file
# Dictionary for model configuration
mdlParams = {}

# Import machine config
pc_cfg = importlib.import_module('pc_cfgs.'+sys.argv[1])
mdlParams.update(pc_cfg.mdlParams)

# Path name where model is saved is the fourth argument
mdlParams['saveDirBase'] = sys.argv[5]

# Which checkpoint should be used, either "best" or "last"
# Also, if there are 2 checkpoints, use "first" to select the earlier one
if len(sys.argv) > 6:
    if 'last' in sys.argv[6]:
        mdlParams['ckpt_name'] = 'checkpoint-'
    else:
        mdlParams['ckpt_name'] = 'checkpoint_best-'
    if 'first' in sys.argv[6]:
        mdlParams['use_first'] = True
else:
    mdlParams['ckpt_name'] = 'checkpoint-'

# Use meta learning?
if len(sys.argv) > 7:
    if 'SVC' in sys.argv[7] or 'RF' in sys.argv[7]:
        mdlParams['learn_on_preds'] = True
        mdlParams['meta_learner'] = sys.argv[7]
    else:
        mdlParams['learn_on_preds'] = False
else:
    mdlParams['learn_on_preds'] = False

# Import model config
model_cfg = importlib.import_module('cfgs.'+sys.argv[2])
mdlParams_model = model_cfg.init(mdlParams)
mdlParams.update(mdlParams_model)

# Third is multi crop yes no
if 'multi' in sys.argv[3]:
    mdlParams['multiCropEval'] = [int(s) for s in re.findall(r'\d+',sys.argv[3])][-1]
    mdlParams['voting_scheme'] = sys.argv[4]
    if 'scale' in sys.argv[3]:
        print("Multi Crop and Scale Eval with crop number:",mdlParams['multiCropEval']," Voting scheme: ",mdlParams['voting_scheme'])
        mdlParams['multiScaleEval'] = True
    elif 'order' in sys.argv[3]:
        mdlParams['orderedCrop'] = True
        # Crop positions, always choose multiCropEval to be 4, 9, 16, 25, etc.
        mdlParams['cropPositions'] = np.zeros([mdlParams['multiCropEval'],2],dtype=np.int64)
        ind = 0
        for i in range(np.int32(np.sqrt(mdlParams['multiCropEval']))):
            for j in range(np.int32(np.sqrt(mdlParams['multiCropEval']))):
                mdlParams['cropPositions'][ind,0] = mdlParams['input_size'][0]/2+i*((mdlParams['input_size_load'][0]-mdlParams['input_size'][0])/(np.sqrt(mdlParams['multiCropEval'])-1))
                mdlParams['cropPositions'][ind,1] = mdlParams['input_size'][1]/2+j*((mdlParams['input_size_load'][1]-mdlParams['input_size'][1])/(np.sqrt(mdlParams['multiCropEval'])-1))
                ind += 1
        # Sanity checks
        print("Positions",mdlParams['cropPositions'])
        # Test image sizes
        test_im = np.zeros(mdlParams['input_size_load'])
        height = mdlParams['input_size'][0]
        width = mdlParams['input_size'][1]
        for i in range(mdlParams['multiCropEval']):
            im_crop = test_im[np.int32(mdlParams['cropPositions'][i,0]-height/2):np.int32(mdlParams['cropPositions'][i,0]-height/2)+height,np.int32(mdlParams['cropPositions'][i,1]-width/2):np.int32(mdlParams['cropPositions'][i,1]-width/2)+width,:]
            print("Shape",i+1,im_crop.shape)       
        print("Multi Crop with order with crop number:",mdlParams['multiCropEval']," Voting scheme: ",mdlParams['voting_scheme'])
    else:
        print("Multi Crop Eval with crop number:",mdlParams['multiCropEval']," Voting scheme: ",mdlParams['voting_scheme'])
        mdlParams['orderedCrop'] = False
else:
    mdlParams['multiCropEval'] = 0

# Set training set to eval mode
mdlParams['trainSetState'] = 'eval'

# Scaler, scales targets to a range of 0-1
if mdlParams['scale_targets']:
    mdlParams['scaler'] = sklearn.preprocessing.MinMaxScaler().fit(mdlParams['targets'][mdlParams['trainInd'],:][:,mdlParams['tar_range'].astype(int)])

# Save results in here
allData = {}
allData['f1Best'] = {}
allData['sensBest'] = {}
allData['specBest'] = {}
allData['accBest'] = {}
allData['waccBest'] = {}
allData['aucBest'] = {}
allData['convergeTime'] = {}
allData['bestPred'] = {}
allData['bestPredMC'] = {}
allData['targets'] = {}
allData['extPred'] = {}
allData['f1Best_meta'] = {}
allData['sensBest_meta'] = {}
allData['specBest_meta'] = {}
allData['accBest_meta'] = {}
allData['waccBest_meta'] = {}
allData['aucBest_meta'] = {}
#allData['convergeTime'] = {}
allData['bestPred_meta'] = {}
allData['targets_meta'] = {}

f1Best = 0
sensBest = 0
specBest = 0
accBest = 0
allaccBest = 0
waccBest = 0
aucBest = 0
maucBest = 0
f1Best_meta = 0
sensBest_meta = 0
specBest_meta = 0
accBest_meta = 0
allaccBest_meta = 0
waccBest_meta = 0
aucBest_meta = 0
maucBest_meta = 0

for cv in range(mdlParams['numCV']):
    # Reset model graph 
    importlib.reload(models)
    #importlib.reload(torchvision)
    # Collect model variables
    modelVars = {}
    modelVars['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(modelVars['device'])
    # Def current CV set
    mdlParams['trainInd'] = mdlParams['trainIndCV'][cv]
    if 'valIndCV' in mdlParams:
        mdlParams['valInd'] = mdlParams['valIndCV'][cv]
    # Def current path for saving stuff
    if 'valIndCV' in mdlParams:
        mdlParams['saveDir'] = mdlParams['saveDirBase'] + '/CVSet' + str(cv)
    else:
        mdlParams['saveDir'] = mdlParams['saveDirBase']

    # Potentially calculate setMean to subtract
    if mdlParams['subtract_set_mean'] == 1:
        mdlParams['setMean'] = np.mean(mdlParams['images_means'][mdlParams['trainInd'],:],(0))
        print("Set Mean",mdlParams['setMean']) 

    # balance classes
    if mdlParams['balance_classes'] < 3 or mdlParams['balance_classes'] == 7:
        class_weights = class_weight.compute_class_weight('balanced',np.unique(np.argmax(mdlParams['labels_array'][mdlParams['trainInd'],:],1)),np.argmax(mdlParams['labels_array'][mdlParams['trainInd'],:],1)) 
        print("Current class weights",class_weights)
        class_weights = class_weights*mdlParams['extra_fac']
        print("Current class weights with extra",class_weights)             
    elif mdlParams['balance_classes'] == 3 or mdlParams['balance_classes'] == 4:
        # Split training set by classes
        not_one_hot = np.argmax(mdlParams['labels_array'],1)
        mdlParams['class_indices'] = []
        for i in range(mdlParams['numClasses']):
            mdlParams['class_indices'].append(np.where(not_one_hot==i)[0])
            # Kick out non-trainind indices
            mdlParams['class_indices'][i] = np.setdiff1d(mdlParams['class_indices'][i],mdlParams['valInd'])
            #print("Class",i,mdlParams['class_indices'][i].shape,np.min(mdlParams['class_indices'][i]),np.max(mdlParams['class_indices'][i]),np.sum(mdlParams['labels_array'][np.int64(mdlParams['class_indices'][i]),:],0))        
    elif mdlParams['balance_classes'] == 5 or mdlParams['balance_classes'] == 6:
        # Other class balancing loss
        class_weights = 1.0/np.mean(mdlParams['labels_array'][mdlParams['trainInd'],:],axis=0)
        print("Current class weights",class_weights) 
        class_weights = class_weights*mdlParams['extra_fac']
        print("Current class weights with extra",class_weights) 
    elif mdlParams['balance_classes'] == 9:
        # Only use HAM indicies for calculation
        indices_ham = mdlParams['trainInd'][mdlParams['trainInd'] < 10015]
        class_weights = 1.0/np.mean(mdlParams['labels_array'][indices_ham,:],axis=0)
        print("Current class weights",class_weights) 

    # Set up dataloaders
    # For train
    dataset_train = utils.ISICDataset(mdlParams, 'trainInd')
    modelVars['dataloader_train'] = DataLoader(dataset_train, batch_size=mdlParams['batchSize'], shuffle=True, num_workers=8, pin_memory=True)
    # For val
    dataset_val = utils.ISICDataset(mdlParams, 'valInd')
    modelVars['dataloader_val'] = DataLoader(dataset_val, batch_size=mdlParams['multiCropEval'], shuffle=False, num_workers=8, pin_memory=True)
    #print("Setdiff",np.setdiff1d(mdlParams['trainInd'],mdlParams['trainInd']))
    # Define model 
    modelVars['model'] = models.getModel(mdlParams['model_type'])()
    #print(modelVars['model'])
    if 'Dense' in mdlParams['model_type']:
        num_ftrs = modelVars['model'].classifier.in_features
        modelVars['model'].classifier = nn.Linear(num_ftrs, mdlParams['numClasses'])
        print(modelVars['model'])
    elif 'dpn' in mdlParams['model_type']:
        num_ftrs = modelVars['model'].classifier.in_channels
        modelVars['model'].classifier = nn.Conv2d(num_ftrs,mdlParams['numClasses'],[1,1])
        #modelVars['model'].add_module('real_classifier',nn.Linear(num_ftrs, mdlParams['numClasses']))
        print(modelVars['model'])
    else:
        num_ftrs = modelVars['model'].last_linear.in_features
        modelVars['model'].last_linear = nn.Linear(num_ftrs, mdlParams['numClasses'])      
    # multi gpu support
    if len(mdlParams['numGPUs']) > 1:
        modelVars['model'] = nn.DataParallel(modelVars['model'])
    modelVars['model']  = modelVars['model'].to(modelVars['device'])
    #summary(modelVars['model'], (mdlParams['input_size'][2], mdlParams['input_size'][0], mdlParams['input_size'][1]))
    # Loss, with class weighting
    if mdlParams['balance_classes'] == 3 or mdlParams['balance_classes'] == 0:
        modelVars['criterion'] = nn.CrossEntropyLoss()
    elif mdlParams['balance_classes'] == 8:
        modelVars['criterion'] = nn.CrossEntropyLoss(reduce=False)
    elif mdlParams['balance_classes'] == 6 or mdlParams['balance_classes'] == 7:
        modelVars['criterion'] = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor(class_weights.astype(np.float32)),reduce=False)
    else:
        modelVars['criterion'] = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor(class_weights.astype(np.float32)))

    # Observe that all parameters are being optimized
    modelVars['optimizer'] = optim.Adam(modelVars['model'].parameters(), lr=mdlParams['learning_rate'])

    # Decay LR by a factor of 0.1 every 7 epochs
    modelVars['scheduler'] = lr_scheduler.StepLR(modelVars['optimizer'], step_size=mdlParams['lowerLRAfter'], gamma=1/np.float32(mdlParams['LRstep']))

    # Define softmax
    modelVars['softmax'] = nn.Softmax(dim=1)

    # Manually find latest chekcpoint, tf.train.latest_checkpoint is doing weird shit
    files = glob(mdlParams['saveDir']+'/*')
    #print("Files",files)
    global_steps = np.zeros([len(files)])
    for i in range(len(files)):
        # Use meta files to find the highest index
        if 'checkpoint' not in files[i]:
            continue
        if mdlParams['ckpt_name'] not in files[i]:
            continue
        # Extract global step
        nums = [int(s) for s in re.findall(r'\d+',files[i])]
        global_steps[i] = nums[-1]
    # Create path with maximum global step found, if first is not wanted
    global_steps = np.sort(global_steps)
    if mdlParams.get('use_first') is not None:
        chkPath = mdlParams['saveDir'] + '/' + mdlParams['ckpt_name'] + str(int(global_steps[-2])) + '.pt'
    else:
        chkPath = mdlParams['saveDir'] + '/' + mdlParams['ckpt_name'] + str(int(np.max(global_steps))) + '.pt'
    print("Restoring: ",chkPath)
    # Load
    state = torch.load(chkPath)
    # Initialize model and optimizer
    modelVars['model'].load_state_dict(state['state_dict'])
    modelVars['optimizer'].load_state_dict(state['optimizer'])   
    # Construct pkl filename: config name, last/best, saved epoch number
    pklFileName = sys.argv[2] + "_" + sys.argv[6] + "_" + str(int(np.max(global_steps))) + ".pkl"
    modelVars['model'].eval()
    if mdlParams['classification']:
        print("CV Set ",cv+1)
        print("------------------------------------")           
        if 'valInd' in mdlParams and (len(sys.argv) <= 8 or mdlParams['learn_on_preds']):
            if len(sys.argv) > 8:
                allFiles = sorted(glob(mdlParams['saveDirBase'] + "/*"))
                found = False
                for fileName in allFiles:
                    if ".pkl" in fileName and sys.argv[6] in fileName:
                        with open(fileName, 'rb') as f:
                            allData_new = pickle.load(f)  
                        if 'bestPredMC' in allData_new:
                            allData = allData_new
                            print("Val predictions for learning are there, continue to prediction on unlabeled data")
                            found = True
                            break
                if found:
                    break
                else:
                    print("No exisiting file with val predictions, evaluating again")
            loss, accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy, predictions, targets, predictions_mc = utils.getErrClassification_mgpu(mdlParams, 'valInd', modelVars)
            print("Training Results:")
            print("----------------------------------")
            print("Loss",np.mean(loss))
            print("F1 Score",f1)            
            print("Sensitivity",sensitivity)
            print("Specificity",specificity)
            print("Accuracy",accuracy)
            print("Per Class Accuracy",waccuracy)
            print("Weighted Accuracy",np.mean(waccuracy))
            print("AUC",auc)
            print("Mean AUC", np.mean(auc))  
            # Save results in dict
            allData['f1Best'][cv] = f1
            f1Best += f1
            allData['sensBest'][cv] = sensitivity
            sensBest += sensitivity
            allData['specBest'][cv] = specificity
            specBest += specificity
            allData['accBest'][cv] = accuracy
            accBest += accuracy
            allData['waccBest'][cv] = waccuracy
            allaccBest += waccuracy
            waccBest += np.mean(waccuracy)
            allData['aucBest'][cv] = auc  
            aucBest += auc
            maucBest += np.mean(auc)
            allData['bestPred'][cv] = predictions
            allData['bestPredMC'][cv] = predictions_mc
            allData['targets'][cv] = targets 
            print("Pred shape",predictions.shape,"Tar shape",targets.shape)
            # Learn on ordered multi-crop results validation -> validation
            if mdlParams['learn_on_preds']:
                accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy = utils.learn_on_predictions(mdlParams, modelVars, allData['bestPredMC'][cv], allData['targets'][cv], split=400)
                print("Training Results (learn on pred):")
                print(mdlParams['meta_learner'])
                print("----------------------------------")
                print("F1 Score",f1)            
                print("Sensitivity",sensitivity)
                print("Specificity",specificity)
                print("Accuracy",accuracy)
                print("Per Class Accuracy",waccuracy)
                print("Weighted Accuracy",np.mean(waccuracy))
                print("AUC",auc)
                print("Mean AUC", np.mean(auc))  
                # Save results in dict
                allData['f1Best_meta'][cv] = f1
                f1Best_meta += f1
                allData['sensBest_meta'][cv] = sensitivity
                sensBest_meta += sensitivity
                allData['specBest_meta'][cv] = specificity
                specBest_meta += specificity
                allData['accBest_meta'][cv] = accuracy
                accBest_meta += accuracy
                allData['waccBest_meta'][cv] = waccuracy
                allaccBest_meta += waccuracy
                waccBest_meta += np.mean(waccuracy)
                allData['aucBest_meta'][cv] = auc  
                aucBest_meta += auc
                maucBest_meta += np.mean(auc)
                allData['bestPred_meta'][cv] = predictions
                allData['targets_meta'][cv] = targets              
        if 'testInd' in mdlParams:        
            loss, accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy, predictions, targets, predictions_mc = utils.getErrClassification_mgpu(mdlParams, 'testInd', modelVars)
            print("Training Results:")
            print("----------------------------------")
            print("Loss",np.mean(loss))
            print("Accuracy",accuracy)
            print("Sensitivity",sensitivity)
            print("Specificity",specificity)
            print("F1 Score",f1)
            print("AUC",auc)
            print("Mean AUC", np.mean(auc))
            print("Per Class Accuracy",waccuracy)
            print("Weighted Accuracy",waccuracy) 
    else:
        # TODO: Regression
        print("Not Implemented")            
# If there is an 8th argument, make extra evaluation for external set
if len(sys.argv) > 8:
    for cv in range(mdlParams['numCV']):
            # Reset model graph 
            importlib.reload(models)
            #importlib.reload(torchvision)
            # Collect model variables
            modelVars = {}
            modelVars['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        
            print("Creating predictions for path ",sys.argv[8])
            # Define the path
            path1 = sys.argv[8]
            # All files in that set
            files = sorted(glob(path1+'/*'))
            # Define new paths
            mdlParams['im_paths'] = []
            for j in range(len(files)):
                inds = [int(s) for s in re.findall(r'\d+',files[j])]
                if 'ISIC_' in files[j]:
                    mdlParams['im_paths'].append(files[j])
            # Add empty labels
            mdlParams['labels_array'] = np.zeros([len(mdlParams['im_paths']),mdlParams['numClasses']],dtype=np.float32)
            # Define everything as a valind set
            mdlParams['valInd'] = np.array(np.arange(len(mdlParams['im_paths'])))
            mdlParams['trainInd'] = mdlParams['valInd']
            mdlParams['saveDir'] = mdlParams['saveDirBase'] + '/CVSet' + str(cv)

            # Set up dataloaders
            # For train
            dataset_train = utils.ISICDataset(mdlParams, 'trainInd')
            modelVars['dataloader_train'] = DataLoader(dataset_train, batch_size=mdlParams['batchSize'], shuffle=True, num_workers=8, pin_memory=True)
            # For val
            dataset_val = utils.ISICDataset(mdlParams, 'valInd')
            modelVars['dataloader_val'] = DataLoader(dataset_val, batch_size=mdlParams['multiCropEval'], shuffle=False, num_workers=8, pin_memory=True)
            #print("Setdiff",np.setdiff1d(mdlParams['trainInd'],mdlParams['trainInd']))
            # Define model 
            modelVars['model'] = models.getModel(mdlParams['model_type'])()
            #print(modelVars['model'])
            if 'Dense' in mdlParams['model_type']:
                num_ftrs = modelVars['model'].classifier.in_features
                modelVars['model'].classifier = nn.Linear(num_ftrs, mdlParams['numClasses'])
            else:
                num_ftrs = modelVars['model'].last_linear.in_features
                modelVars['model'].last_linear = nn.Linear(num_ftrs, mdlParams['numClasses'])
            # multi gpu support
            if len(mdlParams['numGPUs']) > 1:
                modelVars['model'] = nn.DataParallel(modelVars['model'])
            modelVars['model']  = modelVars['model'].to(modelVars['device'])
            #summary(modelVars['model'], (mdlParams['input_size'][2], mdlParams['input_size'][0], mdlParams['input_size'][1]))
            # Loss, with class weighting
            modelVars['criterion'] = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor(class_weights.astype(np.float32)))

            # Observe that all parameters are being optimized
            modelVars['optimizer'] = optim.Adam(modelVars['model'].parameters(), lr=mdlParams['learning_rate'])

            # Decay LR by a factor of 0.1 every 7 epochs
            modelVars['scheduler'] = lr_scheduler.StepLR(modelVars['optimizer'], step_size=mdlParams['lowerLRAfter'], gamma=1/np.float32(mdlParams['LRstep']))

            # Define softmax
            modelVars['softmax'] = nn.Softmax(dim=1)

            # Manually find latest chekcpoint, tf.train.latest_checkpoint is doing weird shit
            files = glob(mdlParams['saveDir']+'/*')
            global_steps = np.zeros([len(files)])
            for i in range(len(files)):
                # Use meta files to find the highest index
                if 'checkpoint' not in files[i]:
                    continue
                if mdlParams['ckpt_name'] not in files[i]:
                    continue
                # Extract global step
                nums = [int(s) for s in re.findall(r'\d+',files[i])]
                global_steps[i] = nums[-1]
            # Create path with maximum global step found, if first is not wanted
            global_steps = np.sort(global_steps)
            if mdlParams.get('use_first') is not None:
                chkPath = mdlParams['saveDir'] + '/' + mdlParams['ckpt_name'] + str(int(global_steps[-2])) + '.pt'
            else:
                chkPath = mdlParams['saveDir'] + '/' + mdlParams['ckpt_name'] + str(int(np.max(global_steps))) + '.pt'
            print("Restoring: ",chkPath)
            
            # Load
            state = torch.load(chkPath)
            # Initialize model and optimizer
            modelVars['model'].load_state_dict(state['state_dict'])
            modelVars['optimizer'].load_state_dict(state['optimizer'])  
            # Get predictions or learn on pred
            modelVars['model'].eval()
            # Get predictions
            loss, accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy, predictions, targets, predictions_mc = utils.getErrClassification_mgpu(mdlParams, 'valInd', modelVars)            
            if mdlParams['learn_on_preds']:
                # Meta learn
                allData['extPred'][cv] = utils.learn_on_predictions(mdlParams, modelVars, allData['bestPredMC'][cv], allData['targets'][cv], split=None, pred_test=predictions_mc)
                pklFileName = sys.argv[2] + "_" + sys.argv[6] + "_" + str(int(np.max(global_steps))) + "_predm.pkl"
            else:
                # Save predictions            
                allData['extPred'][cv] = predictions
                pklFileName = sys.argv[2] + "_" + sys.argv[6] + "_" + str(int(np.max(global_steps))) + "_predn.pkl"

# Mean results over all folds
print("-------------------------------------------------")
print("Mean over all Folds")
print("-------------------------------------------------")
print("F1 Score",f1Best/float(mdlParams['numCV']))       
print("Sensitivtiy",sensBest/float(mdlParams['numCV']))  
print("Specificity",specBest/float(mdlParams['numCV']))  
print("Accuracy",accBest/float(mdlParams['numCV']))  
print("Per Class Accuracy",allaccBest/float(mdlParams['numCV']))
print("Weighted Accuracy",waccBest/float(mdlParams['numCV'])) 
print("AUC",aucBest/float(mdlParams['numCV']))    
print("Mean AUC",maucBest/float(mdlParams['numCV']))      
# Perhaps print results for meta learning
if mdlParams['learn_on_preds']:   
    print("-------------------------------------------------")
    print("Mean over all Folds (meta learning)")
    print("-------------------------------------------------")
    print("F1 Score",f1Best_meta/float(mdlParams['numCV']))       
    print("Sensitivtiy",sensBest_meta/float(mdlParams['numCV']))  
    print("Specificity",specBest_meta/float(mdlParams['numCV']))  
    print("Accuracy",accBest_meta/float(mdlParams['numCV']))  
    print("Per Class Accuracy",allaccBest_meta/float(mdlParams['numCV']))
    print("Weighted Accuracy",waccBest_meta/float(mdlParams['numCV'])) 
    print("AUC",aucBest_meta/float(mdlParams['numCV']))    
    print("Mean AUC",maucBest_meta/float(mdlParams['numCV']))    
# Save dict with results
with open(mdlParams['saveDirBase'] + "/" + pklFileName, 'wb') as f:
    pickle.dump(allData, f, pickle.HIGHEST_PROTOCOL)              