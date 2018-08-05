import os
import torch
#import pandas as pd
from skimage import io, transform
import scipy
import numpy as np
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.metrics import confusion_matrix, auc, roc_curve, f1_score
import math
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# Define ISIC Dataset Class
class ISICDataset(Dataset):
    """ISIC dataset."""

    def __init__(self, mdlParams, indSet):
        """
        Args:
            mdlParams (dict): Configuration for loading
            indSet (string): Indicates train, val, test
        """
        # True/False on ordered cropping for eval
        # Copy stuff from config
        self.input_size = (np.int32(mdlParams['input_size'][0]),np.int32(mdlParams['input_size'][1]))        
        self.orderedCrop = mdlParams['orderedCrop']   
        self.balancing = mdlParams['balance_classes']
        self.subtract_set_mean = mdlParams['subtract_set_mean']
        self.same_sized_crop = mdlParams['same_sized_crops']  
        self.train_eval_state = mdlParams['trainSetState']   
        # Potential setMean to deduce from channels
        self.setMean = mdlParams['setMean'].astype(np.float32)
        # Current indSet = 'trainInd'/'valInd'/'testInd'
        self.indices = mdlParams[indSet]  
        self.indSet = indSet
        # Balanced batching
        if self.balancing == 3 and indSet == 'trainInd':
            # Sample classes equally for each batch
            # First, split set by classes
            not_one_hot = np.argmax(mdlParams['labels_array'],1)
            self.class_indices = []
            for i in range(mdlParams['numClasses']):
                self.class_indices.append(np.where(not_one_hot==i)[0])
                # Kick out non-trainind indices
                self.class_indices[i] = np.setdiff1d(self.class_indices[i],mdlParams['valInd'])
            # Now sample indices equally for each batch by repeating all of them to have the same amount as the max number
            indices = []
            max_num = np.max([len(x) for x in self.class_indices])
            # Go thourgh all classes
            for i in range(mdlParams['numClasses']):
                count = 0
                class_count = 0
                max_num_curr_class = len(self.class_indices[i])
                # Add examples until we reach the maximum
                while(count < max_num):
                    # Start at the beginning, if we are through all available examples
                    if class_count == max_num_curr_class:
                        class_count = 0
                    indices.append(self.class_indices[i][class_count])
                    count += 1
                    class_count += 1
            print("Largest class",max_num,"Indices len",len(indices))
            # Set labels/inputs
            self.labels = mdlParams['labels_array'][indices,:]
            self.im_paths = np.array(mdlParams['im_paths'])[indices].tolist()     
            # Normal train proc
            if self.same_sized_crop:
                cropping = transforms.RandomCrop(self.input_size)
            else:
                cropping = transforms.RandomResizedCrop(self.input_size[0])
            # All transforms
            self.composed = transforms.Compose([
                    cropping,
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ColorJitter(brightness=32. / 255.,saturation=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(torch.from_numpy(self.setMean).float(),torch.from_numpy(np.array([1.,1.,1.])).float())
                    ])                                
        elif self.orderedCrop and (indSet == 'valInd' or self.train_eval_state  == 'eval'):
            # Complete labels array, only for current indSet, repeat for multiordercrop
            inds_rep = np.repeat(mdlParams[indSet], mdlParams['multiCropEval'])
            self.labels = mdlParams['labels_array'][inds_rep,:]
            # Path to images for loading, only for current indSet, repeat for multiordercrop
            self.im_paths = np.array(mdlParams['im_paths'])[inds_rep].tolist()
            print(len(self.im_paths))
            # Set up crop positions for every sample
            self.cropPositions = np.tile(mdlParams['cropPositions'], (mdlParams[indSet].shape[0],1))
            print("CP",self.cropPositions.shape)          
            # Set up transforms
            self.norm = transforms.Normalize(torch.from_numpy(self.setMean).float(),torch.from_numpy(np.array([1.,1.,1.])).float())
            self.trans = transforms.ToTensor()
        elif indSet == 'valInd':
            self.cropping = transforms.RandomResizedCrop(self.input_size)
            # Complete labels array, only for current indSet
            self.labels = mdlParams['labels_array'][mdlParams[indSet],:]
            # Path to images for loading, only for current indSet
            self.im_paths = np.array(mdlParams['im_paths'])[mdlParams[indSet]].tolist()            
        else:
            # Normal train proc
            if self.same_sized_crop:
                cropping = transforms.RandomCrop(self.input_size)
            else:
                cropping = transforms.RandomResizedCrop(self.input_size[0])
            # Color distortion
            if mdlParams.get('full_color_distort') is not None:
                color_distort = transforms.ColorJitter(brightness=32. / 255.,saturation=0.5, contrast = 0.5, hue = 0.2) 
            else:
                color_distort = transforms.ColorJitter(brightness=32. / 255.,saturation=0.5) 
            # All transforms
            self.composed = transforms.Compose([
                    cropping,
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    color_distort,
                    transforms.ToTensor(),
                    transforms.Normalize(torch.from_numpy(self.setMean).float(),torch.from_numpy(np.array([1.,1.,1.])).float())
                    ])                  
            # Complete labels array, only for current indSet
            self.labels = mdlParams['labels_array'][mdlParams[indSet],:]
            # Path to images for loading, only for current indSet
            self.im_paths = np.array(mdlParams['im_paths'])[mdlParams[indSet]].tolist()
    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        # Load image
        x = Image.open(self.im_paths[idx])
        # Get label
        y = self.labels[idx,:]
        # Transform data based on whether train or not train. If train, also check if its train train or train inference
        if self.orderedCrop and (self.indSet == 'valInd' or self.indSet == 'testInd' or self.train_eval_state == 'eval'):
            # Apply ordered cropping to validation or test set
            # First, to pytorch tensor (0.0-1.0)
            x = self.trans(x)
            # Normalize
            x = self.norm(x)
            # Get current crop position
            x_loc = self.cropPositions[idx,0]
            y_loc = self.cropPositions[idx,1]
            # Then, apply current crop
            x = x[:,(x_loc-np.int32(self.input_size[0]/2.)):(x_loc-np.int32(self.input_size[0]/2.))+self.input_size[0],(y_loc-np.int32(self.input_size[1]/2.)):(y_loc-np.int32(self.input_size[1]/2.))+self.input_size[1]]
        elif self.indSet == 'valInd':
            # First, to pytorch tensor (0.0-1.0)
            x = self.trans(x)
            # Normalize
            x = self.norm(x)
            x = self.cropping(x)
        else:
            # Apply
            x = self.composed(x)  
        # Transform y
        y = np.argmax(y)
        y = np.int64(y)
        return x, y, idx

def getErrClassification_mgpu(mdlParams, indices, modelVars):
    """Helper function to return the error of a set
    Args:
      mdlParams: dictionary, configuration file
      indices: string, either "trainInd", "valInd" or "testInd"
    Returns:
      loss: float, avg loss
      acc: float, accuracy
      sensitivity: float, sensitivity
      spec: float, specificity
      conf: float matrix, confusion matrix
    """
    # Set up sizes
    if indices == 'trainInd':
        numBatches = int(math.floor(len(mdlParams[indices])/mdlParams['batchSize']/len(mdlParams['numGPUs'])))
    else:
        numBatches = int(math.ceil(len(mdlParams[indices])/mdlParams['batchSize']/len(mdlParams['numGPUs'])))
    loss_all = np.zeros([numBatches])
    #allInds = np.zeros([len(mdlParams[indices])])
    predictions = np.zeros([len(mdlParams[indices]),mdlParams['numClasses']])
    targets = np.zeros([len(mdlParams[indices]),mdlParams['numClasses']])
    # Consider multi-crop case
    if 'multiCropEval' in mdlParams and mdlParams['multiCropEval'] > 0:
        loss_mc = np.zeros([len(mdlParams[indices])])
        predictions_mc = np.zeros([len(mdlParams[indices]),mdlParams['numClasses'],mdlParams['multiCropEval']])
        targets_mc = np.zeros([len(mdlParams[indices]),mdlParams['numClasses'],mdlParams['multiCropEval']])   
        for i, (inputs, labels, inds) in enumerate(modelVars['dataloader_val']):
            # Get data
            inputs = inputs.to(modelVars['device'])
            labels = labels.to(modelVars['device'])       
            # Not sure if thats necessary
            modelVars['optimizer'].zero_grad()    
            with torch.set_grad_enabled(False):
                # Get outputs
                outputs = modelVars['model'](inputs)
                preds = modelVars['softmax'](outputs)      
                # Loss
                loss = modelVars['criterion'](outputs, labels)           
            # Write into proper arrays
            loss_mc[i] = np.mean(loss.cpu().numpy())
            predictions_mc[i,:,:] = np.transpose(preds)
            tar_not_one_hot = labels.data.cpu().numpy()
            tar = np.zeros((tar_not_one_hot.shape[0], mdlParams['numClasses']))
            tar[np.arange(tar_not_one_hot.shape[0]),tar_not_one_hot] = 1
            targets_mc[i,:,:] = np.transpose(tar)
        # Targets stay the same
        targets = targets_mc[:,:,0]
        if mdlParams['voting_scheme'] == 'vote':
            # Vote for correct prediction
            print("Pred Shape",predictions_mc.shape)
            predictions_mc = np.argmax(predictions_mc,1)    
            print("Pred Shape",predictions_mc.shape) 
            for j in range(predictions_mc.shape[0]):
                predictions[j,:] = np.bincount(predictions_mc[j,:],minlength=mdlParams['numClasses'])   
            print("Pred Shape",predictions.shape) 
        elif mdlParams['voting_scheme'] == 'average':
            predictions = np.mean(predictions_mc,2)
    else:    
        for i, (inputs, labels, indices) in enumerate(modelVars['dataloader_val']):
            # Get data
            inputs = inputs.to(modelVars['device'])
            labels = labels.to(modelVars['device'])       
            # Not sure if thats necessary
            modelVars['optimizer'].zero_grad()    
            with torch.set_grad_enabled(False):
                # Get outputs
                outputs = modelVars['model'](inputs)
                preds = modelVars['softmax'](outputs)      
                # Loss
                loss = modelVars['criterion'](outputs, labels)     
            if (int(math.ceil(len(mdlParams[indices]) / mdlParams['batchSize'])) - 1 == i) and indices is not 'trainInd':
                bSize = len(mdlParams[indices]) - mdlParams['batchSize'] * i
            else:
                bSize = mdlParams['batchSize']                   
            # Write into proper arrays
            for k in range(len(mdlParams['numGPUs'])):
                loss_all[(i*len(mdlParams['numGPUs'])+k)*bSize:(i*len(mdlParams['numGPUs'])+k+1)*bSize] = loss
                predictions[(i*len(mdlParams['numGPUs'])+k)*bSize:(i*len(mdlParams['numGPUs'])+k+1)*bSize,:] = np.transpose(preds)
                targets[(i*len(mdlParams['numGPUs'])+k)*bSize:(i*len(mdlParams['numGPUs'])+k+1)*bSize,:] = labels.data
    # Calculate metrics
    # Accuarcy
    acc = np.mean(np.equal(np.argmax(predictions,1),np.argmax(targets,1)))
    # Confusion matrix
    conf = confusion_matrix(np.argmax(targets,1),np.argmax(predictions,1))
    if conf.shape[0] < mdlParams['numClasses']:
        conf = np.ones([mdlParams['numClasses'],mdlParams['numClasses']])
    # Class weighted accuracy
    wacc = conf.diagonal()/conf.sum(axis=1)    
    # Sensitivity / Specificity
    sensitivity = np.zeros([mdlParams['numClasses']])
    specificity = np.zeros([mdlParams['numClasses']])
    if mdlParams['numClasses'] > 2:
        for k in range(mdlParams['numClasses']):
                sensitivity[k] = conf[k,k]/(np.sum(conf[k,:]))
                true_negative = np.delete(conf,[k],0)
                true_negative = np.delete(true_negative,[k],1)
                true_negative = np.sum(true_negative)
                false_positive = np.delete(conf,[k],0)
                false_positive = np.sum(false_positive[:,k])
                specificity[k] = true_negative/(true_negative+false_positive)
                # F1 score
                f1 = f1_score(np.argmax(predictions,1),np.argmax(targets,1),average='weighted')                
    else:
        tn, fp, fn, tp = confusion_matrix(np.argmax(targets,1),np.argmax(predictions,1)).ravel()
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)
        # F1 score
        f1 = f1_score(np.argmax(predictions,1),np.argmax(targets,1))
    # AUC
    fpr = {}
    tpr = {}
    roc_auc = np.zeros([mdlParams['numClasses']])
    for i in range(mdlParams['numClasses']):
        fpr[i], tpr[i], _ = roc_curve(targets[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return np.mean(loss_all), acc, sensitivity, specificity, conf, f1, roc_auc, wacc, predictions, targets, predictions_mc 


def learn_on_predictions(mdlParams, modelVars, pred_val, tar_val, split=None, cvsize=10, pred_test=None):
    """Helper function to return the error of a set
    Args:
      mdlParams: dictionary, configuration file
      indices: string, either "trainInd", "valInd" or "testInd"
    Returns:
      loss: float, avg loss
      acc: float, accuracy
      sensitivity: float, sensitivity
      spec: float, specificity
      conf: float matrix, confusion matrix
    """     
    # Learn on predictions
    # Split into internal train/val if desired
    if split is not None:
        acc = 0
        wacc = 0
        wacc_avg = 0
        roc_auc = 0
        sensitivity = 0
        specificity = 0
        f1 = 0                         
        for i in range(cvsize):
            inds = np.arange(len(pred_val))
            np.random.shuffle(inds)
            train_inds = inds[split:]
            val_inds = inds[:split]
            # Reshape
            feat_train = np.reshape(pred_val[train_inds,:,:],[len(train_inds),mdlParams['multiCropEval']*mdlParams['numClasses']])#np.reshape(np.transpose(predictions_mc_t,[0,2,1]),[len(mdlParams['trainInd'])*mdlParams['multiCropEval'],mdlParams['numClasses']])
            tar_train = tar_val[train_inds,:]#np.reshape(np.transpose(targets_mc_t,[0,2,1]),[len(mdlParams['trainInd'])*mdlParams['multiCropEval'],mdlParams['numClasses']])
            feat_val = np.reshape(pred_val[val_inds,:,:],[len(val_inds),mdlParams['multiCropEval']*mdlParams['numClasses']])#np.reshape(np.transpose(predictions_mc_v,[0,2,1]),[len(mdlParams['valInd'])*mdlParams['multiCropEval'],mdlParams['numClasses']])
            # Actual targets for evaluation
            targets = tar_val[val_inds,:]#np.reshape(np.transpose(targets_mc_v,[0,2,1]),[len(mdlParams['valInd'])*mdlParams['multiCropEval'],mdlParams['numClasses']])#
            # Train SVM/RF
            if 'RF' in mdlParams['meta_learner']:
                clf = RandomForestClassifier(n_estimators = 100,class_weight='balanced') 
            elif 'SVM' in mdlParams['meta_learner']:
                clf = SVC(kernel='linear',class_weight='balanced',C=1.0)                 
            # Same for all sklearn classifiers
            tar_train_not_one_hot = np.argmax(tar_train,1)
            clf.fit(feat_train, tar_train_not_one_hot)
            predictions_not_one_hot = clf.predict(feat_val)
            predictions = np.zeros([len(predictions_not_one_hot),mdlParams['numClasses']])
            predictions[np.arange(len(predictions_not_one_hot)), predictions_not_one_hot] = 1        
            #print("Train score",clf.score(feat_train, tar_train_not_one_hot))
            # For comparison:
            conf_avg = confusion_matrix(np.argmax(targets,1),np.argmax(np.mean(pred_val[val_inds,:,:],2),1),labels=np.array(np.arange(mdlParams['numClasses'])))
            #print("conf",conf_avg)
            wacc_avg += conf_avg.diagonal()/conf_avg.sum(axis=1)
            # Calculate metrics
            # Accuarcy
            acc += np.mean(np.equal(np.argmax(predictions,1),np.argmax(targets,1)))
            # Confusion matrix
            conf = confusion_matrix(np.argmax(targets,1),np.argmax(predictions,1),labels=np.array(np.arange(mdlParams['numClasses'])))
            #print("Conf",conf)
            # Class weighted accuracy
            wacc += conf.diagonal()/conf.sum(axis=1)    
            # Sensitivity / Specificity
            curr_sensitivity = np.zeros([mdlParams['numClasses']])
            curr_specificity = np.zeros([mdlParams['numClasses']])
            if mdlParams['numClasses'] > 2:
                for k in range(mdlParams['numClasses']):
                        curr_sensitivity[k] = conf[k,k]/(np.sum(conf[k,:]))
                        true_negative = np.delete(conf,[k],0)
                        true_negative = np.delete(true_negative,[k],1)
                        true_negative = np.sum(true_negative)
                        false_positive = np.delete(conf,[k],0)
                        false_positive = np.sum(false_positive[:,k])
                        curr_specificity[k] = true_negative/(true_negative+false_positive)
                        # F1 score
                f1 += f1_score(np.argmax(predictions,1),np.argmax(targets,1),average='weighted')
                sensitivity += curr_sensitivity
                specificity += curr_specificity                
            else:
                tn, fp, fn, tp = confusion_matrix(np.argmax(targets,1),np.argmax(predictions,1)).ravel()
                sensitivity += tp/(tp+fn)
                specificity += tn/(tn+fp)
                # F1 score
                f1 += f1_score(np.argmax(predictions,1),np.argmax(targets,1))
            # AUC
            fpr = {}
            tpr = {}
            curr_roc_auc = np.zeros([mdlParams['numClasses']])
            for i in range(mdlParams['numClasses']):
                fpr[i], tpr[i], _ = roc_curve(targets[:, i], predictions[:, i])
                curr_roc_auc[i] = auc(fpr[i], tpr[i])
            roc_auc += curr_roc_auc
        print(mdlParams['meta_learner'],"Average WACC",np.mean(wacc_avg/cvsize),"Meta WACC",np.mean(wacc/cvsize))
        return acc/cvsize, sensitivity/cvsize, specificity/cvsize, conf, f1/cvsize, roc_auc/cvsize, wacc/cvsize
    else:
        # Reshape
        feat_train = np.reshape(pred_val,[len(pred_val),mdlParams['multiCropEval']*mdlParams['numClasses']])#np.reshape(np.transpose(predictions_mc_t,[0,2,1]),[len(mdlParams['trainInd'])*mdlParams['multiCropEval'],mdlParams['numClasses']])
        tar_train = tar_val#np.reshape(np.transpose(targets_mc_t,[0,2,1]),[len(mdlParams['trainInd'])*mdlParams['multiCropEval'],mdlParams['numClasses']])
        feat_val = np.reshape(pred_test,[len(pred_test),mdlParams['multiCropEval']*mdlParams['numClasses']])#np.reshape(np.transpose(predictions_mc_v,[0,2,1]),[len(mdlParams['valInd'])*mdlParams['multiCropEval'],mdlParams['numClasses']])
        # Actual targets for evaluation does not exist here
        #targets = tar_val[val_inds,:]#np.reshape(np.transpose(targets_mc_v,[0,2,1]),[len(mdlParams['valInd'])*mdlParams['multiCropEval'],mdlParams['numClasses']])#
        # Train SVM/RF
        if 'RF' in mdlParams['meta_learner']:
            clf = RandomForestClassifier(n_estimators = 100)#class_weight='balanced') 
        elif 'SVC' in mdlParams['meta_learner']:
            clf = SVC(kernel='rbf',class_weight='balanced', C=1)
        # Same for all sklearn classifiers
        tar_train_not_one_hot = np.argmax(tar_train,1)
        clf.fit(feat_train, tar_train_not_one_hot)
        predictions_not_one_hot = clf.predict(feat_val)
        predictions = np.zeros([len(predictions_not_one_hot),mdlParams['numClasses']])
        predictions[np.arange(len(predictions_not_one_hot)), predictions_not_one_hot] = 1        
        print("Train score",clf.score(feat_train, tar_train_not_one_hot))
        return predictions


class Nadam(torch.optim.Optimizer):
    """Implements Nadam algorithm (a variant of Adam based on Nesterov momentum).

    It has been proposed in `Incorporating Nesterov Momentum into Adam`__.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 2e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        schedule_decay (float, optional): momentum schedule decay (default: 4e-3)

    __ http://cs229.stanford.edu/proj2015/054_report.pdf
    __ http://www.cs.toronto.edu/~fritz/absps/momentum.pdf
    """

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, schedule_decay=4e-3):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, schedule_decay=schedule_decay)
        super(Nadam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m_schedule'] = 1.
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                # Warming momentum schedule
                m_schedule = state['m_schedule']
                schedule_decay = group['schedule_decay']
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                eps = group['eps']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                momentum_cache_t = beta1 * \
                    (1. - 0.5 * (0.96 ** (state['step'] * schedule_decay)))
                momentum_cache_t_1 = beta1 * \
                    (1. - 0.5 *
                     (0.96 ** ((state['step'] + 1) * schedule_decay)))
                m_schedule_new = m_schedule * momentum_cache_t
                m_schedule_next = m_schedule * momentum_cache_t * momentum_cache_t_1
                state['m_schedule'] = m_schedule_new

                # Decay the first and second moment running average coefficient
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg_sq_prime = exp_avg_sq.div(1. - bias_correction2)

                denom = exp_avg_sq_prime.sqrt_().add_(group['eps'])

                p.data.addcdiv_(-group['lr'] * (1. - momentum_cache_t) / (1. - m_schedule_new), grad, denom)
                p.data.addcdiv_(-group['lr'] * momentum_cache_t_1 / (1. - m_schedule_next), exp_avg, denom)

        return loss