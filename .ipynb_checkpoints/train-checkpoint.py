import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

import time
import math
import shutil
import sys, os

from attacks import Attacks
from model import WideResNet
from fullmodel import FullDenoiser
# from model import WideResNet
# from denoiser import Denoiser
from processor import Preprocessor, AverageMeter, accuracy

TRAIN_AND_TEST = 0
TEST = 1

RAW = 0
ADV = 1
BOTH = 2

class Classifier:
    
    def __init__(self, ds_name, ds_path, lr, iterations, batch_size, print_freq, k, eps, is_normalized,
                 adv_momentum, store_adv=None, load_adv_dir=None, load_adv_name=None, load_dir=None,
                 load_name=None, denoiser_dir=None, denoiser_name=None, save_dir=None):
        
        self.data_processor = Preprocessor(ds_name, ds_path, is_normalized)
        
        # Load Data
        self.train_data, self.test_data, self.N_train, self.N_test = self.data_processor.datasets()
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size)
        
        # Other Variables
        self.save_dir = save_dir
        self.store_adv = store_adv
#         self.test_raw = (test_mode == RAW or test_mode == BOTH)
#         self.test_adv = (test_mode == ADV or test_mode == BOTH)
        
        # Set Model Hyperparameters
        self.learning_rate = lr
        self.iterations = iterations
        self.print_freq = print_freq
        self.cuda = torch.cuda.is_available()
        
        
        # Load Model to Conduct Adversarial Training
        adversarial_model = self.load_model(self.cuda, load_adv_dir, load_adv_name, TEST)
        self.adversarial_generator = Attacks(adversarial_model, eps, self.N_train, self.N_test, 
                                             self.data_processor.get_const(), adv_momentum, 
                                             is_normalized, store_adv)
        
        # Load Target Model
        self.target_model = self.load_model(self.cuda, load_dir, load_name, TEST)
        self.target_model = self.target_model.eval()
        
        target_model = self.load_model(self.cuda, load_dir, load_name, TEST)
        
        self.model = FullDenoiser(target_model).cuda()
        self.model.denoiser = self.load_checkpoint(self.model.denoiser, denoiser_dir, denoiser_name)
                
    
    def load_model(self, is_cuda, load_dir=None, load_name=None, mode=None):
        """ Return WideResNet model, in gpu if applicable, and with provided checkpoint if given"""
        model = WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.0)
        
        # Send to GPU if any
        if is_cuda:
            model = torch.nn.DataParallel(model).cuda()
            print(">>> SENDING MODEL TO GPU...")
        
        # Load checkpoint 
        if load_dir and load_name and mode == TEST:
            model = self.load_checkpoint(model, load_dir, load_name)
            print(">>> LOADING CHECKPOINT:", load_dir)
            
        return model
        
    
    def grad_step(self, x_batch, y_batch):
        """ Performs a step during training. """
        # Compute output for example
        logits = self.target_model(x_batch)
        loss = self.target_model.module.loss(logits, y_batch)
        
        return logits, loss

        # Update Mean loss for current iteration
#         losses.update(loss.item(), x_batch.size(0))
#         prec1 = accuracy(logits.data, y_batch, k=k)
#         top1.update(prec1.item(), x_batch.size(0))
        
#         # compute gradient and do SGD step
#         loss.backward()
#         optimizer.step()
        
#         # Set grads to zero for new iter
#         optimizer.zero_grad()
        
    
    def no_grad_step(self, x_batch, y_batch):
        """ Performs a step during testing."""
        logits, loss = None, None
        with torch.no_grad():
            logits = self.target_model(x_batch)
            loss = self.target_model.module.loss(logits, y_batch)

        # Update Mean loss for current iteration
#         losses.update(loss.item(), x_batch.size(0))
#         prec1 = accuracy(logits.data, y_batch, k=k)
#         top1.update(prec1.item(), x_batch.size(0))

        return logits, loss
    
    
    def train(self, train_max_iter=1, test_max_iter=1):
        
        self.target_model.eval()

        train_loss_hist = []
        train_acc_hist = []
        test_loss_hist = []
        test_acc_hist = []
        
        best_pred = 0.0
        
        end = time.time()
        
        for itr in range(self.iterations):
            
#             self.model.train()
            
            optimizer = optim.Adam(self.model.denoiser.parameters(), lr=self.learning_rate)
            
            losses = AverageMeter()
            batch_time = AverageMeter()
            top1 = AverageMeter()
            
            x_adv = None
            stored = self.adversarial_generator.get_stored(mode='train')
            
            for i, (x, y) in enumerate(self.train_loader):

                x = x.cuda()
                y = y.cuda()

                # FGSM
                if not stored:
                    # 1. Generate Predictions on batch
                    logits, _ = self.no_grad_step(x, y)
                    y_pred = torch.argmax(logits, dim=1)

                    # 2. Generate adversaries with y_pred (avoids 'label leak' problem)
                    x_adv, _ = self.adversarial_generator.fast_pgd(x, y_pred, train_max_iter, mode='train')
#                     self.adversarial_generator.retain_adversaries(x_adv, y, mode='train')
                else:
                    x_adv, _ = self.adversarial_generator.fast_pgd(x, y, train_max_iter, mode='train')
                
                optimizer.zero_grad()
                
                logits_smooth, logits_org, loss = self.model(x, x_adv)
                
                # 6. Update Mean loss for current iteration
                losses.update(loss.item(), x.size(0))
                prec1 = accuracy(logits_smooth.data, y)
                top1.update(prec1.item(), x.size(0))

                # compute gradient and do SGD step
                loss.backward()
                optimizer.step()

                # Set grads to zero for new iter
                optimizer.zero_grad()

                batch_time.update(time.time() - end)
                end = time.time()
                
                if i % self.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                              itr, i, len(self.train_loader), batch_time=batch_time,
                              loss=losses, top1=top1))
            
            # Evaluate on validation set
#             test_loss, test_prec1 = self.test(self.test_loader, test_max_iter)
            
#             train_loss_hist.append(losses.avg)
#             train_acc_hist.append(top1.avg)
#             test_loss_hist.append(test_loss)
#             test_acc_hist.append(test_prec1)
            
            #Store best model
#             is_best = best_pred < test_prec1
            is_best = True
            self.save_checkpoint(is_best, (itr+1), self.model.denoiser.state_dict(), self.save_dir)
#             if is_best:
#                 best_pred = test_prec1
                
            # Adversarial examples generated on the first iteration. Store them if re-using same iteration ones.
#             if self.store_adv:
#                 self.adversarial_generator.set_stored('train', True)
                
        return (train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist)
              
    
    
    def test(self, batch_loader, test_max_iter=1):
#         self.model.eval()
        
        losses = AverageMeter()
        batch_time = AverageMeter()
        top1 = AverageMeter()
        
        end = time.time()
        
        for i, (x,y) in enumerate(batch_loader):
            
            x = x.cuda()
            y = y.cuda()
            
            # Test on adversarial
            logits, _ = self.no_grad_step(x, y)
            y_pred = torch.argmax(logits, dim=1)

            # 2. Generate adversaries with y_pred (avoids 'label leak' problem)
            x_adv, _ = self.adversarial_generator.fast_pgd(x, y_pred, test_max_iter, mode='test')
            
            logits_smooth, logits_org, loss = self.model(x, x_adv)
            
            
            # 6. Update Mean loss for current iteration
            losses.update(loss.item(), x.size(0))
            prec1 = accuracy(logits_smooth.data, y)
            top1.update(prec1.item(), x.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % self.print_freq == 0:
                print('Epoch: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(batch_loader), batch_time=batch_time,
                          loss=losses, top1=top1))

        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
        return (losses.avg, top1.avg)
                      
        

    def save_checkpoint(self, is_best, epoch, state, save_dir, base_name="chkpt"):
        """Saves checkpoint to disk"""
        directory = save_dir
        filename = base_name + ".pth.tar"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + filename
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, directory + base_name + '__model_best.pth.tar')
            
    
    def load_checkpoint(self, model, load_dir, load_name):
        """Load checkpoint from disk"""
        filepath = load_dir + load_name
        if os.path.exists(filepath):
            state_dict = torch.load(filepath)
            model.load_state_dict(state_dict)
            return model
        
        print("Failed to load model. Exiting...")
        sys.exit(1)
                      
