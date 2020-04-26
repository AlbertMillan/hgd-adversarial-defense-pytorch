import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

import argparse
import time
import math
import shutil
import sys, os

from attacks import Attacks
from model import WideResNet
from denoiser import Denoiser
from processor import Preprocessor, AverageMeter, accuracy


TRAIN_AND_TEST = 0
TEST = 1

class Classifier:
    """
    
    """
    
    def __init__(self, ds_name, ds_path, lr, iterations, batch_size, print_freq, k, eps, is_normalized,
                 adv_momentum, store_adv=None, load_adv_dir=None, load_adv_name=None, load_dir=None,
                 load_name=None, save_dir=None):
        
        self.data_processor = Preprocessor(ds_name, ds_path, is_normalized)
        
        # Load Data
        self.train_data, self.test_data, self.N_train, self.N_test = self.data_processor.datasets()
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size)
        
        # Other Variables
        self.save_dir = save_dir
        self.store_adv = store_adv
        
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
        
        # Load Denoiser
        self.denoiser = Denoiser(x_h=32, x_w=32)
        self.denoiser = self.denoiser.cuda()

#         sys.exit()
                
                
    
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
            
            optimizer = optim.Adam(self.denoiser.parameters(), lr=self.learning_rate)
            
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
                    self.adversarial_generator.retain_adversaries(x_adv, y, mode='train')
                else:
                    x_adv, y_adv = self.adversarial_generator.fast_pgd(x, y, train_max_iter, mode='train')
                
                # 3. Compute denoised image. Need to check this...
                noise = self.denoiser.forward(x_adv)
                x_smooth = x_adv + noise
                
#                 print(noise)
                
                # 4. Get logits from smooth and denoised image
                logits_smooth, _ = self.grad_step(x_smooth, y)
                logits_org, _ = self.grad_step(x, y)
                
                # 5. Compute loss
                loss = torch.sum( torch.abs(logits_smooth - logits_org) ) / x.size(0)
                
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
            
            # Store best model
#             is_best = best_pred < test_prec1
#             self.save_checkpoint(is_best, (itr+1), self.model.state_dict(), self.save_dir)
#             if is_best:
#                 best_pred = test_prec1
                
            # Adversarial examples generated on the first iteration. Store them if re-using same iteration ones.
            if self.store_adv:
                self.adversarial_generator.set_stored('train', True)
                
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
            self.test_step(x, y, losses, top1)

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
                      


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training. See code for default values.')
    
    # STORAGE LOCATION VARIABLES
    parser.add_argument('--ds_name', default='CIFAR10', metavar='Dataset', type=str, help='Dataset name')
    parser.add_argument('--ds_path', default='datasets/', metavar='Path', type=str, help='Dataset path')
    parser.add_argument('--load_dir', '--ld', default='chkpt/chkpt_norm/', type=str, help='Path to Model')
    parser.add_argument('--load_name', '--ln', default='chkpt_plain__model_best.pth.tar', type=str, help='File Name')
    parser.add_argument('--load_adv_dir', '--lad', default='chkpt/chkpt_norm/', type=str, help='Path to Model')
    parser.add_argument('--load_adv_name', '--lan', default='chkpt_plain__model_best.pth.tar', type=str, help='File Name')
    parser.add_argument('--save_dir', '--sd', default='model_chkpt/new/', type=str, help='Path to Model')
    parser.add_argument('--save_idx', default=0, type=int, help='ID of results (default: 0)')
#     parser.add_argument('--save_name', '--mn', default='chkpt_plain.pth.tar', type=str, help='File Name')
    
    
    # MODEL HYPERPARAMETERS
    parser.add_argument('--lr', default=0.001, metavar='lr', type=float, help='Learning rate')
    parser.add_argument('--itr', default=30, metavar='iter', type=int, help='Number of iterations')
    parser.add_argument('--batch_size', default=60, metavar='batch_size', type=int, help='Batch size')
    parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float, help='weight decay (default: 2e-4)')
    parser.add_argument('--print_freq', '-p', default=10, type=int, help='print frequency (default: 10)')
    parser.add_argument('--topk', '-k', default=1, type=int, help='Compute accuracy over top k-predictions (default: 1)')
    
    
    # ADVERSARIAL GENERATOR PROPERTIES
    parser.add_argument('--eps', '-e', default=(8./255.), type=float, help='Epsilon (default: 8/255)')
    parser.add_argument('--adv_momentum', default=None, type=float, help='Momentum for adversarial training (default: None)')
    parser.add_argument('--attack', '--att', default=0, type=int, help='Attack Type (default: 0)')
    parser.add_argument('--train_max_iter', default=1, type=int, help='Iterations required to generate adversarial examples during training (default: 1)')
    parser.add_argument('--test_max_iter', default=1, type=int, help='Iterations required to generate adversarial examples during testing (default: 1)')
    parser.add_argument('--test_mode', default=0, type=int, help='Test on raw images (0), adversarial images (1) or both (2) (default: 0)')
    parser.add_argument('--store_adv', default=0, type=int, help='Wether to retain and reuse generated adversaries for training (default: 0)')


    

    
    # OTHER PROPERTIES
    parser.add_argument('--gpu', default="0,1", type=str, help='GPU devices to use (0-7) (default: 0,1)')
    parser.add_argument('--mode', default=0, type=int, help='Wether to perform test without trainig (default: 0)')
    parser.add_argument('--zero_norm', default=0, type=int, help='Whether to perform zero-mean normalization on dataset. (default: 0)')
    
    args = parser.parse_args()
    
    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    
    classifier = Classifier(args.ds_name, args.ds_path, args.lr, args.itr, args.batch_size, args.print_freq,
                            args.topk, args.eps, args.zero_norm, args.adv_momentum, args.store_adv, 
                            args.load_adv_dir, args.load_adv_name, args.load_dir, args.load_name, args.save_dir)
    
    print("==================== TRAINING ====================")
    
    if args.mode == TRAIN_AND_TEST:
        train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist = classifier.train(train_max_iter=args.train_max_iter,
                                                                                          test_max_iter=args.test_max_iter)

        model_type = ['plain','PGD','CW']

        np.save("results/train_loss__"+str(model_type[args.attack])+"__"+str(args.test_max_iter)+".npy", train_loss_hist)
        np.save("results/train_acc__"+str(model_type[args.attack])+"__"+str(args.test_max_iter)+".npy", train_acc_hist)
        np.save("results/test_loss__"+str(model_type[args.attack])+"__"+str(args.test_max_iter)+".npy", test_loss_hist)
        np.save("results/test_acc__"+str(model_type[args.attack])+"__"+str(args.test_max_iter)+".npy", test_acc_hist)
    
    print("==================== TESTING ====================")
    
    if args.mode == TEST:
        pass
#         test_loss, test_acc = classifier.test(classifier.test_loader, args.test_max_iter)
#         np.save('results_2/test_acc__'+str(args.save_idx)+'.npy', test_acc)
    
