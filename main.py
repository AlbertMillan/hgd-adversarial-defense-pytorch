import numpy as np

from train import Classifier

import argparse
import os

TRAIN_AND_TEST = 0
TEST = 1

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training. See code for default values.')
    
    # STORAGE LOCATION VARIABLES
    parser.add_argument('--ds_name', default='CIFAR10', metavar='Dataset', type=str, help='Dataset name')
    parser.add_argument('--ds_path', default='datasets/', metavar='Path', type=str, help='Dataset path')
    parser.add_argument('--load_dir', '--ld', default='chkpt/chkpt_scaled/', type=str, help='Path to Model')
    parser.add_argument('--load_name', '--ln', default='chkpt_plain__model_best.pth.tar', type=str, help='File Name')
    parser.add_argument('--load_adv_dir', '--lad', default='chkpt/chkpt_scaled/', type=str, help='Path to Model')
    parser.add_argument('--load_adv_name', '--lan', default='chkpt_plain__model_best.pth.tar', type=str, help='File Name')
    parser.add_argument('--denoiser_dir', default='chkpt/', type=str, help='Path to Denoiser')
    parser.add_argument('--denoiser_name', default='denoiser_pgd4chkpt.pth.tar', type=str, help='Name of Denoiser chkpt.')
    
    parser.add_argument('--save_dir', '--sd', default='model_chkpt/new/', type=str, help='Path to Model')
    parser.add_argument('--save_idx', default=0, type=int, help='ID of results (default: 0)')
#     parser.add_argument('--save_name', '--mn', default='chkpt_plain.pth.tar', type=str, help='File Name')
    
    
    # MODEL HYPERPARAMETERS
    parser.add_argument('--lr', default=0.001, metavar='lr', type=float, help='Learning rate')
    parser.add_argument('--itr', default=30, metavar='iter', type=int, help='Number of iterations')
    parser.add_argument('--batch_size', default=64, metavar='batch_size', type=int, help='Batch size')
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
                            args.load_adv_dir, args.load_adv_name, args.load_dir, args.load_name, 
                            args.denoiser_dir, args.denoiser_name, args.save_dir)
    
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
        test_loss, test_acc = classifier.test(classifier.test_loader, args.test_max_iter)
#         np.save('results_2/test_acc__'+str(args.save_idx)+'.npy', test_acc)
    