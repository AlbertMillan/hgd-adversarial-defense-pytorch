# python main.py --mode 1 --att 1 --test_max_iter 1 --save_idx 2 --gpu "0,1"

# PLAIN
# python main.py --mode 1 --att 0 --test_max_iter 0 --denoiser_name "denoiser_pgd1chkpt.pth.tar" --save_idx 11 --gpu "0,1" &

# python main.py --mode 1 --att 0 --test_max_iter 0 --denoiser_name "denoiser_pgd4chkpt.pth.tar" --save_idx 12 --gpu "2,3" &

# python main.py --mode 1 --att 0 --test_max_iter 0 --denoiser_name "denoiser_pgd7chkpt.pth.tar" --save_idx 13 --gpu "4,5" 



# FGSM
# python main.py --mode 1 --att 1 --test_max_iter 1 --denoiser_name "denoiser_pgd1chkpt.pth.tar" --save_idx 14 --gpu "0,1" &

# python main.py --mode 1 --att 1 --test_max_iter 1 --denoiser_name "denoiser_pgd4chkpt.pth.tar" --save_idx 15 --gpu "2,3" &

# python main.py --mode 1 --att 1 --test_max_iter 1 --denoiser_name "denoiser_pgd7chkpt.pth.tar" --save_idx 16 --gpu "4,5" 


# # PGD20
# python main.py --mode 1 --att 1 --test_max_iter 20 --denoiser_name "denoiser_pgd1chkpt.pth.tar" --save_idx 17 --gpu "0,1" &

# python main.py --mode 1 --att 1 --test_max_iter 20 --denoiser_name "denoiser_pgd4chkpt.pth.tar" --save_idx 18 --gpu "2,3" &

# python main.py --mode 1 --att 1 --test_max_iter 20 --denoiser_name "denoiser_pgd7chkpt.pth.tar" --save_idx 19 --gpu "4,5" 


# # PGD100
# python main.py --mode 1 --att 1 --test_max_iter 100 --denoiser_name "denoiser_pgd1chkpt.pth.tar" --save_idx 20 --gpu "0,1" &

# python main.py --mode 1 --att 1 --test_max_iter 100 --denoiser_name "denoiser_pgd4chkpt.pth.tar" --save_idx 21 --gpu "2,3" &

# python main.py --mode 1 --att 1 --test_max_iter 100 --denoiser_name "denoiser_pgd7chkpt.pth.tar" --save_idx 22 --gpu "4,5" &


# # MIM20
python main.py --mode 1 --att 0 --test_max_iter 0 --denoiser_dir "chkpt/denoiser_mim7/" --denoiser_name "chkpt__model_best.pth.tar" --save_idx 23 --gpu "0,1" &

python main.py --mode 1 --att 1 --test_max_iter 1 --denoiser_dir "chkpt/denoiser_mim7/" --denoiser_name "chkpt__model_best.pth.tar" --save_idx 24 --gpu "2,3" &

python main.py --mode 1 --att 1 --test_max_iter 20 --denoiser_dir "chkpt/denoiser_mim7/" --denoiser_name "chkpt__model_best.pth.tar" --save_idx 25 --gpu "4,5" &

# python main.py --mode 1 --att 1 --test_max_iter 100 --denoiser_dir "chkpt/denoiser_mim7/" --denoiser_name "chkpt__model_best.pth.tar" --save_idx 26 --gpu "4,5" &

python main.py --mode 1 --att 1 --test_max_iter 20 --adv_momentum 1.0 --denoiser_dir "chkpt/denoiser_mim7/" --denoiser_name "chkpt__model_best.pth.tar" --save_idx 27 --gpu "6,7"