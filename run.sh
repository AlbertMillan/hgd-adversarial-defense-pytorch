python main.py --mode 0 --att 1 --train_max_iter 1 --zero_norm 0 --store_adv 0 --load_adv_dir "chkpt/chkpt_scaled/" --save_dir "chkpt/denoiser_pgd1/" --gpu "0,1" &

python main.py --mode 0 --att 1 --train_max_iter 4 --zero_norm 0 --store_adv 0 --load_adv_dir "chkpt/chkpt_scaled/" --save_dir "chkpt/denoiser_pgd4/" --gpu "2,3,4" &

python main.py --mode 0 --att 1 --train_max_iter 7 --zero_norm 0 --store_adv 0 --load_adv_dir "chkpt/chkpt_scaled/" --save_dir "chkpt/denoiser_pgd7/" --gpu "5,6,7" 

python main.py --mode 0 --att 1 --train_max_iter 7 --adv_momentum 1.0 --zero_norm 0 --store_adv 0 --load_adv_dir "chkpt/chkpt_scaled/" --save_dir "chkpt/denoiser_mim7/" --gpu "0,1,2,3"