# FedSiam
A PyTorch implementation of  **labels-at-server scenario**  for the paper  **FedSiam: Towards Adaptive Federated Semi-Supervised Learning**



you should place your data in `./fedsiam-cikm2021-master/data/mnist` (mnist for example)

## IID Setting

### run MNIST


python fedsiam-pi-main.py --model byol --optimizer lars --data_dir ../data/mnist --output_dir ./outputs/ --backbone Mnist --dataset mnist --batch_size 10 --num_epochs 200 --optimizer lars_simclr --weight_decay 1.5e-6 --base_lr 0.3 --warmup_epochs 10 --label_rate 0.01 --num_workers 4


python fedsiam-mt-main.py --model byol --optimizer lars --data_dir ../data/mnist --output_dir ./outputs/ --backbone Mnist --dataset mnist --batch_size 10 --num_epochs 200 --optimizer lars_simclr --weight_decay 1.5e-6 --base_lr 0.3 --warmup_epochs 10 --label_rate 0.01 --num_workers 4


python fedfixmatch-main.py --model byol --optimizer lars --data_dir ../data/mnist --output_dir ./outputs/ --backbone Mnist --dataset mnist --batch_size 10 --num_epochs 200 --optimizer lars_simclr --weight_decay 1.5e-6 --base_lr 0.3 --warmup_epochs 10 --label_rate 0.01 --num_workers 4

python fedmatch-main.py --model byol --optimizer lars --data_dir ../data/mnist --output_dir ./outputs/ --backbone Mnist --dataset mnist --batch_size 10 --num_epochs 200 --optimizer lars_simclr --weight_decay 1.5e-6 --base_lr 0.3 --warmup_epochs 10 --label_rate 0.01 --num_workers 4


### run SVHN

python fedfixmatch-main.py --model byol --optimizer lars --data_dir ../data/svhn --output_dir ./outputs/ --backbone Svhn --dataset svhn --batch_size 10 --num_epochs 150 --optimizer lars_simclr --weight_decay 1.5e-6 --base_lr 0.3 --warmup_epochs 10 --label_rate 0.01 --num_workers 4



python fedmatch-main.py --model byol --optimizer lars --data_dir ../data/svhn --output_dir ./outputs/ --backbone Svhn --dataset svhn --batch_size 10 --num_epochs 150 --optimizer lars_simclr --weight_decay 1.5e-6 --base_lr 0.3 --warmup_epochs 10 --label_rate 0.01 --num_workers 4



python fedsiam-pi-main.py --model byol --optimizer lars --data_dir ../data/svhn --output_dir ./outputs/ --backbone Svhn --dataset svhn --batch_size 10 --num_epochs 150 --optimizer lars_simclr --weight_decay 1.5e-6 --base_lr 0.3 --warmup_epochs 10 --label_rate 0.01 --num_workers 4



python fedsiam-mt-main.py --model byol --optimizer lars --data_dir ../data/svhn --output_dir ./outputs/ --backbone Svhn --dataset svhn --batch_size 10 --num_epochs 150 --optimizer lars_simclr --weight_decay 1.5e-6 --base_lr 0.3 --warmup_epochs 10 --label_rate 0.01 --num_workers 4



### run CIFAR-10


python fedfixmatch-main.py --model byol --optimizer lars --data_dir ../data/cifar --output_dir ./outputs/ --backbone Cifar --dataset cifar10 --batch_size 10 --num_epochs 200 --optimizer lars_simclr --weight_decay 1.5e-6 --base_lr 0.3 --warmup_epochs 10 --label_rate 0.01 --num_workers 4


python fedsiam-pi-main.py --model byol --optimizer lars --data_dir ../data/cifar --output_dir ./outputs/ --backbone Cifar --dataset cifar10 --batch_size 10 --num_epochs 200 --optimizer lars_simclr --weight_decay 1.5e-6 --base_lr 0.3 --warmup_epochs 10 --label_rate 0.01 --num_workers 4


python fedsiam-mt-main.py --model byol --optimizer lars --data_dir ../data/cifar --output_dir ./outputs/ --backbone Cifar --dataset cifar10 --batch_size 10 --num_epochs 200 --optimizer lars_simclr --weight_decay 1.5e-6 --base_lr 0.3 --warmup_epochs 10 --label_rate 0.01 --num_workers 4

python fedmatch-main.py --model byol --optimizer lars --data_dir ../data/cifar --output_dir ./outputs/ --backbone Cifar --dataset cifar10 --batch_size 10 --num_epochs 200 --optimizer lars_simclr --weight_decay 1.5e-6 --base_lr 0.3 --warmup_epochs 10 --label_rate 0.01 --num_workers 4









## Non-IID Setting

### run MNIST


python fedsiam-pi-main.py --model byol --optimizer lars --data_dir ../data/mnist --output_dir ./outputs/ --backbone Mnist --dataset mnist --batch_size 10 --num_epochs 200 --optimizer lars_simclr --weight_decay 1.5e-6 --base_lr 0.3 --warmup_epochs 10 --label_rate 0.01 --num_workers 4 --iid noniid




python fedsiam-mt-main.py --model byol --optimizer lars --data_dir ../data/mnist --output_dir ./outputs/ --backbone Mnist --dataset mnist --batch_size 10 --num_epochs 200 --optimizer lars_simclr --weight_decay 1.5e-6 --base_lr 0.3 --warmup_epochs 10 --label_rate 0.01 --num_workers 4 --iid noniid


python fedfixmatch-main.py --model byol --optimizer lars --data_dir ../data/mnist --output_dir ./outputs/ --backbone Mnist --dataset mnist --batch_size 10 --num_epochs 200 --optimizer lars_simclr --weight_decay 1.5e-6 --base_lr 0.3 --warmup_epochs 10 --label_rate 0.01 --num_workers 4 --iid noniid

python fedmatch-main.py --model byol --optimizer lars --data_dir ../data/mnist --output_dir ./outputs/ --backbone Mnist --dataset mnist --batch_size 10 --num_epochs 200 --optimizer lars_simclr --weight_decay 1.5e-6 --base_lr 0.3 --warmup_epochs 10 --label_rate 0.01 --num_workers 4 --iid noniid


### run SVHN

python fedfixmatch-main.py --model byol --optimizer lars --data_dir ../data/svhn --output_dir ./outputs/ --backbone Svhn --dataset svhn --batch_size 10 --num_epochs 150 --optimizer lars_simclr --weight_decay 1.5e-6 --base_lr 0.3 --warmup_epochs 10 --label_rate 0.01 --num_workers 4 --iid noniid



python fedmatch-main.py --model byol --optimizer lars --data_dir ../data/svhn --output_dir ./outputs/ --backbone Svhn --dataset svhn --batch_size 10 --num_epochs 150 --optimizer lars_simclr --weight_decay 1.5e-6 --base_lr 0.3 --warmup_epochs 10 --label_rate 0.01 --num_workers 4 --iid noniid



python fedsiam-pi-main.py --model byol --optimizer lars --data_dir ../data/svhn --output_dir ./outputs/ --backbone Svhn --dataset svhn --batch_size 10 --num_epochs 150 --optimizer lars_simclr --weight_decay 1.5e-6 --base_lr 0.3 --warmup_epochs 10 --label_rate 0.01 --num_workers 4 --iid noniid



python fedsiam-mt-main.py --model byol --optimizer lars --data_dir ../data/svhn --output_dir ./outputs/ --backbone Svhn --dataset svhn --batch_size 10 --num_epochs 150 --optimizer lars_simclr --weight_decay 1.5e-6 --base_lr 0.3 --warmup_epochs 10 --label_rate 0.01 --num_workers 4 --iid noniid



### run CIFAR-10


python fedfixmatch-main.py --model byol --optimizer lars --data_dir ../data/cifar --output_dir ./outputs/ --backbone Cifar --dataset cifar10 --batch_size 10 --num_epochs 200 --optimizer lars_simclr --weight_decay 1.5e-6 --base_lr 0.3 --warmup_epochs 10 --label_rate 0.01 --num_workers 4 --iid noniid


python fedsiam-pi-main.py --model byol --optimizer lars --data_dir ../data/cifar --output_dir ./outputs/ --backbone Cifar --dataset cifar10 --batch_size 10 --num_epochs 200 --optimizer lars_simclr --weight_decay 1.5e-6 --base_lr 0.3 --warmup_epochs 10 --label_rate 0.01 --num_workers 4 --iid noniid


python fedsiam-mt-main.py --model byol --optimizer lars --data_dir ../data/cifar --output_dir ./outputs/ --backbone Cifar --dataset cifar10 --batch_size 10 --num_epochs 200 --optimizer lars_simclr --weight_decay 1.5e-6 --base_lr 0.3 --warmup_epochs 10 --label_rate 0.01 --num_workers 4 --iid noniid



python fedmatch-main.py --model byol --optimizer lars --data_dir ../data/cifar --output_dir ./outputs/ --backbone Cifar --dataset cifar10 --batch_size 10 --num_epochs 200 --optimizer lars_simclr --weight_decay 1.5e-6 --base_lr 0.3 --warmup_epochs 10 --label_rate 0.01 --num_workers 4 --iid noniid










