# FedSiam: Towards Adaptive Federated Semi-Supervised Learning

A PyTorch implementation of  **labels-at-client scenario**  for the paper  **FedSiam: Towards Adaptive Federated Semi-Supervised Learning**

We have 5 baselines (FedAvg, FedAvg+, FedSem, FedFixMatch, FedMatch) and 3 proposed FedSemi frameworks (FedSemi-D, FedSemi-Pi, FedSemi-MT) in our experiment. 

We do our experiments on MNIST, Cifar-10, and SVHN datasets.

you should place your data in `./fedsiam-cikm2021-master/data/mnist` (mnist for example)

## Getting Started

python>=3.6  
pytorch>=0.4

To install PyTorch, see installation instructions on the [PyTorch website](https://pytorch.org/get-started/locally).





## Some Exampless

We provide some examples here.



#### MNIST IID

> python main_fedsiam_d.py --dataset mnist --epochs 50 --gpu 0 --label_rate 0.1 --iid iid --phi_g 3



#### MNIST Non-IID-I

> python main_fedsiam_d.py --dataset mnist --epochs 50 --gpu 0 --label_rate 0.1 --iid noniid_tradition --phi_g 3



#### MNIST Non-IID-III

> python main_fedsiam_d.py --dataset mnist --epochs 50 --gpu 0 --label_rate 0.1 --iid noniid_ssl --phi_g 3



#### CIFAR-10 IID

> python main_fedsiam_d.py --dataset cifar --epochs 50 --gpu 0 --label_rate 0.15 --iid iid --ramp rectangle --local_bs 30



#### SVHN Non-IID-II

> python main_fedsiam_d.py --dataset svhn --epochs 50 --gpu 0 --label_rate 0.15 --iid noniid_improve --phi_g 3






## References
See our  paper and supplementary material for more details as well as all references.
