import datetime
import json
import torch
import numpy as np
import os
import random
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import RandomSampler
from tqdm import tqdm


# Import data utilities
import data_utils.active_learning.active_learning as active_learning
from data_utils.fast_mnist import create_MNIST_dataset
import data_utils.ood_detection.cifar10 as cifar10
import data_utils.ood_detection.cifar100 as cifar100
import data_utils.ood_detection.lsun as lsun
import data_utils.ood_detection.svhn as svhn
import data_utils.ood_detection.mnist as mnist
import data_utils.ood_detection.gauss as gauss
import data_utils.ood_detection.tiny_imagenet as tiny_imagenet

# Import network architectures
from net.resnet import resnet18
from net.vgg import vgg16

# Import train and test utils
from utils.train_utils import train_single_epoch, model_save_name

# Importing uncertainty metrics
from metrics.uncertainty_confidence import entropy, logsumexp, confidence, sumexp, maxval
from metrics.classification_metrics import test_classification_net
from metrics.classification_metrics import test_classification_net_ensemble

# Importing args
from utils.args import al_args

# Importing GMM utilities
from utils.gmm_utils import get_embeddings, gmm_evaluate, gmm_fit, gmm_evaluate_with_perturbation
from utils.ensemble_utils import ensemble_forward_pass

# Mapping model name to model function
models = {"resnet18": resnet18, "vgg16": vgg16}


def class_ratio(data_loader):
    #统计每一类别的比率
    num_classes = 10
    class_n = len(data_loader.dataset)
    class_count = torch.zeros(num_classes)
    for data, label in data_loader:
        class_count += torch.Tensor([torch.sum(label == c) for c in range(num_classes)])

    class_prob = class_count / class_n
    return class_prob


def compute_density(logits, class_ratio):
    return torch.sum((torch.exp(logits) * class_ratio), dim=1)


if __name__ == "__main__":
    args = al_args().parse_args()
    print(args)
    if args.al_type=="gmm" and args.perturbation==1:
        epsilons = [-0.01,0.01]
    else:
        epsilons = [0.0]
    print("epsilons:",epsilons)
        
    for epsilon in epsilons:
        # Setting additional parameters
        torch.manual_seed(args.seed)
        device = torch.device(f"cuda:{args.gpu}")

        model_fn = models[args.model_name]

        # Creating the datasets
        num_classes = 10
        if args.dataset == "mnist":
            mean = [0.1307,0.1307,0.1307]
            std = [0.3081,0.3081,0.3081]
            train_dataset, test_dataset = create_MNIST_dataset(device)  #60000, 10000
        elif args.dataset == "cifar10":
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
            normalize = transforms.Normalize(
                mean=mean,
                std=std,
            )
            # define transform
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
            train_dataset = datasets.CIFAR10(
                root="./data",
                train=True,
                download=False,
                transform=train_transform,
            ) #50000
            test_dataset = datasets.CIFAR10(
                root="./data",
                train=False,
                download=False,
                transform=train_transform,
            )#10000
            
        if args.dataset=="mnist":
            args.num_initial_samples = 20 #初始化时给20个训练样本
            args.max_training_samples = 300  #20--(+5)-->300
            args.acquisition_batch_size = 10  #每次增加10个样本
            args.epochs = 10 #每次训练10个epoch
            args.train_batch_size = 64
        else:
            args.num_initial_samples = 2000 #初始化时给1000个训练样本
            args.max_training_samples = 20000  #2000--(+1000)-->20000
            args.acquisition_batch_size = 1000  #每次增加1000个样本
            args.epochs = 100 #每次训练100个epoch
            args.train_batch_size = 256
            
        if args.al_type == "gmm":
            args.sn=True
            args.mod=True
        else: #ensemble,entropy,random都不用修改网络
            args.sn=False
            args.mod=False
        
        # Creating a validation split
        idxs = list(range(len(train_dataset)))
        split = int(np.floor(len(train_dataset)*1/5))
        np.random.seed(args.seed)
        np.random.shuffle(idxs)

        train_idx, val_idx = idxs[split:], idxs[:split]
        val_dataset = torch.utils.data.Subset(train_dataset, val_idx)  #10000
        print("val_dataset",len(val_dataset))
        train_dataset = torch.utils.data.Subset(train_dataset, train_idx)  #50000
        print("train_dataset",len(train_dataset))

        initial_sample_indices = active_learning.get_balanced_sample_indices(
            train_dataset,
            num_classes=num_classes,
            n_per_digit=args.num_initial_samples / num_classes,
        )

        kwargs = {"num_workers": 0, "pin_memory": False}
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)#512
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs) 

        # Run experiment
        num_run = 3
        test_accs = {}

        for i in range(num_run):
            test_accs[i] = []
            

        for run in range(num_run):
            print("Experiment run: " + str(run) + " =====================================================================>")
            torch.manual_seed(args.seed + run)

            # Setup data for the experiment 
            # Split off the initial samples first
            active_learning_data = active_learning.ActiveLearningData(train_dataset)
            # Acquiring the first training dataset from the total pool. This is random acquisition
            active_learning_data.acquire(initial_sample_indices)

            # Train loader for the current acquired training set
            sampler = active_learning.RandomFixedLengthSampler(dataset=active_learning_data.training_dataset, target_length=1920)#len(active_learning_data.training_dataset)=20,len(sampler)=1920
            train_loader = torch.utils.data.DataLoader(
                active_learning_data.training_dataset,
                sampler=sampler,
                batch_size=args.train_batch_size,
                **kwargs,
            )

            small_train_loader = torch.utils.data.DataLoader(
                active_learning_data.training_dataset,
                shuffle=True,
                batch_size=args.train_batch_size,
                **kwargs,
            )

            # Pool loader for the current acquired training set
            pool_loader = torch.utils.data.DataLoader(
                active_learning_data.pool_dataset,
                batch_size=args.scoring_batch_size, #128
                shuffle=False,
                **kwargs,
            )

            # Run active learning iterations
            active_learning_iteration = 0
            while True:
                print("Active Learning Iteration: " + str(active_learning_iteration) + " ================================>")

                lr = 0.01 #学习率参数 mnist:0.1，cifar10:0.01???
                weight_decay = 5e-4
                if args.al_type == "ensemble":
                    model_ensemble = [
                        model_fn(spectral_normalization=args.sn, mod=args.mod).to(device=device)
                        for _ in range(args.num_ensemble)
                    ]
                    optimizers = []
                    for model in model_ensemble:
                        optimizers.append(torch.optim.Adam(model.parameters(), weight_decay=weight_decay))
                        model.train()
                else:
                    # import pdb;pdb.set_trace()
                    # model = model_fn(spectral_normalization=args.sn, mod=args.mod, mnist=(args.dataset == "mnist")).to(device=device)
                    model = model_fn(spectral_normalization=args.sn, mod=args.mod).to(device=device)
                    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)
                    model.train()

                # Train
                print("Length of train dataset: " + str(len(train_loader.dataset)))
                best_model = None
                best_val_accuracy = 0
                
                if args.dataset=="cifar10":
                    if len(train_loader.dataset)<10000:
                        args.epochs=100
                    else:
                        args.epochs=200
                for epoch in tqdm(range(args.epochs)):
                    if args.al_type == "ensemble":
                        for (model, optimizer) in zip(model_ensemble, optimizers):
                            loss,acc = train_single_epoch(epoch, model, train_loader, optimizer, device)
                    else:
                        loss,acc = train_single_epoch(epoch, model, train_loader, optimizer, device)
                        # pass

                    _, val_accuracy, _, _, _ = (test_classification_net_ensemble(model_ensemble, val_loader, device=device)
                                                if args.al_type == "ensemble" else test_classification_net(model, val_loader, device=device))
                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        best_model = model_ensemble if args.al_type == "ensemble" else model
                    
                
                if args.al_type == "ensemble":
                    model_ensemble = best_model
                else:
                    model = best_model

                if args.al_type == "gmm":
                    # Fit the GMM on the trained model
                    model.eval()
                    print("small train loader",len(small_train_loader))
                    embeddings, labels,norm = get_embeddings(
                        model,
                        small_train_loader,
                        num_dim=512,
                        dtype=torch.double,
                        device=device,
                        storage_device=device,
                    )
                    gaussians_model, jitter_eps = gmm_fit(embeddings=embeddings, labels=labels, num_classes=num_classes)
                print("Training ended")

                # Testing the models
                if args.al_type == "ensemble":
                    print("Testing the model: Ensemble======================================>")
                    for model in model_ensemble:
                        model.eval()
                    (
                        conf_matrix,
                        accuracy,
                        labels_list,
                        predictions,
                        confidences,
                    ) = test_classification_net_ensemble(model_ensemble, test_loader, device=device)

                else:
                    print("Testing the model: Softmax/GMM======================================>")
                    (
                        conf_matrix,
                        accuracy,
                        labels_list,
                        predictions,
                        confidences,
                    ) = test_classification_net(model, test_loader, device=device)
                percentage_correct = 100.0 * accuracy
                test_accs[run].append(percentage_correct)

                print(f"run:{run},active learning iteration:{active_learning_iteration},active learning data len:{len(active_learning_data.training_dataset)},test set accuracy: ({percentage_correct:.2f}%)")

                if len(active_learning_data.training_dataset) >= args.max_training_samples:
                    break

                # Acquisition phase
                N = len(active_learning_data.pool_dataset)

                print("Performing acquisition ========================================")
                if args.al_type == "ensemble":
                    for model in model_ensemble:
                        model.eval()
                    ensemble_uncs = []
                    with torch.no_grad():
                        for data, _ in pool_loader:
                            data = data.to(device)
                            mean_output, predictive_entropy, mi = ensemble_forward_pass(model_ensemble, data)

                            ensemble_uncs.append(mi if args.mi else predictive_entropy)
                        ensemble_uncs = torch.cat(ensemble_uncs, dim=0)

                        (
                            candidate_scores,
                            candidate_indices,
                        ) = active_learning.get_top_k_scorers(ensemble_uncs, args.acquisition_batch_size)
                elif args.al_type == "gmm":
                    model.eval()
                    class_prob = class_ratio(train_loader)  #统计每一类别的比率
                    if args.perturbation == 0:
                        logits, labels,_ = gmm_evaluate(model, gaussians_model, pool_loader, device, num_classes, storage_device=device)
                    elif args.perturbation == 1: #添加输入扰动
                        logits, labels,_,_,_ = gmm_evaluate_with_perturbation(model,
                                                                        gaussians_model,
                                                                        pool_loader,
                                                                        device=device,
                                                                        num_classes=num_classes,
                                                                        storage_device=device,
                                                                        epsilon=epsilon,
                                                                        mean=mean,
                                                                        std=std)
                    (
                        candidate_scores,
                        candidate_indices,
                    ) = active_learning.get_top_k_scorers(
                        # compute_density(logits, class_prob.to(device)), #logsumexp换max
                        maxval(logits),
                        args.acquisition_batch_size,
                        uncertainty=False,
                    )
                elif args.al_type == "entropy":
                    model.eval()
                    logits = []
                    with torch.no_grad():
                        # import pdb;pdb.set_trace()
                        for data, _ in pool_loader:
                            data = data.to(device)
                            logits.append(model(data))
                        logits = torch.cat(logits, dim=0)

                    (
                        candidate_scores,
                        candidate_indices,
                    ) = active_learning.find_acquisition_batch(logits, args.acquisition_batch_size, entropy)
                elif args.al_type=="random":#随机获取
                    nums = len(active_learning_data.pool_dataset)
                    candidate_indices = random.sample(list(range(0,nums)),args.acquisition_batch_size)

                print("*********pool dataset size************:",len(active_learning_data.pool_dataset))
                # Performing acquisition
                active_learning_data.acquire(candidate_indices)
                active_learning_iteration += 1

        # Save the dictionaries
        save_name = model_save_name(args.model_name, args.sn, args.mod, args.coeff, args.seed)
        save_ensemble_mi = "_mi" if (args.al_type == "ensemble" and args.mi) else ""
        input_perturbation = "_input_perturbation" if (args.perturbation) else ""
        accuracy_file_name = "test_accs_" + save_name + '_' + args.al_type + save_ensemble_mi + input_perturbation + f"_{args.dataset}_{epsilon}.json"
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time, "%Y_%m_%d_%H_%M_%S")
        log_dir = f"results/active_learning/{args.dataset}/{time_str}"
        os.makedirs(log_dir,exist_ok=True)
        with open(os.path.join(log_dir, accuracy_file_name), "w") as acc_file:
            json.dump(test_accs, acc_file)
            print(f"save results to {os.path.join(log_dir, accuracy_file_name)}")
