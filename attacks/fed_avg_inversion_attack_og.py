import sys
sys.path.append("..")
from utils import categorical_gumbel_softmax_sampling, categorical_softmax, get_acc_and_bac, continuous_sigmoid_bound, Timer
import torch
from .initializations import _uniform_initialization, _gaussian_initialization, _mean_initialization, \
    _dataset_sample_initialization, _likelihood_prior_sample_initialization, _mixed_initialization, \
    _best_sample_initialization
from .priors import _joint_gmm_prior, _mean_field_gmm_prior, _categorical_prior, _categorical_l2_prior, \
    _categorical_mean_field_jensen_shannon_prior, _continuous_uniform_prior, _theoretical_optimal_prior, \
    _theoretical_typicality_prior, _theoretical_marginal_prior, _theoretical_marginal_typicality_prior
from .inversion_losses import _weighted_CS_SE_loss, _gradient_norm_weighted_CS_SE_loss, _squared_error_loss, _cosine_similarity_loss
from .ensembling import pooled_ensemble
from collections import OrderedDict
from models import MetaMonkey
import numpy as np
import copy
import pickle
import os
import multiprocessing
from defenses import dp_defense
from fair_loss import FairLoss
import torch.nn as nn


def caller(x):
    return os.system(x)


def epoch_matching_prior_mean_square_error(epoch_data, device=None):
    """
    Permutation invariant prior that can be applied over the individual datapoints in the epochs. We first average up
    each dataset in the epoch and then calculate pairwise L2 distances between the epoch-data. It is normalized for
    number of features and number of epochs.

    :param epoch_data: (list of torch.tensor) List of the data-tensors used for each epoch.
    :param device: (str) Name of the device on which the tensors are stored. If None is given, the device on which the
        first of the epoch data is taken.
    :return: prior (torch.tensor) The calculated value of the prior with gradient information.

    """
    n_epochs = len(epoch_data)
    n_features = epoch_data[0].size()[-1]
    if device is None:
        device = epoch_data[0].device
    average_local_data = torch.stack([1/data.size()[0] * data.sum(dim=0) for data in epoch_data]).to(device)
    prior = torch.tensor([0.], device=device)
    for i in range(n_epochs):
        prior += 1/(n_epochs**2) * 1/n_features * (average_local_data - average_local_data[i]).pow(2).sum()
    return prior


def simulate_local_training_for_attack(client_net, lr, criterion, dataset, labels, original_params,
                                       reconstructed_data_per_epoch, local_batch_size, priors=None,
                                       epoch_matching_prior=None, softmax_trick=True, gumbel_softmax_trick=False,
                                       sigmoid_trick=False, temperature=None, apply_projection_to_features=None,
                                       device=None):
    """
    Simulates the local training such that it can be differentiated through with the Pytorch engine.

    :param client_net: (MetaMonkey) A MetaMonkey wrapped nn.Module neural network that supports parameter assignment$
        directly through assigning and OrderedDict.
    :param lr: (float) The learning rate of the local training.
    :param criterion: (nn.Module) The loss function of the training.
    :param dataset: (datasets.BaseDataset) The dataset with which we work. It contains usually the data necessary for
        the calculation of the prior.
    :param labels: (torch.tensor) The labels for a whole local epoch, ordered as the batches should be.
    :param original_params: (OrderedDict) The original parameter dictionary of the network before training.
    :param reconstructed_data_per_epoch: (list of torch.tensor) List of the concatenated batches of data used for
        training. This is what we optimize for.
    :param local_batch_size: (int) The batch size of the local training.
    :param priors: (list of tuple(float, str)) The regularization parameter(s) plus the name(s) of the prior(s) we wish
        to use. Default None accounts to no prior.
    :param epoch_matching_prior: tuple(float, str) The regularization parameter of the epoch matching prior plus its
        name. If None is given (default), then no epoch matching prior will be applied.
    :param softmax_trick: (bool) Toggle to apply the softmax trick to the categorical features. Effectively, it serves
        as a structural prior on the features.
    :param gumbel_softmax_trick: (bool) Toggle to apply the gumbel-softmax trick to the categorical features.
    :param sigmoid_trick: (bool) Apply the sigmoid trick to the continuous features to enforce the bounds.
    :param apply_projection_to_features: (list) If given, both the softmax trick and the gumbel softmax trick will be
        applied only to the set of features given in this list.
    :param temperature: (float) Temperature parameter for the softmax in the categorical prior.
    :param device: (str) Name of the device on which the tensors are stored.
    :return: resulting_two_point_gradient: (list of torch.tensor) Two-point gradient estimate over a local training.
    """
    if device is None:
        device = dataset.device

    if apply_projection_to_features is None:
        apply_projection_to_features = 'all'

    available_priors = {
        'categorical_prior': _categorical_prior,
        'cont_uniform': _continuous_uniform_prior,
        'cont_joint_gmm': _joint_gmm_prior,
        'cont_mean_field_gmm': _mean_field_gmm_prior,
        'cat_mean_field_JS': _categorical_mean_field_jensen_shannon_prior,
        'cat_l2': _categorical_l2_prior,
        'theoretical_optimal': _theoretical_optimal_prior,
        'theoretical_typicality': _theoretical_typicality_prior,
        'theoretical_marginal': _theoretical_marginal_prior,
        'theoretical_marginal_typicality': _theoretical_marginal_typicality_prior
    }

    available_epoch_matching_priors = {
        'mean_squared_error': epoch_matching_prior_mean_square_error
    }

    if priors is not None:
        # will raise a key error of we chose a non-implemented prior
        prior_params = [prior_params[0] for prior_params in priors]
        prior_loss_functions = [available_priors[prior_params[1]] for prior_params in priors]
    else:
        prior_loss_functions = None
        prior_params = None

    regularizer = torch.as_tensor([0.01], device=device)

    n_data_lines = labels.size()[0]
    for local_epoch, reconstructed_data in enumerate(reconstructed_data_per_epoch):

        n_batches = int(np.ceil(n_data_lines / local_batch_size))
        
        for b in range(n_batches):
            current_batch_X = reconstructed_data[b*local_batch_size:min(n_data_lines, (b+1)*local_batch_size)]
            current_batch_y = labels[b*local_batch_size:min(n_data_lines, (b+1)*local_batch_size)].clone().detach()

            # apply softmax or gumbel-softmax
            if gumbel_softmax_trick:
                x_rec = categorical_gumbel_softmax_sampling(current_batch_X, tau=temperature, dataset=dataset)
                categoricals_projected = True
            elif softmax_trick:
                x_rec = categorical_softmax(current_batch_X, tau=temperature, dataset=dataset,
                                            apply_to=apply_projection_to_features)
                categoricals_projected = True
            else:
                x_rec = current_batch_X * 1.
                categoricals_projected = False

            if sigmoid_trick:
                x_rec = continuous_sigmoid_bound(x_rec, dataset=dataset, T=temperature)

            outputs = client_net(x_rec, client_net.parameters)
            training_loss = criterion(outputs, current_batch_y)

            binary_tensor = outputs.argmax(dim=1)
            
            
            grad = torch.autograd.grad(training_loss, client_net.parameters.values(), retain_graph=True,
                                       create_graph=True, only_inputs=True, allow_unused=True)

            client_net.parameters = OrderedDict((name, param - lr * param_grad) for ((name, param), param_grad) in zip(client_net.parameters.items(), grad))



            # keep track of a regularizer if needed
            if priors is not None:
                for prior_param, prior_function in zip(prior_params, prior_loss_functions):
                    regularizer += 1/(n_batches*local_epoch) * prior_param * prior_function(x_reconstruct=x_rec,
                                                                                            dataset=dataset,
                                                                                            softmax_trick=categoricals_projected,
                                                                                            labels=current_batch_y,
                                                                                            T=temperature)
        # print(local_epoch)
        # if local_epoch in {4}:
        #     print("Hello")
        #     print(binary_tensor, "rec < , >ori", current_batch_y)


    # if we have an epoch matching prior, we calculate its value, for this, we have to reapply any projections made on
    # the data previously
    if epoch_matching_prior is not None:
        epoch_matching_prior_param = epoch_matching_prior[0]
        epoch_matching_prior_function = available_epoch_matching_priors[epoch_matching_prior[1]]

        # reapply the projections if any
        if softmax_trick or gumbel_softmax_trick:
            projected_epoch_data = [categorical_softmax(epoch_data, dataset=dataset, tau=temperature,
                                                        apply_to=apply_projection_to_features) for epoch_data in reconstructed_data_per_epoch]
        else:
            projected_epoch_data = reconstructed_data_per_epoch
        # reapply the sigmoid if given
        if sigmoid_trick:
            projected_bounded_epoch_data = [continuous_sigmoid_bound(pd, dataset=dataset, T=temperature) for pd in projected_epoch_data]
        else:
            projected_bounded_epoch_data = projected_epoch_data
        regularizer += epoch_matching_prior_param * epoch_matching_prior_function(projected_bounded_epoch_data, device=device)

    # end of training, time to extract the parameters
    resulting_parameters = list(client_net.parameters.values())
    resulting_two_point_gradient = [original_param - param for original_param, param in
                                    zip(original_params, resulting_parameters)]

    return resulting_two_point_gradient, regularizer


def fed_avg_attack(original_net, attacked_clients_params, n_local_epochs, local_batch_size, lr,
                   dataset, per_client_ground_truth_data, per_client_ground_truth_labels, attack_iterations=1000,
                   attack_learning_rate=0.06, reconstruction_loss='cosine_sim', priors=None, epoch_matching_prior=None,
                   initialization_mode='uniform', softmax_trick=True, gumbel_softmax_trick=False, temperature_mode=None,
                   sigmoid_trick=False, sign_trick=True, apply_projection_to_features=None, device=None):
    """
    FedAVG attack following Dimitrov et al. 2022.
    """
    if device is None:
        device = dataset.device

    # attack setups
    rec_loss_function = {
        'squared_error': _squared_error_loss,
        'cosine_sim': _cosine_similarity_loss,
        'weighted_combined': _weighted_CS_SE_loss,
        'norm_weighted_combined': _gradient_norm_weighted_CS_SE_loss
    }

    initialization = {
        'uniform': _uniform_initialization,
        'gaussian': _gaussian_initialization,
        'mean': _mean_initialization,
        'dataset_sample': _dataset_sample_initialization,
        'likelihood_sample': _likelihood_prior_sample_initialization,
        'mixed': _mixed_initialization,
        'best_sample': _best_sample_initialization
    }

    temperature_configs = {
        'cool': (1000., 0.98),
        'constant': (1., 1.),
        'heat': (0.1, 1.01)
    }

    if reconstruction_loss not in list(rec_loss_function.keys()):
        raise NotImplementedError(
            f'The desired loss function is not implemented, available loss function are: {list(rec_loss_function.keys())}')

    final_reconstructions_per_client = []
    final_loss_per_client = []

    # we will go by attacked client and then completely restart every time
    for attacked_client, (attacked_client_params, ground_truth_data, ground_truth_labels) in enumerate(zip(attacked_clients_params, per_client_ground_truth_data, per_client_ground_truth_labels)):
        # fix the client network and extract its starting parameters
        original_params = [param.detach().clone() for param in original_net.parameters()]
        true_two_point_gradient = [(original_param - new_param).detach().clone() for original_param, new_param in zip(original_params, attacked_client_params)]

        # we reconstruct independently in each epoch and aggregate in the end, as per Dimitrov et al.
        # initialize the data
        torch.set_printoptions(sci_mode=False)
        
        with open(f'clients_data/reconstr_and_GT/inversion/normal/reconstructions_ground_truths_0.pkl', 'rb') as file:
            recn_gt = pickle.load(file)    
                
        reconstructions_1 = recn_gt['reconstructions']  # reconstructions[0][0] -->(4,)
        ground_truths_1 = recn_gt['ground_truths']
        # print("recn_gt val \n",reconstructions_1[0][0][2:3,:100])

        # print("reconstructions_1 shape ",reconstructions_1[0][0].shape)
                
            
        with open('clients_data/processed_data/directInputX_hello.pkl', 'rb') as f:
            loaded_dataloader = pickle.load(f)
        
        # ---------------- Uniform -----------------
                
        tensor = reconstructions_1[0][0]        

        # tensor = loaded_dataloader

        tensor_min = tensor.min()
        tensor_max = tensor.max()
        print(tensor_min,"::og Recon:", tensor_max)
        print(ground_truths_1[0][0].min() ,"::og GT:", ground_truths_1[0][0].max() )

        """
        Exp1:
        1. inversion+ uni_init : 71.4%

        normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)   # this is for 1nd, 2nd, 5th attempt , 6th attamp(tableak+sigmoid_off)
        uniform_tensor = tensor_min + normalized_tensor * (tensor_max - tensor_min)  # this is for 3rd attempt
        rand_tensor = (normalized_tensor -tensor_min) / (tensor_max - tensor_min)   # this is for 4th attempt

        Exp2:
        1. Tableak + uni_init +without sigmoid trick: 83.9 % 
        normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min) : 100%

        """

        normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)   # this is for 1nd, 2nd, 5th attempt , 6th attamp(tableak+sigmoid_off)
        # print("normalized_tensor ",normalized_tensor[2:3,:100])
    
        # uniform_tensor = tensor_min + normalized_tensor * (tensor_max - tensor_min)  # this is for 3rd attempt
        a, b = -1, 20  # Modify these if needed

        # Transform to uniform distribution within [a, b]
        uniform_tensor = a + (tensor * (b - a))
        # print("uniform_tensor val ",uniform_tensor[2:3,:100])
        
        exp_tensor=torch.exp(normalized_tensor)
        # exp_tensor=torch.exp(uniform_tensor)   # 60% with this: uniform_tensor1

        
        tensor_min1 = exp_tensor.min()
        tensor_max1 = exp_tensor.max()

        normalized_tensor1 = (exp_tensor - tensor_min1) / (tensor_max1 - tensor_min1)
        normalized_tensor2 = (exp_tensor - tensor_min) / (tensor_max - tensor_min)

        # uniform_tensor1 = a + (exp_tensor * (b - a))
        uniform_tensor1 = tensor_min1 + (exp_tensor * (tensor_max1 - tensor_min1))

        
        # sig_moid=1 / (1 + torch.exp(-uniform_tensor1))
        # print(f"After exp_tensor, Min {exp_tensor.min()} and Max {exp_tensor.max()} \n ",exp_tensor[2:3,:100])
        # print(f"After sig_moid, Min {sig_moid.min()} and Max {sig_moid.max()} \n ",sig_moid[2:3,:100])
        # print(f"After normalized_tensor1, Min {normalized_tensor1.min()} and Max {normalized_tensor1.max()} \n ",normalized_tensor1[2:3,:100])
        # print(f"After normalized_tensor2, Min {normalized_tensor2.min()} and Max {normalized_tensor2.max()} \n ",normalized_tensor2[2:3,:100])
        # print(f"After uniform_tensor1, Min {uniform_tensor1.min()} and Max {uniform_tensor1.max()} \n ",uniform_tensor1[2:3,:100])
        
        
        # uniform_tensor = tensor_min + torch.rand_
        # like(normalized_tensor) * (tensor_max - tensor_min)

        # rand_tensor = (tensor - tensor_min) / (tensor_max - tensor_min) 
  
        # print("uniform_tensor val ",uniform_tensor[2:3,:100])
        # print("scaled_noisy_tensor val ",scaled_noisy_tensor[2:3,:100])
        # print("normalized_tensor val \n",normalized_tensor[2:3,:100])
        # print("rand_tensor val \n",rand_tensor[2:3,:100])
        # print("loaded_dataloader val \n",loaded_dataloader[2:3,:100])
        # mean = 0.0
        # std = 1.0
        # noise = torch.normal(mean, std, size=normalized_tensor.shape)
       
        # low, high = -0.1, 0.1  # Noise values will be sampled between -0.1 and 0.1

        # noise = torch.empty_like(normalized_tensor).uniform_(low, high)
        # noisy_matrix = tensor + noise
        # print("noisy_matrix val \n",noisy_matrix[2:3,:100])
        #------------- Guss --------------
        # tensor = loaded_dataloader
        # mean = tensor.mean()
        # std = tensor.std()
        # normalized_tensor = (tensor - mean) / std
        # normalized_tensor = (normalized_tensor - tensor_min) / (tensor_max - tensor_min) 

        
        # print("ground_truth_data.shape::",ground_truth_data.shape)
        # print("normalized tensor  values",normalized_tensor[2:3,:25])

        reconstructed_data_per_epoch = [initialization[initialization_mode](ground_truth_data, dataset, device) for _ in range(n_local_epochs)]
        print("default initialization_mode is used")

        # print("n_local_epochs ", n_local_epochs)        
        # reconstructed_data_per_epoch = [ground_truth_data for _ in range(n_local_epochs)]

        # reconstructed_data_per_epoch = [reconstructions_1[0][0].detach().requires_grad_() for _ in range(n_local_epochs)]
        # print("reconstructions_1 is used")

        # reconstructed_data_per_epoch = [normalized_tensor.detach().requires_grad_() for _ in range(n_local_epochs)]
        # print("normalized_tensor is used")

        # reconstructed_data_per_epoch = [uniform_tensor.detach().requires_grad_() for _ in range(n_local_epochs)]
        # print("uniform_tensor is used")
        
        # reconstructed_data_per_epoch = [uniform_tensor1.detach().requires_grad_() for _ in range(n_local_epochs)]
        # print("uniform_tensor1 is used")

        # reconstructed_data_per_epoch = [exp_tensor.detach().requires_grad_() for _ in range(n_local_epochs)]
        # print("exp_tensor is used")
        
        # reconstructed_data_per_epoch = [normalized_tensor1.detach().requires_grad_() for _ in range(n_local_epochs)]
        # print("normalized_tensor1 is used")
        
        # reconstructed_data_per_epoch = [normalized_tensor2.detach().requires_grad_() for _ in range(n_local_epochs)]
        # print("normalized_tensor2 is used")

        # reconstructed_data_per_epoch = [sig_moid.detach().requires_grad_() for _ in range(n_local_epochs)]   
        # print("sig_moid is used")     
        
        # reconstructed_data_per_epoch = [loaded_dataloader.requires_grad_() for _ in range(n_local_epochs)]
        # print("loaded_dataloader is used")

        for reconstructed_data in reconstructed_data_per_epoch:
            reconstructed_data.requires_grad = True

        # print("GT  \n",ground_truth_data[2:3,:100])
        # print("recon \n",loaded_dataloader[2:3,:100])

        # print("recon shape",loaded_dataloader.shape)
        # print("Gt shape",ground_truth_data.shape)
        previous_params = [tensor.clone().detach() for tensor in reconstructed_data_per_epoch]
        weight_changes = []
        # optimizer = torch.optim.AdamW(reconstructed_data_per_epoch, lr=attack_learning_rate)
        # optimizer = torch.optim.AdamW(reconstructed_data_per_epoch, lr=0.08, betas=(0.95, 0.99), weight_decay=0.01) 

        
        lr_order = [0.08, 0.077, 0.065, 0.053, 0.04, 0.035,0.03] #70.4 with 256/32
        betas_order = [(0.96, 0.99), (0.95, 0.99),(0.94, 0.99), (0.93, 0.989),(0.92, 0.987),(0.918, 0.985),(0.915, 0.982)] #70.4
        weight_decay_order=[0.01,0.01,0.01,0.0085,0.0084,0.007,0.007] #old


        # lr_order=[0.009,0.01,0.013,0.015,0.017, 0.02]
        # lr_order=[0.013,0.028,0.039,0.049,0.059,0.068]
        # lr_order = [0.02, 0.017, 0.015, 0.013, 0.01, 0.009]
        # lr_order = [0.03, 0.027, 0.025, 0.023, 0.02, 0.01, 0.009] #70.9

        # lr_order = [0.08, 0.077, 0.065, 0.053, 0.04, 0.03,0.029] # 70.2...
        
        
        # betas_order = [(0.96, 0.999), (0.95, 0.997),(0.94, 0.995), (0.93, 0.989),(0.92, 0.987),(0.91, 0.985),(0.9, 0.982)] # 70.2..


        # betas_order = [(0.96, 0.999), (0.95, 0.996),(0.94, 0.994), (0.93, 0.987),(0.92, 0.984),(0.91, 0.982),(0.89, 0.98)]

        # weight_decay_order=[0.01,0.01,0.01,0.0085,0.0084,0.007,0.0065]

        # weight_decay_order = [0.01, 0.01, 0.009, 0.008, 0.0075, 0.0065, 0.0055]


        optimizer = torch.optim.AdamW(reconstructed_data_per_epoch, lr=lr_order[0], betas=betas_order[0], weight_decay=weight_decay_order[0]) 
        
        T = temperature_configs[temperature_mode][0]
        weight_changes = [] 
        
        for it in range(attack_iterations):

            if it==250:
                optimizer = torch.optim.AdamW(reconstructed_data_per_epoch, lr=lr_order[1], betas=betas_order[1] , weight_decay=weight_decay_order[1]) 
                print("at 250 Params are changed")
            if it==500:
                optimizer = torch.optim.AdamW(reconstructed_data_per_epoch, lr=lr_order[2], betas=betas_order[2], weight_decay=weight_decay_order[2])
                print("at 500 Params are changed")
            # if it==750:
            #     optimizer = torch.optim.AdamW(reconstructed_data_per_epoch, lr=lr_order[3], betas=betas_order[3], weight_decay=weight_decay_order[3]) 
            #     print("at 750 Params are changed")
            if it==1000:                
                optimizer = torch.optim.AdamW(reconstructed_data_per_epoch, lr=lr_order[4], betas=betas_order[4], weight_decay=weight_decay_order[4])                
                print("at 1000 Params are changed")            
            if it==1300:                                 
                optimizer = torch.optim.AdamW(reconstructed_data_per_epoch, lr=lr_order[5], betas=betas_order[5], weight_decay=weight_decay_order[5])                
                print("at 1300 Params are changed")
            # if it==1600:                                 
            #     optimizer = torch.optim.AdamW(reconstructed_data_per_epoch, lr=lr_order[6], betas=betas_order[6], weight_decay=weight_decay_order[6])                
            #     print("at 1600 Params are changed")

            optimizer.zero_grad()

            original_net.zero_grad()
            client_net = MetaMonkey(copy.deepcopy(original_net))
            criterion = torch.nn.CrossEntropyLoss()

            resulting_two_point_gradient, regularizer = simulate_local_training_for_attack(
                client_net=client_net,
                lr=lr,
                criterion=criterion,
                dataset=dataset,
                labels=ground_truth_labels,
                original_params=original_params,
                reconstructed_data_per_epoch=reconstructed_data_per_epoch,
                local_batch_size=local_batch_size,
                priors=priors,
                epoch_matching_prior=epoch_matching_prior,
                softmax_trick=softmax_trick,
                gumbel_softmax_trick=gumbel_softmax_trick,
                sigmoid_trick=sigmoid_trick,
                apply_projection_to_features=apply_projection_to_features,
                temperature=T
            )

            # calculate the final objective
            loss = rec_loss_function[reconstruction_loss](resulting_two_point_gradient, true_two_point_gradient, device)
            loss += regularizer
            
            loss.backward()
            # norm=torch.nn.utils.clip_grad_norm_(reconstructed_data_per_epoch, max_norm=5.0)

            if it % 100 == 0:
                total_grad = 0
                total_params = 0               
                total_grad_client=0
                for name, param in client_net.named_parameters():
                    if param.grad is not None:
                        total_grad_client += param.grad.abs().sum().item() 
                        # total_params += param.grad.numel()
                  
                for reconstructed_data in reconstructed_data_per_epoch:
                    if reconstructed_data.grad is not None:
                        total_grad += reconstructed_data.grad.abs().sum().item() 
                        # total_params += reconstructed_data.grad.numel()
                     
                with torch.no_grad():
                    total_weight_change = 0
                    for i, reconstructed_data in enumerate(reconstructed_data_per_epoch):
                        weight_change = torch.norm(reconstructed_data - previous_params[i], p=2).item()
                        total_weight_change += weight_change
                        previous_params[i] = reconstructed_data.clone().detach()  

                    weight_changes.append(total_weight_change)  

                print(f"Iteration {it}, Loss: {loss.item():.5f},Reconstr Total Gradient: {total_grad:.5f}, "
                    f"Weight Change: {total_weight_change:.5f},Client Total Gradient: {total_grad_client:.5f}")
              

            if sign_trick:
                for reconstructed_data in reconstructed_data_per_epoch:
                    reconstructed_data.grad.sign_()

            optimizer.step()

            # adjust the temperature
            T *= temperature_configs[temperature_mode][1]

        # if we used the sigmoid trick, we reapply it
        if sigmoid_trick:
            sigmoid_reconstruction = [continuous_sigmoid_bound(rd, dataset=dataset, T=T) for rd in reconstructed_data_per_epoch]
            reconstructed_data_per_epoch = sigmoid_reconstruction

        # after the optimization has finished for the given client, we project and match the data
        epoch_pooling = 'soft_avg+softmax' if softmax_trick or gumbel_softmax_trick else 'soft_avg'
        final_reconstruction = pooled_ensemble([reconstructed_data.clone().detach() for reconstructed_data in reconstructed_data_per_epoch],
                                               reconstructed_data_per_epoch[0].clone().detach(), dataset,
                                               pooling=epoch_pooling)
        final_reconstructions_per_client.append(final_reconstruction)

        # with the aggregated datapoint, we can finally run it again through the process to record its loss
        final_reconstruction_projected = dataset.project_batch(final_reconstruction, standardized=dataset.standardized)
        client_net = MetaMonkey(copy.deepcopy(original_net))
        criterion = torch.nn.CrossEntropyLoss()
        final_resulting_two_point_gradient, _ = simulate_local_training_for_attack(
                client_net=client_net,
                lr=lr,
                criterion=criterion,
                dataset=dataset,
                labels=ground_truth_labels,
                original_params=original_params,
                reconstructed_data_per_epoch=[final_reconstruction_projected for _ in range(n_local_epochs)],
                local_batch_size=local_batch_size,
                priors=None,
                softmax_trick=softmax_trick,
                gumbel_softmax_trick=gumbel_softmax_trick,
                apply_projection_to_features=apply_projection_to_features,
                temperature=T
        )
        final_loss = rec_loss_function[reconstruction_loss](final_resulting_two_point_gradient, true_two_point_gradient, device)

        final_loss_per_client.append(final_loss.detach().item())

    return final_reconstructions_per_client, final_loss_per_client


def fed_avg_attack_parallelized_over_clients(original_net, attacked_clients_params, n_local_epochs, local_batch_size,
                                             lr, dataset, per_client_ground_truth_data, per_client_ground_truth_labels,
                                             metadata_path='metadata', attack_iterations=1000, attack_learning_rate=0.06,
                                             reconstruction_loss='cosine_sim', priors=None, epoch_matching_prior=None,
                                             initialization_mode='uniform', softmax_trick=True, sigmoid_trick=False,
                                             gumbel_softmax_trick=False, temperature_mode=None, sign_trick=True,
                                             apply_projection_to_features=None, max_n_cpus=50, first_cpu=0, device=None):
    """
    FedAVG attack following Dimitrov et al. 2022.
    """
    if device is None:
        device = dataset.device

    n_clients = len(per_client_ground_truth_data)

    # create the working directory
    metadata_path += f'{np.random.randint(0, 1000, 1).item()}'  # disturb by a random integer to avoid conflicts
    os.makedirs(metadata_path, exist_ok=True)

    # save everything that is needed for the per client parallelization
    with open(f'{metadata_path}/original_net.pickle', 'wb') as f:
        pickle.dump(original_net, f)
    with open(f'{metadata_path}/attacked_clients_params.pickle', 'wb') as f:
        pickle.dump(attacked_clients_params, f)
    with open(f'{metadata_path}/per_client_ground_truth_data.pickle', 'wb') as f:
        pickle.dump(per_client_ground_truth_data, f)
    with open(f'{metadata_path}/per_client_ground_truth_labels.pickle', 'wb') as f:
        pickle.dump(per_client_ground_truth_labels, f)
    with open(f'{metadata_path}/dataset.pickle', 'wb') as f:
        pickle.dump(dataset, f)
    with open(f'{metadata_path}/apply_projection_to_features.pickle', 'wb') as f:
        pickle.dump(apply_projection_to_features, f)
    # prior could also be None
    if priors is not None:
        with open(f'{metadata_path}/priors.pickle', 'wb') as f:
            pickle.dump(priors, f)
    # epoch matching prior could also be None
    if epoch_matching_prior is not None:
        with open(f'{metadata_path}/epoch_matching_prior.pickle', 'wb') as f:
            pickle.dump(epoch_matching_prior, f)

    # call all scripts to complete the individual inversions
    random_seeds = np.random.randint(0, 15000, size=n_clients)
    split_seeds = np.array_split(random_seeds, int(np.ceil(n_clients/max_n_cpus)))
    split_client_ranges = np.array_split(np.arange(n_clients), int(np.ceil(n_clients/max_n_cpus)))
    for split_seed, split_client_range in zip(split_seeds, split_client_ranges):
        process_pool = multiprocessing.Pool(processes=n_clients)
        all_processes_to_execute = []
        for idx, (client, rs) in enumerate(zip(split_client_range, split_seed)):
            command = f'taskset -c {first_cpu + idx} python single_fedavg_inversion_for_client.py --random_seed {rs} ' \
                    f'--client {client} --metadata_path {metadata_path} --lr {lr} --local_batch_size {local_batch_size} ' \
                    f'--n_local_epochs {n_local_epochs} --attack_learning_rate {attack_learning_rate} ' \
                    f'--attack_iterations {attack_iterations} --temperature_mode {temperature_mode} ' \
                    f'--initialization_mode {initialization_mode} --reconstruction_loss {reconstruction_loss} ' \
                    f'--device {device}'
            if softmax_trick:
                command += ' --softmax_trick'
            if gumbel_softmax_trick:
                command += ' --gumbel_softmax_trick'
            if sign_trick:
                command += ' --sign_trick'
            if sigmoid_trick:
                command += ' --sigmoid_trick'
            all_processes_to_execute.append(command)
        process_pool.map(caller, tuple(all_processes_to_execute))

    # after all inversions have been executed, load the data and organize it in the desired format
    final_reconstructions_per_client, final_loss_per_client = [], []
    for client in range(n_clients):
        final_reconstructions_per_client.append(torch.tensor(np.load(f'{metadata_path}/single_inversion_{client}.npy')).to(device))
        final_loss_per_client.append(np.load(f'{metadata_path}/single_inversion_loss_{client}.npy').item())

    # delete the metadata
    os.system(f'rm -rf {metadata_path}')

    return final_reconstructions_per_client, final_loss_per_client

def prepare_qat_model(model, backend='qnnpack'):
    """
    Prepare model for Quantization Aware Training
    """
    model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    torch.backends.quantized.engine = backend
    model.fuse_model()
    torch.quantization.prepare_qat(model, inplace=True)
    return model

def collect_quant_params(net, quantized_net):

    processed_params = []

    # Clone and detach the original parameters
    original_params = [param.clone().detach() for param in net.parameters()]

    for original_param in original_params:
        matched = False 
        for name, module in quantized_net.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight
                weight_tensor = weight() if callable(weight) else weight

                if isinstance(weight_tensor, torch.Tensor):
                    # Dequantize if necessary and detach the tensor
                    weight_tensor = weight_tensor.dequantize().detach() if weight_tensor.is_quantized else weight_tensor.detach()
                    
                    if weight_tensor.shape == original_param.shape:
                        processed_params.append(weight_tensor)
                        matched = True
                        break 

            if hasattr(module, 'bias') and module.bias is not None:
                bias = module.bias
                bias_tensor = bias() if callable(bias) else bias

                if isinstance(bias_tensor, torch.Tensor):
                    bias_tensor = bias_tensor.detach()
                   
                    if bias_tensor.shape == original_param.shape:
                        processed_params.append(bias_tensor)
                        matched = True
                        break 

        if not matched:
            print(f"Warning: No matching parameter found for shape {original_param.shape}")
    
    return processed_params
def train_and_attack_fed_avg(net, n_clients, n_global_epochs, n_local_epochs, local_batch_size, lr, dataset, shuffle=False,
                             attacked_clients=None, attack_iterations=1000, reconstruction_loss='cosine_sim', priors=None,
                             epoch_matching_prior=None, post_selection=1, attack_learning_rate=0.06, return_all=False,
                             pooling=None, perfect_pooling=False, initialization_mode='uniform', softmax_trick=True,
                             gumbel_softmax_trick=False, sigmoid_trick=False, temperature_mode='constant',
                             sign_trick=True, fish_for_features=None, device=None, verbose=False, max_n_cpus=50, first_cpu=0,
                             max_client_dataset_size=None, parallelized=False, metadata_path='metadata',state_name=0):
    
    
    if device is None:
        device = dataset.device

    if attacked_clients is None:
        attacked_clients = []
    elif attacked_clients == 'all':
        attacked_clients = list(np.arange(n_clients))

    if max_client_dataset_size is None:
        max_client_dataset_size = len(dataset)

    # attack data and training statistics
    per_global_epoch_per_client_reconstructions = []
    per_global_epoch_per_client_ground_truth = []
    training_data = np.zeros((n_global_epochs, 2))

    # get the data and then split it into client datasets
    # if shuffle:
    #     dataset.shuffle()
        
    Xtrain, ytrain = dataset.get_Xtrain(), dataset.get_ytrain()
    split_size = min(max_client_dataset_size, int(np.ceil(Xtrain.size()[0] / n_clients)))  # ceiling
    Xtrain_splits = [Xtrain[i*split_size:min(int(Xtrain.size()[0]), (i+1)*split_size)].clone().detach() for i in range(n_clients)]
    ytrain_splits = [ytrain[i*split_size:min(int(Xtrain.size()[0]), (i+1)*split_size)].clone().detach() for i in range(n_clients)]

    print("Data use for training:: ", Xtrain_splits[0].shape)
    # Xtrain, ytrain = dataset.get_Xtrain(state_name), dataset.get_ytrain(state_name)
    # print("Data used for training:", [Xtrain[:split_size].clone().detach().shape], ytrain_splits)


    # instantiate the loss
    criterion = torch.nn.CrossEntropyLoss()

    timer = Timer(n_global_epochs)
    Normal_flag= True
    
    defense_flag= False
    noise_scale=0.1

    fairness_flag= False
    senstive_attr=9 # 8: Race and 9: Sex
    
    defense_fairness_flag= False

    attack_bool= True

    # pre_trained_model_path="clients_data/clients_trained_model/pre_trained_model.pth"
    # print("pre_trained model is loaded")
    # state_dict = torch.load(pre_trained_model_path)
    # weights_dict = {k: v for k, v in state_dict.items() if 'weight' in k}
    # net.load_state_dict(weights_dict, strict=False)

    # training loop
    for global_epoch in range(n_global_epochs):
        timer.start()

        acc, bac = get_acc_and_bac(net, dataset.get_Xtest(), dataset.get_ytest())
        if verbose:
            print(f'Global Epochs: {global_epoch + 1}/{n_global_epochs}    Acc: {acc * 100:.2f}%    BAcc: {bac * 100:.2f}%    {timer}', end='\r')
        training_data[global_epoch] = acc, bac

        # create the current client net copies
        # client_nets = [copy.deepcopy(net) for _ in range(n_clients)]
        print("n_clients", n_clients)
        client_nets = [copy.deepcopy(net) for _ in range(n_clients)]
        
        # client_nets = [prepare_qat_model(client_net) for client_net in client_nets]

        # iterate through each client (this should be done in parallel in theory)
        for client, (client_X, client_y, client_net) in enumerate(zip(Xtrain_splits, ytrain_splits, client_nets)):
           
            # print(f"Data use for training:: for client: {client} :length:", ytrain_splits)   

            # print(f"Data use for training:: for client: {client} :length:", client_y.shape,"values",client_y)
            
            # do the local training for each client
            n_batches = int(np.ceil(client_X.size()[0] / local_batch_size))
            if Normal_flag is True:
                print("Normal training")
                print("n_batches is",n_batches)
                print("local_epoch is ",n_local_epochs)
                
                for local_epoch in range(n_local_epochs):
    
                    # complete an epoch
                    for b in range(n_batches):
                        current_batch_X = client_X[b * local_batch_size:min(int(client_X.size()[0]), (b+1)*local_batch_size)].clone().detach()
                        current_batch_y = client_y[b * local_batch_size:min(int(client_X.size()[0]), (b+1)*local_batch_size)].clone().detach()
    
                        outputs = client_net(current_batch_X)
                        loss = criterion(outputs, current_batch_y)
                        grad = torch.autograd.grad(loss, client_net.parameters(), retain_graph=True)
    
                        with torch.no_grad():
                            for param, param_grad in zip(client_net.parameters(), grad):
                                param -= lr * param_grad
                                
                    client_net.eval()
                    # quantized_net = torch.quantization.convert(client_net.eval(), inplace=False)
                    # quantized_net.eval()

                    with torch.no_grad():
                        val_running_loss = 0.0
                        val_correct = 0
                        val_total = 0
                    
                        inputs, labels = dataset.get_Xtest(), dataset.get_ytest()
                        # Ensure labels are 1D tensor of integers
                        # labels = labels.long()
                        val_n_batches = int(np.ceil(inputs.size()[0] / local_batch_size))
                    
                        for b in range(val_n_batches):
                            val_batch_X = inputs[b * local_batch_size:min(int(inputs.size()[0]), (b + 1) * local_batch_size)].clone().detach()
                            val_batch_y = labels[b * local_batch_size:min(int(labels.size()[0]), (b + 1) * local_batch_size)].clone().detach()
                    
                            outputs = client_net(val_batch_X)
                    
                            # Ensure outputs are logits for CrossEntropyLoss
                            val_loss = criterion(outputs, val_batch_y)
                            val_running_loss += val_loss.item()
                    
                            # Use argmax for predicted class
                            predicted_classes = outputs.argmax(dim=1)
                            val_correct += (predicted_classes == val_batch_y).sum().item()
                            val_total += val_batch_y.size(0)
                    
                        val_epoch_loss = val_running_loss / len(inputs)
                        val_accuracy = val_correct / val_total
                    
                        # print("val_accuracy and val_epoch_loss: ", val_accuracy, val_epoch_loss)
                        client_net.train()
                        
                model_path = f"clients_data/clients_trained_model/client{state_name}.pth"
                torch.save(client_net.state_dict(), model_path)
                print("Model Size:", os.path.getsize(model_path) / 1024, "KB")

            elif defense_flag is True:
                print("DP called")
                print("n_batches is",n_batches)
                print("local_epoch is ",n_local_epochs)                
                print("Noise scale is ", noise_scale)

                X_train, y_train = dataset.get_Xtrain(), dataset.get_ytrain()
                batch_size=local_batch_size
                n_samples= n_clients

                for j in range(n_samples):
                    for l in range(n_local_epochs):
                        timer.start()
                        
                        print(f'Noise Scale: {noise_scale}    Sample: {j+1}    Epoch: {l+1}    Acc: {100*acc:.1f}%    Bac: {100*bac:.1f}%    {timer}', end='\r')

                        permutation_indices = np.random.permutation(len(X_train))
                        X_train_permuted, y_train_permuted = X_train[permutation_indices].detach().clone(), y_train[permutation_indices].detach().clone()

                        for k in range(n_batches):

                            # optimizer.zero_grad()
                            client_net.zero_grad()
                            X_batch, y_batch = X_train_permuted[k*batch_size:max(len(X_train_permuted), (k+1)*batch_size)], y_train_permuted[k*batch_size:max(len(X_train_permuted), (k+1)*batch_size)]
                            
                            outputs=client_net(X_batch)

                            loss = criterion(outputs, y_batch)
                            
                            # defense
                            grad = [g.detach() for g in torch.autograd.grad(loss, client_net.parameters())]            
                            perturbed_grad = dp_defense(grad, noise_scale) if noise_scale > 0 else grad

                            with torch.no_grad():
                                # make the update
                                for p, g in zip(client_net.parameters(), perturbed_grad):
                                    p.data = p.data - lr * g

                        # X_test, y_test = dataset.get_Xtest(), dataset.get_ytest()                                    
                        # acc, bac = get_acc_and_bac(client_net, X_test, y_test)

                        client_net.eval()
                        # quantized_net = torch.quantization.convert(client_net.eval(), inplace=False)
                        # quantized_net.eval()
                        
                        with torch.no_grad():
                            val_running_loss = 0.0
                            val_correct = 0
                            val_total = 0

                            inputs, labels = dataset.get_Xtest(), dataset.get_ytest()
                            val_n_batches = int(np.ceil(inputs.size()[0] / local_batch_size))
                            
                            for b in range(val_n_batches):
                                val_batch_X = inputs[b * local_batch_size:min(int(inputs.size()[0]), (b + 1) * local_batch_size)].clone().detach()
                                val_batch_y = labels[b * local_batch_size:min(int(labels.size()[0]), (b + 1) * local_batch_size)].clone().detach()

                                outputs = client_net(val_batch_X)
                                val_loss = criterion(outputs, val_batch_y)
                                val_running_loss += val_loss.item()


                                if outputs.size(1) == 1:
                                    predicted_classes = (outputs > 0.5).float().squeeze(1)
                                else:  # Multi-class output
                                    predicted_classes = outputs.argmax(dim=1)                 
                                    
                                val_correct += (predicted_classes == val_batch_y).sum().item()
                                val_total += val_batch_y.size(0)

                            val_epoch_loss = val_running_loss / len(inputs)
                            val_accuracy = val_correct / val_total

                            print("val_accuracy and val_epoch_loss: ", val_accuracy,val_epoch_loss )
                            client_net.train()

                model_path = f"clients_data/client_DP_trained_model/client{state_name}.pth"
                torch.save(client_net.state_dict(), model_path)
                print("Model Size:", os.path.getsize(model_path) / 1024, "KB")
                
            elif fairness_flag is True:
                
                print("fairess called")                
                print("n_batches is",n_batches)
                print("local_epoch is ",n_local_epochs)                                            
                
                for local_epoch in range(n_local_epochs):

                    # complete an epoch
                    for b in range(n_batches):
                        
                        current_batch_X = client_X[b * local_batch_size:min(int(client_X.size()[0]), (b+1)*local_batch_size)].clone().detach()
                        current_batch_y = client_y[b * local_batch_size:min(int(client_X.size()[0]), (b+1)*local_batch_size)].clone().detach()

                        # for two groups individually
                        fair_loss_grp_1=FairLoss(torch.nn.BCELoss(), current_batch_X[:, 8].detach().unique(), 'accuracy')
                        fair_loss_grp_2=FairLoss(torch.nn.BCELoss(), current_batch_X[:, 9].detach().unique(), 'accuracy')
                        
                        outputs = client_net(current_batch_X)                   
                        loss_1 = criterion(outputs, current_batch_y)        
                        
                        # loss_intersectional=intersectional_loss(current_batch_X,outputs,current_batch_y)                    
                        
                        loss_2 = fair_loss_grp_1(current_batch_X[:, 8], torch.sigmoid(outputs[:, 1]), current_batch_y.float())
                        loss_3 = fair_loss_grp_2(current_batch_X[:, 9], torch.sigmoid(outputs[:, 1]), current_batch_y.float())
                        
                        final_loss=loss_1+loss_2+loss_3

                        # final_loss.backward()
            
                        grad = torch.autograd.grad(final_loss, client_net.parameters(), retain_graph=True)
                        
                        with torch.no_grad():
                            for param, param_grad in zip(client_net.parameters(), grad):
                                param -= lr * param_grad

                    client_net.eval()
                    # quantized_net = torch.quantization.convert(client_net.eval(), inplace=False)
                    # quantized_net.eval()                    
                    
                    with torch.no_grad():
                        val_running_loss = 0.0
                        val_correct = 0
                        val_total = 0

                        inputs, labels = dataset.get_Xtest(), dataset.get_ytest()
                        # labels = labels.unsqueeze(1).float() 
                        # labels = labels.long()
                        val_n_batches = int(np.ceil(inputs.size()[0] / local_batch_size))

                        for b in range(val_n_batches):
                            val_batch_X = inputs[b * local_batch_size:min(int(inputs.size()[0]), (b + 1) * local_batch_size)].clone().detach()
                            val_batch_y = labels[b * local_batch_size:min(int(labels.size()[0]), (b + 1) * local_batch_size)].clone().detach()

                            outputs = client_net(val_batch_X)
                            val_loss = criterion(outputs, val_batch_y)
                            val_running_loss += val_loss.item()
                            
                            # If output has a single class probability
                            if outputs.size(1) == 1:
                                predicted_classes = (outputs > 0.5).float().squeeze(1)
                            else:  # Multi-class output
                                predicted_classes = outputs.argmax(dim=1)     
                                
                            # predicted_classes = (outputs > 0.5).float()
                            
                            val_correct += (predicted_classes == val_batch_y).sum().item()
                            val_total += val_batch_y.size(0)

                        val_epoch_loss = val_running_loss / len(inputs)
                        val_accuracy = val_correct / val_total

                        print("val_accuracy and val_epoch_loss: ", val_accuracy,val_epoch_loss )
                        client_net.train()     
                
                model_path = f"clients_data/clients_fair_trained_model/client{state_name}.pth"
                torch.save(client_net.state_dict(), model_path)
                print("Model Size:", os.path.getsize(model_path) / 1024, "KB")
 
            elif defense_fairness_flag is True:
                print("defense and fairness")

                noise_scale=0.1
                X_train, y_train = dataset.get_Xtrain(), dataset.get_ytrain()
                batch_size=local_batch_size
                n_samples= n_clients

                for j in range(n_samples):
                    for l in range(n_local_epochs):
                        timer.start()
                        
                        print(f'Noise Scale: {noise_scale}    Sample: {j+1}    Epoch: {l+1}    Acc: {100*acc:.1f}%    Bac: {100*bac:.1f}%    {timer}', end='\r')

                        permutation_indices = np.random.permutation(len(X_train))
                        X_train_permuted, y_train_permuted = X_train[permutation_indices].detach().clone(), y_train[permutation_indices].detach().clone()

                        for k in range(n_batches):

                            # optimizer.zero_grad()
                            client_net.zero_grad()
                            X_batch, y_batch = X_train_permuted[k*batch_size:max(len(X_train_permuted), (k+1)*batch_size)], y_train_permuted[k*batch_size:max(len(X_train_permuted), (k+1)*batch_size)]
                            
                            # fair_loss=FairLoss(torch.nn.BCELoss(), X_batch[:, senstive_attr].detach().unique(), 'accuracy')

                            # for two groups individually
                            fair_loss_grp_1=FairLoss(torch.nn.BCELoss(), X_batch[:, 8].detach().unique(), 'accuracy')
                            fair_loss_grp_2=FairLoss(torch.nn.BCELoss(), X_batch[:, 9].detach().unique(), 'accuracy')

                            outputs=client_net(X_batch)
                            # y_batch=y_batch.unsqueeze(1).float()
                            # loss = criterion(outputs, y_batch)

                            loss_1 = criterion(outputs, y_batch)
                            # print(X_batch)
                                                                                
                            # loss_intersectional=intersectional_loss(X_batch,outputs,y_batch)

                            # loss_2 = fair_loss(X_batch[:, senstive_attr],outputs,y_batch)   
                            loss_2 = fair_loss_grp_1(X_batch[:, 8], torch.sigmoid(outputs[:, 1]), y_batch.float())
                            loss_3 = fair_loss_grp_2(X_batch[:, 9], torch.sigmoid(outputs[:, 1]), y_batch.float())

                            
                            final_loss=loss_1+loss_2+loss_3

                            # defense
                            grad = [g.detach() for g in torch.autograd.grad(final_loss, client_net.parameters())]                                        
                            perturbed_grad = dp_defense(grad, noise_scale) if noise_scale > 0 else grad

                            with torch.no_grad():
                                # make the update
                                for p, g in zip(client_net.parameters(), perturbed_grad):
                                    p.data = p.data - lr * g
                
                        client_net.eval()
                        # quantized_net = torch.quantization.convert(client_net.eval(), inplace=False)
                        # quantized_net.eval()  
                        
                        with torch.no_grad():
                            val_running_loss = 0.0
                            val_correct = 0
                            val_total = 0

                            inputs, labels = dataset.get_Xtest(), dataset.get_ytest()
                            # labels = labels.unsqueeze(1).float()                        
                            val_n_batches = int(np.ceil(inputs.size()[0] / local_batch_size))

                            for b in range(val_n_batches):
                                val_batch_X = inputs[b * local_batch_size:min(int(inputs.size()[0]), (b + 1) * local_batch_size)].clone().detach()
                                val_batch_y = labels[b * local_batch_size:min(int(labels.size()[0]), (b + 1) * local_batch_size)].clone().detach()

                                outputs = client_net(val_batch_X)
                                val_loss = criterion(outputs, val_batch_y)
                                val_running_loss += val_loss.item()
                                
                                if outputs.size(1) == 1:
                                    predicted_classes = (outputs > 0.5).float().squeeze(1)
                                else:  # Multi-class output
                                    predicted_classes = outputs.argmax(dim=1)  
                                    
                                val_correct += (predicted_classes == val_batch_y).sum().item()
                                val_total += val_batch_y.size(0)

                            val_epoch_loss = val_running_loss / len(inputs)
                            val_accuracy = val_correct / val_total

                            print("val_accuracy and val_epoch_loss: ", val_accuracy,val_epoch_loss )
                            client_net.train()

                model_path = f"clients_data/clients_DP_Fair_trained_model/client{state_name}.pth"
                torch.save(client_net.state_dict(), model_path)
                print("Model Size:", os.path.getsize(model_path) / 1024, "KB")               
                
        # extract the parameters from the client nets
        # processed_params=collect_quant_params(net, quantized_net)

        # clients_params = processed_params
        clients_params = [[param.clone().detach() for param in client_net.parameters()] for client_net in client_nets]
        print("len:",len (clients_params))

        # -------------- ATTACK -------------- #
        per_client_all_reconstructions = [[] for _ in range(len(attacked_clients))]
        per_client_best_reconstructions = [None for _ in range(len(attacked_clients))]
        per_client_best_scores = [None for _ in range(len(attacked_clients))]
        per_client_ground_truth_data = [Xtrain_splits[attacked_client].detach().clone() for attacked_client in attacked_clients]
        per_client_ground_truth_labels = [ytrain_splits[attacked_client].detach().clone() for attacked_client in attacked_clients]
        attacked_clients_params = [[param.clone().detach() for param in clients_params[attacked_client]] for attacked_client in attacked_clients]

        # attacked_clients_params=[clients_params]

        for _ in range(post_selection):

            if parallelized:
                print("parallelized")
                per_client_candidate_reconstructions, per_client_final_losses = fed_avg_attack_parallelized_over_clients(
                    original_net=copy.deepcopy(net),
                    attacked_clients_params=attacked_clients_params,
                    attack_iterations=attack_iterations,
                    attack_learning_rate=attack_learning_rate,
                    n_local_epochs=n_local_epochs,
                    local_batch_size=local_batch_size,
                    lr=lr,
                    dataset=dataset,
                    per_client_ground_truth_data=per_client_ground_truth_data,
                    per_client_ground_truth_labels=per_client_ground_truth_labels,
                    reconstruction_loss=reconstruction_loss,
                    priors=priors,
                    epoch_matching_prior=epoch_matching_prior,
                    initialization_mode=initialization_mode,
                    softmax_trick=softmax_trick,
                    gumbel_softmax_trick=gumbel_softmax_trick,
                    sigmoid_trick=sigmoid_trick,
                    temperature_mode=temperature_mode,
                    sign_trick=sign_trick,
                    apply_projection_to_features=fish_for_features,
                    max_n_cpus=max_n_cpus,
                    first_cpu=first_cpu,
                    device=device,
                    metadata_path=metadata_path
                )
            else:
                print("parallelized Off")
                per_client_candidate_reconstructions, per_client_final_losses = fed_avg_attack(
                    original_net=copy.deepcopy(net),
                    attacked_clients_params=attacked_clients_params,
                    attack_iterations=attack_iterations,
                    attack_learning_rate=attack_learning_rate,
                    n_local_epochs=n_local_epochs,
                    local_batch_size=local_batch_size,
                    lr=lr,
                    dataset=dataset,
                    per_client_ground_truth_data=per_client_ground_truth_data,
                    per_client_ground_truth_labels=per_client_ground_truth_labels,
                    reconstruction_loss=reconstruction_loss,
                    priors=priors,
                    epoch_matching_prior=epoch_matching_prior,
                    initialization_mode=initialization_mode,
                    softmax_trick=softmax_trick,
                    gumbel_softmax_trick=gumbel_softmax_trick,
                    sigmoid_trick=sigmoid_trick,
                    temperature_mode=temperature_mode,
                    sign_trick=sign_trick,
                    apply_projection_to_features=fish_for_features,
                    device=device
                )

            # enter the results in the collectors
            for client_idx in range(len(attacked_clients)):
                per_client_all_reconstructions[client_idx].append(per_client_candidate_reconstructions[client_idx].detach().clone())
                if (per_client_best_scores[client_idx] is None) or (per_client_best_scores[client_idx] > per_client_final_losses[client_idx]):
                    per_client_best_scores[client_idx] = per_client_final_losses[client_idx]
                    per_client_best_reconstructions[client_idx] = per_client_candidate_reconstructions[client_idx].detach().clone()

        if return_all:
            per_global_epoch_per_client_reconstructions.append(per_client_all_reconstructions)
        elif pooling is not None:
            if perfect_pooling:
                per_client_pooled = [pooled_ensemble(all_reconstructions, ground_truth_data, dataset, pooling=pooling)
                                     for all_reconstructions, ground_truth_data in zip(per_client_all_reconstructions, per_client_ground_truth_data)]
            else:
                per_client_pooled = [pooled_ensemble(all_reconstructions, best_reconstruction, dataset, pooling=pooling)
                                     for all_reconstructions, best_reconstruction in zip(per_client_all_reconstructions, per_client_best_reconstructions)]
            per_global_epoch_per_client_reconstructions.append(per_client_pooled)
        else:
            per_global_epoch_per_client_reconstructions.append(per_client_best_reconstructions)
        per_global_epoch_per_client_ground_truth.append(per_client_ground_truth_data)
        # -------------- ATTACK END -------------- #

        # Continue the training
        # transpose the list
        transposed_clients_params = [[] for _ in range(len(clients_params[0]))]
        for client_params in clients_params:
            for i, param in enumerate(client_params):
                transposed_clients_params[i].append(param.clone().detach())

        # aggregate the params using mean aggregation
        aggregated_params = [torch.mean(torch.stack(params_over_clients), dim=0) for params_over_clients in transposed_clients_params]

        # write the new parameters into the main network
        with torch.no_grad():
            for param, agg_param in zip(net.parameters(), aggregated_params):
                param.copy_(agg_param)
        
        timer.end()
    timer.duration()

    return net, training_data, per_global_epoch_per_client_reconstructions, per_global_epoch_per_client_ground_truth
