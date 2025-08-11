import argparse
import copy
import glob
import json
import numpy as np
import os
import os.path
import os.path
import queue
import random
import shutil
import subprocess
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from concurrent.futures import ProcessPoolExecutor
from torch import optim
from torch.utils.data import DataLoader

from model_utils import loss_smoothed, loss_nll, get_std_opt, tied_featurize, pdb_to_jsonl, featurize_pdb, ProteinMPNN
from utils import  StructureDataset
from utils_2 import  extract_plddt_from_json, combined_loss_fn, combined_loss_fn, check_and_clear_folder, highfold, find_file_with_name
from utils_3 import fape_loss

scaler = torch.cuda.amp.GradScaler()
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def main(args):
    base_folder = time.strftime(args.path_for_outputs, time.localtime())
    if base_folder[-1] != '/':
        base_folder += '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ['model_weights']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)

    PATH = args.previous_checkpoint
    logfile = base_folder + 'log_10.3.1_mon.txt'
    if not PATH:
        with open(logfile, 'w') as f:
            f.write('Epoch\tTrain\tValidation\n')
    ##
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    model = ProteinMPNN(node_features=args.hidden_dim, 
                        edge_features=args.hidden_dim, 
                        hidden_dim=args.hidden_dim, 
                        num_encoder_layers=args.num_encoder_layers, 
                        num_decoder_layers=args.num_encoder_layers, 
                        k_neighbors=args.num_neighbors, 
                        dropout=args.dropout, 
                        augment_eps=args.backbone_noise)
    model.to(device)
    

    if PATH:
        checkpoint = torch.load(PATH)
        total_step = checkpoint['step'] #write total_step from the checkpoint
        epoch = checkpoint['epoch'] #write epoch from the checkpoint
        # total_step = 0
        # epoch = 0
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        total_step = 0
        epoch = 0

    optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)
    
    if PATH:
        optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    with ProcessPoolExecutor(max_workers=12) as executor:

        dataset_train_all = StructureDataset(args.jsonl_path_train, truncate=None, max_length=args.max_length) # n ge [5ge zi dian]
        dataset_valid_all = StructureDataset(args.jsonl_path_valid, truncate=None, max_length=args.max_length)

        bz_train = args.batch_size_train
        bz_valid = args.batch_size_valid

        start_index_train = 0
        start_index_valid = 0
        
        reload_c = 1
        # reload_c = 0 
        for e in range(args.num_epochs):
            t0 = time.time()
            e = epoch + e
            model.train()
            '!!!!!!!!!!'

            train_sum, train_weights = 0., 0.
            train_acc = 0.
            train_stru_sum = 0.0
            train_stru_ave = 0.0
            train_loss = 0.0
            train_loss_ave = 0.0

            if start_index_train > len(dataset_train_all):
                start_index_train = 0
                dataset_train = dataset_train_all[start_index_train:start_index_train + bz_train]
            else:
                dataset_train = dataset_train_all[start_index_train:start_index_train + bz_train]

            if start_index_valid > len(dataset_valid_all):
                start_index_valid = 0
                dataset_valid = dataset_valid_all[start_index_valid:start_index_valid + bz_valid]
            else:
                dataset_valid = dataset_valid_all[start_index_valid:start_index_valid + bz_valid]
                reload_c += 1

            if os.path.isfile(args.chain_id_train_jsonl):
                with open(args.chain_id_train_jsonl, 'r') as json_file:
                    json_list = list(json_file)
                for json_str in json_list:
                    chain_id_dict_all = json.loads(json_str)
            else:
                chain_id_dict_all = None

            if os.path.isfile(args.chain_id_val_jsonl):
                with open(args.chain_id_val_jsonl, 'r') as json_file_val:
                    json_list_val = list(json_file_val)
                for json_str_val in json_list_val:
                    chain_id_dict_all_val = json.loads(json_str_val)
            else:
                chain_id_dict_all_val = None

            for ix_, protein in enumerate(dataset_train):
                #
                start_batch = time.time()

                input_folder = "./seq/monomers"
                output_folder = "./structure/monomers"

                name = protein['name']
                score_list, global_score_list, all_probs_list, S_sample_list, all_log_probs_list = [], [], [], [], []
                (   X, S, mask, lengths,
                    chain_M, chain_encoding_all,
                    chain_list_list, visible_list_list,
                    masked_list_list, masked_chain_length_list_list,
                    chain_M_pos, omit_AA_mask, residue_idx,
                    dihedral_mask, tied_pos_list_of_lists_list,
                    pssm_coef, pssm_bias, pssm_log_odds_all,
                    bias_by_res_all, tied_beta) = tied_featurize([protein], device)

                optimizer.zero_grad()
                mask_for_loss = mask*chain_M
                    
                log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)

                max_indices = torch.argmax(log_probs, dim=-1) 

                seqs = ''
                for idx, seq_indices in enumerate(max_indices):
                    seqs = ''.join([alphabet[i] for i in seq_indices.tolist()])
                

                folder_path = output_folder + '/' + name + '_AF2_output'

                check_and_clear_folder(folder_path)
                highfold(seqs, name, input_folder, output_folder, protein['num_of_chains'])
        
                'HighFold'
                search_string_pdb = '_unrelaxed_rank_001_alphafold2_multimer_v3_model'
                pdb_file = find_file_with_name(folder_path, search_string_pdb)
                if pdb_file == 'no such file':
                    continue
                jsonl_pdb = pdb_to_jsonl(pdb_file)

                search_string_json = '_scores_rank_001_alphafold2_multimer_v3_model'
                scores_json_file = find_file_with_name(folder_path, search_string_json)

                X_pre = featurize_pdb(jsonl_pdb, device)

                plddt_list = extract_plddt_from_json(scores_json_file)

                pep_plddt = plddt_list[-len(seqs):]

                same_chain = torch.tensor(1)
                X_fape = X.squeeze(0)[:, :3, :]
                X_pre_fape = X_pre.squeeze(0)[:, :3, :]

                loss_fape_value = fape_loss(X_fape, X_pre_fape, mask_for_loss, same_chain, A=torch.tensor(X.shape[1]), device=device)

                loss_fape = loss_fape_value*loss_av_smoothed/20

                pep_plddt_ave = np.mean(pep_plddt)

                if pep_plddt_ave > 70:
                    alpha = 0.1
                else:

                    alpha = 0.05

                loss_all = combined_loss_fn(loss_av_smoothed, loss_fape, protein['num_of_chains'], alpha)

                ###############
                loss_all.backward()

                if args.gradient_norm > 0.0:
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_norm)

                optimizer.step()

                loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)

                train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                train_weights += torch.sum(mask_for_loss).cpu().data.numpy()

                train_stru_sum += loss_fape.detach().cpu().numpy()
                train_loss += loss_all.detach().cpu().numpy()

                total_step += 1

            model.eval()

            with torch.no_grad():
                validation_sum, validation_weights = 0., 0.
                validation_acc = 0.

                #####
                val_stru_sum = 0.#
                val_stru_ave = 0.#
                for ix, protein in enumerate(dataset_valid):
                    score_list, global_score_list, all_probs_list, S_sample_list, all_log_probs_list = [], [], [], [], []

                    input_folder = "./val_seq/monomers"
                    output_folder = "./val_structure/monomers"

                    name = protein['name']

                    score_list, global_score_list, all_probs_list, S_sample_list, all_log_probs_list = [], [], [], [], []
                    (   X, S, mask, lengths,
                        chain_M, chain_encoding_all,
                        chain_list_list, visible_list_list,
                        masked_list_list, masked_chain_length_list_list,
                        chain_M_pos, omit_AA_mask, residue_idx,
                        dihedral_mask, tied_pos_list_of_lists_list,
                        pssm_coef, pssm_bias, pssm_log_odds_all,
                        bias_by_res_all, tied_beta) = tied_featurize([protein], device)
                    len_pep = torch.sum(chain_encoding_all == 1).item()
                    '!!!!!!!!!!'
                
                    randn_2 = torch.randn(chain_M.shape, device=X.device)
                    omit_AAs_list = args.omit_AAs  
                    omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)
                    bias_AAs_np = np.zeros(len(alphabet))

                    sample_dict = model.sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=0.1, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, bias_by_res=bias_by_res_all)
                    S_sample = sample_dict["S"]

                    log_probs = model(X, S_sample, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_2, use_input_decoding_order=True, decoding_order=sample_dict["decoding_order"])
                    mask_for_loss = mask*chain_M

                    loss, loss_av, true_false = loss_nll(S, log_probs, mask_for_loss)

                    max_indices = torch.argmax(log_probs, dim=-1)

                    seqs = ''
                    for idx, seq_indices in enumerate(max_indices):
                        seqs = ''.join([alphabet[i] for i in seq_indices.tolist()])
                            
                    folder_path = output_folder + '/' + name + '_AF2_output'

                    check_and_clear_folder(folder_path)
                    highfold(seqs, name, input_folder, output_folder, protein['num_of_chains'])                    
                    
                    search_string_pdb = '_unrelaxed_rank_001_alphafold2_multimer_v3_model'
                    pdb_file = find_file_with_name(folder_path, search_string_pdb)
                    if pdb_file == 'no such file':
                        continue
                    jsonl_pdb = pdb_to_jsonl(pdb_file)

                    X_pre = featurize_pdb(jsonl_pdb, device)

                    same_chain = torch.tensor(1)
                    X_fape = X.squeeze(0)[:, :3, :]
                    X_pre_fape = X_pre.squeeze(0)[:, :3, :]
                    loss_fape_value = fape_loss(X_fape, X_pre_fape, mask_for_loss, same_chain, A=torch.tensor(X.shape[1]), device=device)

                    val_stru_sum += loss_fape_value.cpu().numpy()
                    
                    validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                    validation_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                    validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()
            
            train_loss = train_sum / train_weights
            train_accuracy = train_acc / train_weights
            train_perplexity = np.exp(train_loss)
            validation_loss = validation_sum / validation_weights
            validation_accuracy = validation_acc / validation_weights
            validation_perplexity = np.exp(validation_loss)
            
            train_perplexity_ = np.format_float_positional(np.float32(train_perplexity), unique=False, precision=3)     
            validation_perplexity_ = np.format_float_positional(np.float32(validation_perplexity), unique=False, precision=3)
            train_accuracy_ = np.format_float_positional(np.float32(train_accuracy), unique=False, precision=3)
            validation_accuracy_ = np.format_float_positional(np.float32(validation_accuracy), unique=False, precision=3)
            
            ######
            train_stru_ave = np.format_float_positional(np.float32(train_stru_sum / (ix_+1)), unique=False, precision=3)#
            val_stru_ave = np.format_float_positional(np.float32(val_stru_sum / (ix+1)), unique=False, precision=3)#


            t1 = time.time()
            dt = np.format_float_positional(np.float32(t1-t0), unique=False, precision=1) 
            with open(logfile, 'a') as f:
                f.write(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, train_stru_FAPE_ave: {train_stru_ave}, valid_acc: {validation_accuracy_}, val_stru_FAPE_ave: {val_stru_ave}, train_loss: {train_loss}\n')
            print(f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, train_stru_FAPE_ave: {train_stru_ave}, valid_acc: {validation_accuracy_}, val_stru_FAPE_ave: {val_stru_ave}, train_loss: {train_loss}')

            ##################
            start_index_train = start_index_train + bz_train
            start_index_valid = start_index_valid + bz_valid
                    
            checkpoint_filename_last = base_folder+'model_weights/epoch_last.pt'.format(e+1, total_step)
            torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.optimizer.state_dict(),
                        }, checkpoint_filename_last)

            if (e+1) % args.save_model_every_n_epochs == 0:
                checkpoint_filename = base_folder+'model_weights/epoch{}_step{}.pt'.format(e+1, total_step)
                torch.save({
                        'epoch': e+1,
                        'step': total_step,
                        'num_edges' : args.num_neighbors,
                        'noise_level': args.backbone_noise, 
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.optimizer.state_dict(),
                        }, checkpoint_filename)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--path_for_training_data", type=str, default="my_path/pdb_2021aug02", help="path for loading training data") 

    argparser.add_argument("--chain_id_train_jsonl",type=str, default='./data/Cyclic peptide/train/assigned_pdbs.jsonl', help="Path to a dictionary specifying which chains need to be designed and which ones are fixed, if not specied all chains will be designed.")
    argparser.add_argument("--chain_id_val_jsonl",type=str, default='./data/Cyclic peptide/val/assigned_pdbs.jsonl', help="Path to a dictionary specifying which chains need to be designed and which ones are fixed, if not specied all chains will be designed.")
    
    argparser.add_argument("--jsonl_path_train", type=str, default="./data/Cyclic peptide/train/parsed_pdbs.jsonl",help="path for loading training data") #json
    argparser.add_argument("--jsonl_path_valid", type=str, default="./data/Cyclic peptide/val/parsed_pdbs.jsonl",help="path for loading valid data") #json
    
    argparser.add_argument("--path_for_outputs", type=str, default="./test_81_stru3/30_recycle3", help="path for logs and model weights=./exp_020")#
    #
    argparser.add_argument("--previous_checkpoint", type=str, default="./script/model_weight/HighMPNN.pt", help="path for previous model weights, e.g. file.pt")#

    argparser.add_argument("--num_epochs", type=int, default=10, help="number of epochs to train for 200")#
    argparser.add_argument("--batch_size_train", type=int, default=256, help="number of tokens for one batch") #
    argparser.add_argument("--batch_size_valid", type=int, default=32, help="number of tokens for one batch") #
    ##
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=5, help="save model weights every n epochs = 10")#
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=2, help="reload training data every n epochs")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000, help="number of training example to load for one epoch")
    argparser.add_argument("--max_protein_length", type=int, default=10000, help="maximum length of the protein complext")
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph")   
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.2, help="amount of noise added to backbone during training")   
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--debug", type=bool, default=False, help="minimal data loading for debugging")
    argparser.add_argument("--gradient_norm", type=float, default=-1.0, help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision", type=bool, default=False, help="train with mixed precision")
    ##
    argparser.add_argument("--max_length", type=int, default=200000, help="Max sequence length")
    ##
    argparser.add_argument("--omit_AAs", type=list, default='X', help="Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.")

    args = argparser.parse_args()    
    main(args)   
