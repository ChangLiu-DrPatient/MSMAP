import os, pickle, torch, argparse
import os.path as osp
import torch.nn as nn
import numpy as np
from model import *
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import *
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold
from rich.progress import track

cur_dir = osp.abspath(osp.dirname(__file__))
os.chdir(cur_dir)

SEED=56
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
CUTOFF_DICT = {800:16, 400:12, 200:8, 100:4}


def get_args():
    parser = argparse.ArgumentParser()
    # specifics for training
    parser.add_argument("--device", choices=[-1,0,1,2,3], default=2, type=int, help='device number (-1 for cpu)')
    parser.add_argument("--k", default=4, type=int, help='cross-validation folds')
    parser.add_argument("--bsz", default=512, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-5, type=float)
    parser.add_argument("--epochs", default=128, type=int)
    parser.add_argument("--split-seed", default=26, type=int)
    parser.add_argument("--obj", default='dist', choices=['dist', 'onlyCLS', 'clusCLS'], type=str)

    # specifics for data
    parser.add_argument('--num-spec', type=int, default=1000, help='Number of spectra per cluster')
    parser.add_argument('--randomize', default=True, action='store_true')
    # specifics for model
    parser.add_argument("--classifier-channels", default=1, type=int)
    parser.add_argument("--snp_dim", default=128, type=float)

    args=parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    rand = '_rand' if args.randomize else ''
    device = torch.device(f'cuda:{args.device}' if args.device >= 0 else 'cpu')
    os.makedirs('models/synth_spec/num_{}{}/seed_{}'.format(args.num_spec, rand, args.split_seed), exist_ok=True)
    os.makedirs('synth_spec_results/num_{}{}/seed_{}'.format(args.num_spec, rand, args.split_seed), exist_ok=True)
        
    with open(f'../synth_spec/synth_specs_{args.num_spec}{rand}.pkl', 'rb') as f:
        synth_dict = pickle.load(f)
    
    # tasks: (1) predict loc; (2) regress on pairwise distance within specs in a batch
    full_dataset = []
    for i, ((loc_1, loc_2), spec_dict) in enumerate(synth_dict.items()):

        # sine_waves = waves_dict['sine_waves']
        # triangle_waves = waves_dict['triangle_waves']

        # sine_latents = waves_dict['sine_latents']
        # triangle_latents = waves_dict['triangle_latents']

        # sine_ks = waves_dict['sine_ks']
        # triangle_ks = waves_dict['triangle_ks']

        # group_ids = waves_dict['group_ids']
        list_of_values = list(spec_dict.values())
        for j in range(len(list_of_values)):
            v = list_of_values[j]
            rel_intensities = v['rel_intensities']
            spectra = v['spectra']
            for rel_intensity, spec in zip(rel_intensities, spectra):
                full_dataset.append((spec/np.sum(spec), i, rel_intensity, j + i * len(list_of_values)))
        # print(full_dataset[-1][1:])
    # print(len(full_dataset), len(full_dataset[0][0]))
    # assert 1==0

    num_labels = i + 1
    full_index = [i for i in range(len(full_dataset))]
    Y = [y for _, y, _, _ in full_dataset]
    full_group_ids = relabel(np.array([gid for _, _, _, gid in full_dataset]))
    print(f'number of group ids: {len(set(full_group_ids))}')


    # gkf = StratifiedGroupKFold(n_splits=args.k, random_state=args.split_seed, shuffle=True)
    gkf = StratifiedKFold(n_splits=args.k, random_state=args.split_seed, shuffle=True)

    # for fold, (train_index, val_index) in enumerate(gkf.split(full_index, Y, full_group_ids)):
    for fold, (train_index, val_index) in enumerate(gkf.split(full_index, Y)):
        # if fold > 0: break

        with open(f'synth_spec_results/num_{args.num_spec}{rand}/seed_{args.split_seed}/val_index_fold{fold}.pkl', 'wb') as f:
            pickle.dump(val_index, f)
        with open(f'synth_spec_results/num_{args.num_spec}{rand}/seed_{args.split_seed}/train_index_fold{fold}.pkl', 'wb') as f:
            pickle.dump(train_index, f)
        # continue

        print('Fold {}: Train {} Val {}'.format(fold, len(train_index), len(val_index)))

        X_train = torch.from_numpy(np.stack([full_dataset[i][0] for i in train_index], axis=0)[:, np.newaxis, :])
        Y_train = torch.from_numpy(np.array([full_dataset[i][1] for i in train_index]))
        latent_train = torch.from_numpy(np.array([full_dataset[i][2] for i in train_index]))
        group_ids_train = torch.from_numpy(np.array([full_dataset[i][3] for i in train_index]))
        
        X_val = torch.from_numpy(np.stack([full_dataset[i][0] for i in val_index], axis=0)[:, np.newaxis, :])
        Y_val = torch.from_numpy(np.array([full_dataset[i][1] for i in val_index]))
        latent_val = torch.from_numpy(np.array([full_dataset[i][2] for i in val_index]))
        group_ids_val = torch.from_numpy(np.array([full_dataset[i][3] for i in val_index]))

        train_dataset = TensorDataset(X_train, Y_train, latent_train, group_ids_train)
        val_dataset = TensorDataset(X_val, Y_val, latent_val, group_ids_val)

        train_loader = DataLoader(dataset=train_dataset, batch_size=args.bsz, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.bsz, shuffle=True)

        # continue
        if args.obj == 'clusCLS':
            model = unet_all_in_one(num_species=num_labels, num_classes=len(set(full_group_ids)), vec_len=len(full_dataset[0][0])).to(device)
        else:
            model = unet_snp(num_species=num_labels, snp_dim=args.snp_dim, vec_len=len(full_dataset[0][0])).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-5)
        
        criterion_r = nn.MSELoss(reduction='mean') # reconstruction loss
        criterion_species = nn.CrossEntropyLoss(ignore_index=-1) # classification loss
        criterion_latent = nn.CrossEntropyLoss(ignore_index=-1) if args.obj == 'clusCLS' else nn.MSELoss() #SNPLoss(threshold = CUTOFF_DICT[args.num_spec]) # latent regression loss
        # criterion_latent = SNPLoss(threshold=2 * args.sigma_latent * np.sqrt(2))

        # start training
        best_acc = 0
        best_acc_latent = 0
        best_loss_latent = 1e8
        # best_loss_latent_pool = 1e8
        best_epoch = 0

        for epoch in track(range(args.epochs), description=f'Fold {fold}'):
            model.train()
            losses_r = []
            losses_species = []
            losses_latent = []
            # losses_latent_pool = []

            total_correct_species = 0
            total_correct_group_id = 0
            total_samples = 0

            for _, (data, species, latent, group_ids) in enumerate(train_loader):
                data, species, latent, group_ids = data.to(device, dtype=torch.float), species.to(device), latent.to(device, dtype=torch.float), group_ids.to(device)

                if args.obj == 'clusCLS':
                    r, h_latent, h_species, h0, b = model(data)
                else:
                    r, h_species, h_latent, b = model(data)

                
                # get pretext accuracy
                _, predicted_species = torch.max(h_species, 1)
                correct_species = (predicted_species == species).sum().item()
                total_correct_species += correct_species
                total_samples += species.shape[0]

                # Reconstruction loss
                loss_r = criterion_r(r, data)
                losses_r.append(loss_r.item())

                # species classification loss
                loss_species = criterion_species(h_species, species)
                losses_species.append(loss_species.item())

                # latentuency dist regression loss
                if args.obj == 'clusCLS':
                    _, predicted_group_id = torch.max(h_latent, 1)
                    correct_group_id = (predicted_group_id == group_ids).sum().item()
                    total_correct_group_id += correct_group_id
                    loss_latent = criterion_latent(h_latent, group_ids)
                else:
                    loss_latent = 0
                    # separate the indexes of sine and triangle waves
                    sp_count = 0
                    for label in range(num_labels):
                        sp_idx = torch.where(species == label)
                        if len(sp_idx) == 0:
                            continue
                        sp_count += 1
                        gt_latent_dist = torch.cdist(latent[sp_idx], latent[sp_idx])
                        h_latent_dist = torch.cdist(h_latent[sp_idx], h_latent[sp_idx])
                        loss_latent += criterion_latent(h_latent_dist, gt_latent_dist)
                    loss_latent /= sp_count
                
                losses_latent.append(loss_latent.item())

                # Total loss
                if args.obj == 'onlyCLS':
                    loss = loss_r + loss_species
                else:
                    loss = loss_r + loss_species + loss_latent

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss_r = sum(losses_r) / len(losses_r)
            avg_loss_species = sum(losses_species) / len(losses_species)
            avg_loss_latent = sum(losses_latent) / len(losses_latent)

            accuracy_species = total_correct_species / total_samples
            accuracy_latent = total_correct_group_id / total_samples

            avg_loss_r_val, avg_loss_species_val, avg_loss_latent_val, accuracy_species_val, accuracy_latent_val =\
                 eval_synth_spec(model, val_loader, device, criterion_r, criterion_species, criterion_latent, obj=args.obj, num_labels=num_labels)
            print(f'Epoch {epoch} Train Loss R: {avg_loss_r:.3f} species: {avg_loss_species:.3f} latent: {avg_loss_latent:.3f} Acc: {accuracy_species:.3f} LAcc: {accuracy_latent:.3f} \
                  Val Loss R: {avg_loss_r_val:.3f} species: {avg_loss_species_val:.3f} latent: {avg_loss_latent_val:.3f} Acc: {accuracy_species_val:.3f} LAcc: {accuracy_latent_val:.3f}')
            
            if args.obj == 'onlyCLS':
                if accuracy_species_val > best_acc:
                    best_acc = accuracy_species_val
                    best_loss_latent = avg_loss_latent_val
                    best_epoch = epoch
                    torch.save(model.state_dict(), f'models/synth_spec/num_{args.num_spec}{rand}/seed_{args.split_seed}/{args.obj}_fold{fold}.pt')
                if 1 - accuracy_species_val < 1e-8:
                    print('already perfect accuracy')
                    break
            elif args.obj == 'dist':
                if avg_loss_latent_val < best_loss_latent:
                    best_acc = accuracy_species_val
                    best_loss_latent = avg_loss_latent_val
                    best_epoch = epoch
                    torch.save(model.state_dict(), f'models/synth_spec/num_{args.num_spec}{rand}/seed_{args.split_seed}/{args.obj}_fold{fold}.pt')
            elif args.obj == 'clusCLS':
                if  accuracy_latent_val > best_acc_latent:
                    best_acc_latent = accuracy_latent_val
                    best_acc = accuracy_species_val
                    best_loss_latent = avg_loss_latent_val
                    best_epoch = epoch
                    torch.save(model.state_dict(), f'models/synth_spec/num_{args.num_spec}{rand}/seed_{args.split_seed}/{args.obj}_fold{fold}.pt')
            scheduler.step()
            # assert 1==0
        print(f'Fold {fold} Best Epoch: {best_epoch} Best Loss latent: {best_loss_latent:.3f} Best Acc: {best_acc:.3f}')
        # torch.save(model.state_dict(), f'models/synth_spec/num_{args.num_spec}/seed_{args.split_seed}/{args.obj}_fold{fold}.pt')
            