import os, pickle, torch, argparse
import os.path as osp
import torch.nn as nn
import numpy as np
from model import *
from torch.utils.data import TensorDataset, DataLoader
from utils import *
from rich.progress import track
from scipy.spatial.distance import pdist

cur_dir = osp.abspath(osp.dirname(__file__))
os.chdir(cur_dir)

SEED=56
CUTOFF_DICT = {800:16, 400:12, 200:8, 100:4}
NUMCLUS_DICT = {800:4, 400:8, 200:16, 100:32}
METRIC_LIST = ['nmi', 'ari', 'f1_score']

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=[-1,0,1,2,3], default=0, type=int, help='device number (-1 for cpu)')
    parser.add_argument("--k", default=4, type=int, help='cross-validation folds')
    parser.add_argument("--bsz", default=512, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-5, type=float)
    parser.add_argument("--epochs", default=128, type=int)
    parser.add_argument("--split-seed", default=26, type=int)
    parser.add_argument("--obj", default='dist', choices=['dist', 'onlyCLS', 'clusCLS', 'rawclus'], type=str)

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
    device = torch.device(f'cuda:{args.device}' if args.device >= 0 else 'cpu')
    rand = '_rand' if args.randomize else ''
    os.makedirs('models/synth_spec/num_{}{}/seed_{}'.format(args.num_spec, rand, args.split_seed), exist_ok=True)
    os.makedirs('synth_spec_results/num_{}{}/seed_{}'.format(args.num_spec, rand, args.split_seed), exist_ok=True)
        
    with open(f'../synth_spec/synth_specs_{args.num_spec}{rand}.pkl', 'rb') as f:
        synth_dict = pickle.load(f)
    
    # tasks: (1) predict loc; (2) regress on pairwise distance within specs in a batch
    full_dataset = []
    for i, ((loc_1, loc_2), spec_dict) in enumerate(synth_dict.items()):

        list_of_values = list(spec_dict.values())
        for j in range(len(list_of_values)):
            v = list_of_values[j]
            rel_intensities = v['rel_intensities']
            spectra = v['spectra']
            for rel_intensity, spec in zip(rel_intensities, spectra):
                full_dataset.append((spec/np.sum(spec), i, rel_intensity, j + i * len(list_of_values)))



    num_labels = i + 1
    full_index = [i for i in range(len(full_dataset))]
    Y = [y for _, y, _, _ in full_dataset]
    full_group_ids = relabel(np.array([gid for _, _, _, gid in full_dataset]))
    print(f'number of group ids: {len(set(full_group_ids))}')

    type_group_dict = {name: {'idxs': [], 'group_ids': [], 'latent': [], 'acc_per_species': [-1 for _ in range(args.k)]} for name in [i for i in range(num_labels)] + ['Overall']}
    for i, (_, y, latent, gid) in enumerate(full_dataset):
        type_group_dict['Overall']['idxs'].append(i)
        type_group_dict['Overall']['group_ids'].append(full_group_ids[i])
        type_group_dict['Overall']['latent'].append(latent)

        type_group_dict[y]['idxs'].append(i)
        type_group_dict[y]['group_ids'].append(full_group_ids[i])
        type_group_dict[y]['latent'].append(latent)
    
    moi_cutoffs_dict = {k:[0 for _ in range(args.k)] for k in METRIC_LIST}
    if args.obj in ['dist', 'onlyCLS']:
        gt_cutoff = CUTOFF_DICT[args.num_spec] #!
        # best_cutoffs = {}
        
        # step 1: get embeddings
        embedds_latent_folds = []
        for fold in range(args.k):
            acc_per_species_fold = {key: [] for key in type_group_dict.keys()}
            with open(f'synth_spec_results/num_{args.num_spec}{rand}/seed_{args.split_seed}/train_index_fold{fold}.pkl', 'rb') as f:
                train_index = pickle.load(f)
            with open(f'synth_spec_results/num_{args.num_spec}{rand}/seed_{args.split_seed}/val_index_fold{fold}.pkl', 'rb') as f:
                val_index = pickle.load(f)

            X_train = torch.from_numpy(np.stack([full_dataset[i][0] for i in train_index], axis=0)[:, np.newaxis, :])
            X_val = torch.from_numpy(np.stack([full_dataset[i][0] for i in val_index], axis=0)[:, np.newaxis, :])
            y_val = torch.from_numpy(np.array([full_dataset[i][1] for i in val_index]))
            train_dataset = TensorDataset(X_train, torch.tensor(train_index))
            val_dataset = TensorDataset(X_val, y_val, torch.tensor(val_index))
            train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
            val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

            model = unet_snp(num_species=num_labels, snp_dim=args.snp_dim, vec_len=len(full_dataset[0][0]))
            state_dict_path = f'models/synth_spec/num_{args.num_spec}{rand}/seed_{args.split_seed}/{args.obj}_fold{fold}.pt'
            model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))

            device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
            embedds_latent = {}

            model = model.to(device)
            model.eval()

            print('Getting best cutoffs for fold {}...'.format(fold))
            embeds_h_latent_train = []
            gt_clusters = []
            with torch.no_grad():
                for _, (data, idxs) in enumerate(train_loader):
                    data = data.to(device, dtype=torch.float)
                    r, h_species, h_latent, b = model(data)
                    h_latent = h_latent.cpu().numpy().squeeze()
                    embeds_h_latent_train.append(h_latent)
                    gt_clusters.append(full_group_ids[idxs.item()])
            best_metrics = {k:0 for k in METRIC_LIST}
            print(f'gt cutoff: {gt_cutoff}')
            embeds_idx_sample = np.random.choice(len(embeds_h_latent_train), 100, replace=False)
            latent_train_pdists = pdist(np.stack([embeds_h_latent_train[i] for i in embeds_idx_sample], axis=0))
            # randomly sample 1000 pdists:
            print(min(latent_train_pdists), max(latent_train_pdists))
            # assert 1==0
            # print(min(latent_train_pdists), max(latent_train_pdists))
            # assert 1==0
            for cutoff in np.arange(int(np.round(min(latent_train_pdists))), max(np.round(latent_train_pdists)), gt_cutoff):
                pred_clusters, _ = get_predicted_components(embeds_h_latent_train, cutoff=cutoff)
                recall, precision, f1_score, ari, nmi, num_nonsingle_clusters = get_metrics(np.array(gt_clusters), pred_clusters)
                for moi in METRIC_LIST:
                    metric_value = eval(moi)
                    if metric_value > best_metrics[moi]:
                        best_metrics[moi] = metric_value
                        moi_cutoffs_dict[moi][fold] = cutoff

            print(f'Best metrics: ', best_metrics)

            print('Getting embeddings for fold {}...'.format(fold))
            with torch.no_grad():
                for _, (data, labels, idxs) in enumerate(val_loader):
                    data = data.to(device, dtype=torch.float)
                    r, h_species, h_latent, b = model(data)
                    h_latent = h_latent.cpu().numpy().squeeze()
                    embedds_latent[idxs.item()] = h_latent
                    predicted_species = torch.max(h_species, 1).indices.item()
                    gt_label = labels.item()
                    acc_per_species_fold['Overall'].append(1 if predicted_species == gt_label else 0)
                    acc_per_species_fold[gt_label].append(1 if predicted_species == gt_label else 0)

            for key, value in acc_per_species_fold.items():
                print(np.sum(value), len(value))
                if len(value) > 0:
                    type_group_dict[key]['acc_per_species'][fold] = np.mean(value)
            # assert 1==0

            # print(len(embedds_latent))
            # with open(f'synth_spec_results/num_{args.num_spec}/seed_{args.split_seed}/embedds_latent_{args.obj}_fold{fold}.pkl', 'wb') as f:
            #     pickle.dump(embedds_latent, f)
            embedds_latent_folds.append(embedds_latent)
        print(f'Best cutoffs: ', moi_cutoffs_dict)
        # assert 1==0
        type_group_dict['Overall']['best_cutoffs'] = moi_cutoffs_dict
        # step 2: get connected components
        # embedds_latent_folds = [pickle.load(open(f'synth_spec_results/num_{args.num_spec}/seed_{args.split_seed}/embedds_latent_{args.obj}_fold{fold}.pkl', 'rb')) for fold in range(args.k)]
        print('getting connected components for all species...')

        for key, value in track(type_group_dict.items()):
            labels = value['group_ids']
            pred_fold_cncs = [{} for _ in range(args.k)]
            pred_fold_dists = [{} for _ in range(args.k)]
            gt_fold_latents = [[] for _ in range(args.k)]
            gt_fold_cncs = [[] for _ in range(args.k)]
            species_idx_fold = [[] for _ in range(args.k)]

            for i in range(len(value['idxs'])):
                idx = value['idxs'][i]
                for fold in range(args.k):
                    if idx in embedds_latent_folds[fold]:
                        species_idx_fold[fold].append(i)
                        gt_fold_cncs[fold].append(labels[i])
                        gt_fold_latents[fold].append(value['latent'][i])
                    
            #     if idx in embedds_latent_0:
            #         fold_0_idxs.append(i)
            #         type_group_dict[key]['cnc_fold0'].append(value['group_ids'][i])
            #         type_group_dict[key]['freq0'].append(value['latent'][i])
            #     elif idx in embedds_latent_1:
            #         fold_1_idxs.append(i)
            #         type_group_dict[key]['cnc_fold1'].append(value['group_ids'][i])
            #         type_group_dict[key]['freq1'].append(value['latent'][i])
            # print(len(fold_0_idxs), len(fold_1_idxs))
            # labels_nn_0 = [predictions_latent_0[value['idxs'][idx]] for idx in fold_0_idxs]
            # labels_nn_1 = [predictions_latent_1[value['idxs'][idx]] for idx in fold_1_idxs]

            for fold in range(args.k):
                embeds_species_fold = np.array([embedds_latent_folds[fold][value['idxs'][idx]] for idx in species_idx_fold[fold]])
                if len(embeds_species_fold) < 2: continue
                for metric, moi_cutoffs in moi_cutoffs_dict.items():
                    print(f'using {metric} cutoff: {moi_cutoffs[fold]}')
                    pred_labels, pred_dist_mat = get_predicted_components(embeds_species_fold, cutoff=moi_cutoffs[fold])
                    # pred_labels, pred_dist_mat = get_predicted_components(embeds_species_fold, cutoff=gt_cutoff)
                    # print(pred_dist_mat)
                    pred_fold_cncs[fold][metric] = pred_labels
                    pred_fold_dists[fold][metric] = pred_dist_mat
                # labels_nn, pred_dist_mat = get_predicted_components(embeds_species_fold, cutoff=best_cutoffs[fold])
                pred_labels= get_predicted_components_num(embeds_species_fold, num_clusters=len(set(gt_fold_cncs[fold])))
                pred_fold_cncs[fold]['num'] = pred_labels
                # print(pred_dist_mat)
                # print(pred_fold_cncs[fold])
                # assert 1==0
            type_group_dict[key]['pred_fold_cncs'] = pred_fold_cncs
            type_group_dict[key]['pred_fold_dists'] = pred_fold_dists
            type_group_dict[key]['gt_fold_latents'] = gt_fold_latents
            type_group_dict[key]['gt_fold_cncs'] = gt_fold_cncs

    elif args.obj == 'rawclus':
        specs_folds = []
        for fold in range(args.k):
            with open(f'synth_spec_results/num_{args.num_spec}/seed_{args.split_seed}/train_index_fold{fold}.pkl', 'rb') as f:
                train_index = pickle.load(f)
            with open(f'synth_spec_results/num_{args.num_spec}/seed_{args.split_seed}/val_index_fold{fold}.pkl', 'rb') as f:
                val_index = pickle.load(f)

            X_train = torch.from_numpy(np.stack([full_dataset[i][0] for i in train_index], axis=0)[:, np.newaxis, :])
            X_val = torch.from_numpy(np.stack([full_dataset[i][0] for i in val_index], axis=0)[:, np.newaxis, :])
            
            train_dataset = TensorDataset(X_train, torch.tensor(train_index))
            val_dataset = TensorDataset(X_val, torch.tensor(val_index))
            train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
            val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

            gt_clusters = []
            specs_fold_train = []
            for _, (data, idxs) in enumerate(train_loader):
                specs_fold_train.append(data.cpu().numpy().squeeze())
                gt_clusters.append(full_group_ids[idxs.item()])
            
            spec_idx_sample = np.random.choice(len(specs_fold_train), size=100, replace=False)
            spec_pdists = pdist(np.stack([specs_fold_train[i] for i in spec_idx_sample], axis=0))
            min_cutoff, max_cutoff = min(spec_pdists), max(spec_pdists)
            print(min_cutoff, max_cutoff)
            # get best cuoff for each metric
            best_metrics = {k:0 for k in METRIC_LIST}
            for cutoff in np.linspace(min_cutoff, max_cutoff, 20): #!
                pred_clusters, _ = get_predicted_components(specs_fold_train, cutoff=cutoff)
                recall, precision, f1_score, ari, nmi, num_nonsingle_clusters = get_metrics(np.array(gt_clusters), pred_clusters)
                for moi in METRIC_LIST:
                    metric_value = eval(moi)
                    if metric_value > best_metrics[moi]:
                        best_metrics[moi] = metric_value
                        moi_cutoffs_dict[moi][fold] = cutoff
            print(f'Best metrics: ', best_metrics)
            
            specs = {}
            for _, (data, idxs) in enumerate(val_loader):  # TODO: get species-wise accuracy
                    data = data.cpu().numpy().squeeze()
                    specs[idxs.item()] = data
            
            specs_folds.append(specs)

        print(f'Best cutoffs: ', moi_cutoffs_dict)
        type_group_dict['Overall']['best_cutoffs'] = moi_cutoffs_dict   
        print('getting connected components for all species...')

        for key, value in track(type_group_dict.items()):
            labels = value['group_ids']
            pred_fold_cncs = [{} for _ in range(args.k)]
            pred_fold_dists = [{} for _ in range(args.k)]
            gt_fold_latents = [[] for _ in range(args.k)]
            gt_fold_cncs = [[] for _ in range(args.k)]
            species_idx_fold = [[] for _ in range(args.k)]

            # determine which fold each specid belongs to
            for i in range(len(value['idxs'])):
                idx = value['idxs'][i]
                for fold in range(args.k):
                    if idx in specs_folds[fold]:
                        species_idx_fold[fold].append(i)
                        gt_fold_cncs[fold].append(labels[i])
                        gt_fold_latents[fold].append(value['latent'][i])
            
            for fold in range(args.k):
                embeds_species_fold = np.array([specs_folds[fold][value['idxs'][idx]] for idx in species_idx_fold[fold]])
                if len(embeds_species_fold) < 2: continue
                for metric, moi_cutoffs in moi_cutoffs_dict.items():
                    print(f'using {metric} cutoff: {moi_cutoffs[fold]}')
                    pred_labels, pred_dist_mat = get_predicted_components(embeds_species_fold, cutoff=moi_cutoffs[fold])
                    # pred_labels, pred_dist_mat = get_predicted_components(embeds_species_fold, cutoff=gt_cutoff)
                    # print(pred_dist_mat)
                    pred_fold_cncs[fold][metric] = pred_labels
                    pred_fold_dists[fold][metric] = pred_dist_mat
                # labels_nn, pred_dist_mat = get_predicted_components(embeds_species_fold, cutoff=best_cutoffs[fold])
                pred_labels= get_predicted_components_num(embeds_species_fold, num_clusters=len(set(gt_fold_cncs[fold])))
                pred_fold_cncs[fold]['num'] = pred_labels
                # print(pred_dist_mat)
                # print(pred_fold_cncs[fold])
                # assert 1==0
            type_group_dict[key]['pred_fold_cncs'] = pred_fold_cncs
            type_group_dict[key]['pred_fold_dists'] = pred_fold_dists
            type_group_dict[key]['gt_fold_latents'] = gt_fold_latents
            type_group_dict[key]['gt_fold_cncs'] = gt_fold_cncs
    
    else:
        predictions_latent_folds = []
        for fold in range(args.k):
            acc_per_species_fold = {key: [] for key in type_group_dict.keys()}
            with open(f'synth_spec_results/num_{args.num_spec}{rand}/seed_{args.split_seed}/val_index_fold{fold}.pkl', 'rb') as f:
                val_index = pickle.load(f)

            X_val = torch.from_numpy(np.stack([full_dataset[i][0] for i in val_index], axis=0)[:, np.newaxis, :])
            y_val = torch.from_numpy(np.array([full_dataset[i][1] for i in val_index]))

            val_dataset = TensorDataset(X_val, y_val, torch.tensor(val_index))
            val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

            model = unet_all_in_one(num_species=num_labels, num_classes=len(set(full_group_ids)), vec_len=len(full_dataset[0][0]))
            state_dict_path = f'models/synth_spec/num_{args.num_spec}{rand}/seed_{args.split_seed}/{args.obj}_fold{fold}.pt'
            model.load_state_dict(torch.load(state_dict_path))
            device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
            
            model = model.to(device)
            model.eval()
            predictions_latent = {}
            print('Getting predicted clusters for fold {}...'.format(fold))
            with torch.no_grad():
                for _, (data, labels, idxs) in enumerate(val_loader):
                    data = data.to(device, dtype=torch.float)
                    r, h_latent, h_species, h0, b = model(data)
                    _, predicted_group_id = torch.max(h_latent, 1)
                    predictions_latent[idxs.item()] = predicted_group_id.item()
                    predicted_species = torch.max(h_species, 1).indices.item()
                    gt_label = labels.item()
                    acc_per_species_fold['Overall'].append(1 if predicted_species == gt_label else 0)
                    acc_per_species_fold[gt_label].append(1 if predicted_species == gt_label else 0)

            for key, value in acc_per_species_fold.items():
                if len(value) > 0:
                    type_group_dict[key]['acc_per_species'][fold] = np.mean(value)
            # print(list(predictions_latent.items())[:5])
            # with open(f'synth_spec_results/num_{args.num_spec}/seed_{args.split_seed}/predictions_latent_{args.obj}_fold{fold}.pkl', 'wb') as f:
            #     pickle.dump(predictions_latent, f)
            predictions_latent_folds.append(predictions_latent)
        

        # assert 1==0
            
        # predictions_latent_folds = [pickle.load(open(f'synth_spec_results/num_{args.num_spec}/seed_{args.split_seed}/predictions_latent_{args.obj}_fold{fold}.pkl', 'rb')) for fold in range(args.k)]
        
        for key, value in track(type_group_dict.items()):
            labels = value['group_ids']
            pred_fold_cncs = [[] for _ in range(args.k)]
            gt_fold_latents = [[] for _ in range(args.k)]
            gt_fold_cncs = [[] for _ in range(args.k)]
            species_idx_fold = [[] for _ in range(args.k)]

            for i in range(len(value['idxs'])):
                idx = value['idxs'][i]
                for fold in range(args.k):
                    if idx in predictions_latent_folds[fold]:
                        species_idx_fold[fold].append(i)
                        gt_fold_cncs[fold].append(labels[i])
                        gt_fold_latents[fold].append(value['latent'][i])
            
            for fold in range(args.k):
                pred_fold_cncs[fold] = [predictions_latent_folds[fold][value['idxs'][idx]] for idx in species_idx_fold[fold]]
            
            type_group_dict[key]['pred_fold_cncs'] = pred_fold_cncs
            type_group_dict[key]['gt_fold_latents'] = gt_fold_latents
            type_group_dict[key]['gt_fold_cncs'] = gt_fold_cncs

    # step 3: get metrics
    if args.obj == 'clusCLS':
        for key, value in type_group_dict.items():
            print(key)
            pred_fold_cncs = value['pred_fold_cncs']
            gt_fold_cncs = value['gt_fold_cncs']
            accs = value['acc_per_species']

            min_lens = [min(len(pred_fold_cncs[i]), len(gt_fold_cncs[i])) for i in range(args.k)]
            for fold in range(args.k):
                if min_lens[fold] < 2:
                    print(f'skipping fold{fold} due to empty samples')
                    continue
                acc = accs[fold]
                recall, precision, f1_score, ari, nmi, num_nonsingle_clusters = get_metrics(np.array(gt_fold_cncs[fold]), np.array(pred_fold_cncs[fold]))
                metrics_dict = {'precision': precision, 'recall': recall, 'f1 score': f1_score, 'ARI': ari, 'NMI': nmi, 'acc': acc}
                type_group_dict[key][f'metrics_{fold}'] = metrics_dict
                print(f'Fold {fold}: Num spec {len(gt_fold_cncs[fold])}; gt Connected components: {len(set(gt_fold_cncs[fold]))}; Pred components: {len(set(pred_fold_cncs[fold]))}, Nonsingle components: {num_nonsingle_clusters}, NMI: {nmi:.2f}; ARI: {ari:.2f}; Precision: {precision:.2f}; Recall: {recall:.2f}; F1: {f1_score:.2f}; acc: {acc:.2f}')

        print('\n')
            
    else:
        for key, value in type_group_dict.items():
            print(key)
            pred_fold_cncs = value['pred_fold_cncs']
            gt_fold_cncs = value['gt_fold_cncs']
            accs = value['acc_per_species']

            min_lens = [min(len(pred_fold_cncs[i]['nmi']), len(gt_fold_cncs[i])) for i in range(args.k)]
            for fold in range(args.k):
                if min_lens[fold] < 2:
                    print(f'skipping fold{fold} due to empty samples')
                    continue
                acc = accs[fold]
                metrics_dict = {}
                for moi in pred_fold_cncs[fold].keys():
                    pred_labels = pred_fold_cncs[fold][moi]
                    recall, precision, f1_score, ari, nmi, num_nonsingle_clusters = get_metrics(np.array(gt_fold_cncs[fold]), np.array(pred_labels))
                    metrics_dict[moi] = {'precision': precision, 'recall': recall, 'f1 score': f1_score, 'ARI': ari, 'NMI': nmi, 'acc': acc}
                    print(f'Fold {fold} {moi}: Num spec {len(gt_fold_cncs[fold])}; gt Connected components: {len(set(gt_fold_cncs[fold]))}; Pred components: {len(set(pred_fold_cncs[fold][moi]))}, Nonsingle components: {num_nonsingle_clusters}, \
                          NMI: {nmi:.2f}; ARI: {ari:.2f}; Precision: {precision:.2f}; Recall: {recall:.2f}; F1: {f1_score:.2f}; acc: {acc:.2f}')

                print('\n')

            type_group_dict[key][f'metrics_{fold}'] = metrics_dict
            print('\n')
    pickle.dump(type_group_dict, open(f'synth_spec_results/num_{args.num_spec}{rand}/seed_{args.split_seed}/results_{args.obj}.pkl','wb'))
    print('Done!')