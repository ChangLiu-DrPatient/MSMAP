import torch, re
import numpy as np 
import pandas as pd
import torch.nn as nn
import os.path as osp
from collections import Counter
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import AgglomerativeClustering
from rich.progress import track
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from itertools import product


RANDOM_STATE=56
np.random.seed(56)

# relabel labels (fill in gaps)
def relabel(labels):
    unique_labels = np.unique(labels)
    new_labels = np.zeros(labels.shape)
    for i, label in enumerate(unique_labels):
        new_labels[labels == label] = i
    return new_labels.astype(int)


# Model evaluation functions
# proprietary data eval
def eval_all_in_one(model, data_loader, device, pad_len, criterion_r, criterion_h, criterion_species):
    model.to(device)
    model.eval()
    losses_r = []
    losses_h = []
    losses_species = []

    total_correct = 0
    total_correct_species = 0
    total_samples = 0
    with torch.no_grad():
        for _, (data, species, labels) in enumerate(data_loader):

            cal_loss_mask = torch.ones_like(data)# do not calculate losss for padded portions
            cal_loss_mask[:, :, -pad_len:] = 0
            non_pad_eles = data.shape[-1] - pad_len

            cal_loss_mask = cal_loss_mask.to(device)
            data = data.to(device, dtype=torch.float)
            labels = labels.to(device)
            species = species.to(device)

            r, h, h_species, h0, b = model(data)

            # get predictions and update accuracy stats
            _, predicted = torch.max(h, 1)
            correct = (predicted == labels).sum().item()
            total_correct += correct

            _, predicted_species = torch.max(h_species, 1)
            correct_species = (predicted_species == species).sum().item()
            total_correct_species += correct_species

            total_samples += labels.shape[0]
            
            # get losses for backprop
            loss_r = criterion_r(r, data)
            loss_r = loss_r * cal_loss_mask
            loss_r = loss_r.squeeze().sum() / non_pad_eles

            loss_h = criterion_h(h, labels)
            loss_species = criterion_species(h_species, species)

            losses_r.append(loss_r.detach().cpu().item())
            losses_h.append(loss_h.detach().cpu().item())
            losses_species.append(loss_species.detach().cpu().item())

        avg_loss_r = sum(losses_r) / len(losses_r)
        avg_loss_h = sum(losses_h) / len(losses_h)
        avg_loss_species = sum(losses_species) / len(losses_species)

        accuracy = total_correct / total_samples
        accuracy_species = total_correct_species / total_samples
        
    return avg_loss_r, avg_loss_h, avg_loss_species, accuracy, accuracy_species


def eval_snp(model, specids_val, data_loader, device, pad_len, snp_matrix_species, criterion_r, criterion_snp, criterion_species):
    model.to(device)
    model.eval()
    losses_r = []
    losses_snp = []
    losses_species = []

    total_correct_species = 0
    total_samples = 0
    with torch.no_grad():
        for _, (data, species, idxs) in enumerate(data_loader):

            # get the sub-snp matrices for the samples in this batch
            idxs = idxs.numpy()
            specIDs = specids_val[idxs]
            
            specID_idx_dict = {specIDs[i]: i for i in range(len(specIDs))}
            sub_snp_matrix_species = get_sub_snp_matrix_from_specIDs(snp_matrix_species, specIDs, device)
            for key, value in sub_snp_matrix_species.items():
                value['idxs'] = [specID_idx_dict[x] for x in value['specIDs']]
            
            cal_loss_mask = torch.ones_like(data)# do not calculate losss for padded portions
            cal_loss_mask[:, :, -pad_len:] = 0
            non_pad_eles = data.shape[-1] - pad_len

            cal_loss_mask = cal_loss_mask.to(device)
            data = data.to(device, dtype=torch.float)
            species = species.to(device)

            r, h_species, h0, b = model(data)


            # get predictions and update accuracy stats
            _, predicted_species = torch.max(h_species, 1)
            correct_species = (predicted_species == species).sum().item()
            total_correct_species += correct_species

            total_samples += species.shape[0]
            
            # get losses for backprop
            loss_r = criterion_r(r, data)
            loss_r = loss_r * cal_loss_mask
            loss_r = loss_r.squeeze().sum() / non_pad_eles

            loss_snp = 0
            for key, value in sub_snp_matrix_species.items():
                value['h0'] = h0[value['idxs']]
                value['h0_cdist'] = torch.cdist(value['h0'], value['h0'])
                loss_snp += criterion_snp(value['h0_cdist'], value['SNP'])
            
            loss_snp /= len(sub_snp_matrix_species)  # average over species


            loss_species = criterion_species(h_species, species)


            losses_snp.append(loss_snp.detach().cpu().item())
            losses_r.append(loss_r.detach().cpu().item())
            losses_species.append(loss_species.detach().cpu().item())


        avg_loss_r = sum(losses_r) / len(losses_r)
        avg_loss_species = sum(losses_species) / len(losses_species)
        avg_loss_snp = sum(losses_snp) / len(losses_snp)

        accuracy_species = total_correct_species / total_samples
        
    return avg_loss_r, avg_loss_species, avg_loss_snp, accuracy_species


# synthetic data eval
def eval_synth(model, data_loader, device, criterion_r, criterion_wtype, criterion_freq, obj):
    model.to(device)
    model.eval()
    losses_r = []
    losses_wtype = []
    losses_freq_dist = []


    total_correct_wtype = 0
    total_samples = 0

    with torch.no_grad():
        for _, (data, wtype, freq) in enumerate(data_loader):
            data, wtype, freq = data.to(device, dtype=torch.float), wtype.to(device), freq.to(device, dtype=torch.float)

            r, h_wtype, h0, b = model(data)
            # get accuracy
            _, predicted_wtype = torch.max(h_wtype, 1)
            correct_wtype = (predicted_wtype == wtype).sum().item()
            total_correct_wtype += correct_wtype

            total_samples += wtype.shape[0]
            
            # Reconstruction loss
            loss_r = criterion_r(r, data)
            losses_r.append(loss_r.item())

            # wtype classification loss
            loss_wtype = criterion_wtype(h_wtype, wtype)
            losses_wtype.append(loss_wtype.item())

            # frequency dist regression loss
            loss_freq_dist = 0
            # separate the indexes of sine and triangle waves
            sine_idx = torch.where(wtype == 0)
            triangle_idx = torch.where(wtype == 1)
            if len(sine_idx[0]) > 0:  # sanity-nan check
                sine_freqs = freq[sine_idx]
                sine_freq_dist = torch.cdist(sine_freqs, sine_freqs)
                h0_sine_dist = torch.cdist(h0[sine_idx], h0[sine_idx])
                loss_freq_dist += criterion_freq(h0_sine_dist, sine_freq_dist)
            if len(triangle_idx[0]) > 0:
                triangle_freqs = freq[triangle_idx]
                triangle_freq_dist = torch.cdist(triangle_freqs, triangle_freqs)
                h0_triangle_dist = torch.cdist(h0[triangle_idx], h0[triangle_idx])
                loss_freq_dist += criterion_freq(h0_triangle_dist, triangle_freq_dist)
            


            losses_freq_dist.append(loss_freq_dist.item())
            
        avg_loss_r = sum(losses_r) / len(losses_r)
        avg_loss_wtype = sum(losses_wtype) / len(losses_wtype)
        avg_loss_freq_dist = sum(losses_freq_dist) / len(losses_freq_dist)

        accuracy_wtype = total_correct_wtype / total_samples

    return avg_loss_r, avg_loss_wtype, avg_loss_freq_dist, accuracy_wtype
        

def eval_synth_freqcls(model, data_loader, device, criterion_r, criterion_wtype, criterion_freq):   
    model.to(device)
    model.eval()
    losses_r = []
    losses_wtype = []
    losses_freq_cls = []

    total_correct_wtype = 0
    total_correct_freq = 0
    total_samples = 0

    with torch.no_grad():
        for _, (data, wtype, group_ids) in enumerate(data_loader):
            data, wtype = data.to(device, dtype=torch.float), wtype.to(device)
            group_ids = group_ids.to(device)
            r, h, h_wtype, h0, b = model(data)
            
            # get accuracy
            _, predicted_wtype = torch.max(h_wtype, 1)
            correct_wtype = (predicted_wtype == wtype).sum().item()
            total_correct_wtype += correct_wtype

            _, predicted_group_id = torch.max(h, 1)
            correct_group_id = (predicted_group_id == group_ids).sum().item()
            total_correct_freq += correct_group_id

            total_samples += wtype.shape[0]

            # Reconstruction loss
            loss_r = criterion_r(r, data)
            losses_r.append(loss_r.item())

            # wtype classification loss
            loss_wtype = criterion_wtype(h_wtype, wtype)
            losses_wtype.append(loss_wtype.item())

            # frequency dist regression loss
            loss_freq_cls = criterion_freq(h, group_ids)
            losses_freq_cls.append(loss_freq_cls.item())

            loss = loss_r + loss_wtype + loss_freq_cls

        avg_loss_r = sum(losses_r) / len(losses_r)
        avg_loss_wtype = sum(losses_wtype) / len(losses_wtype)
        avg_loss_freq_cls = sum(losses_freq_cls) / len(losses_freq_cls)

        accuracy_wtype = total_correct_wtype / total_samples
        accuracy_freq = total_correct_freq / total_samples

    return avg_loss_r, avg_loss_freq_cls, avg_loss_wtype, accuracy_freq, accuracy_wtype

# synthetic spectra eval
def eval_synth_spec(model, data_loader, device, criterion_r, criterion_species, criterion_latent, obj, num_labels):
    model.to(device)
    model.eval()
    losses_r = []
    losses_species = []
    losses_latent = []


    total_correct_species = 0
    total_correct_group_id = 0
    total_samples = 0

    with torch.no_grad():
        for _, (data, species, latent, group_ids) in enumerate(data_loader):
            data, species, latent, group_ids = data.to(device, dtype=torch.float), species.to(device), latent.to(device, dtype=torch.float), group_ids.to(device)

            if obj == 'clusCLS':
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
            if obj == 'clusCLS':
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
            
        avg_loss_r = sum(losses_r) / len(losses_r)
        avg_loss_species = sum(losses_species) / len(losses_species)
        avg_loss_latent = sum(losses_latent) / len(losses_latent)

        accuracy_species = total_correct_species / total_samples
        accuracy_latent = total_correct_group_id / total_samples

    return avg_loss_r, avg_loss_species, avg_loss_latent, accuracy_species, accuracy_latent 
        
# driams data eval
def eval_amr(model, amr_val, data_loader, device, pad_len, criterion_r, criterion_amr, criterion_species):
    model.to(device)
    model.eval()
    losses_r = []
    losses_amr = []
    losses_species = []

    total_correct_species = 0
    total_samples = 0
    with torch.no_grad():
        for _, (data, species, idxs) in enumerate(data_loader):

            amr_profiles = [amr_val[i] for i in idxs]
            amr_sim = get_driams_sim_mat(amr_profiles)
            amr_dissim = len(amr_profiles[0])  - amr_sim
            amr_dissim = torch.from_numpy(amr_dissim).float().to(device)

            cal_loss_mask = torch.ones_like(data)# do not calculate losss for padded portions
            cal_loss_mask[:, :, -pad_len:] = 0
            non_pad_eles = data.shape[-1] - pad_len

            cal_loss_mask = cal_loss_mask.to(device)
            data = data.to(device, dtype=torch.float)
            species = species.to(device)

            r, h_species, h0, b = model(data)


            # get predictions and update accuracy stats
            _, predicted_species = torch.max(h_species, 1)
            correct_species = (predicted_species == species).sum().item()
            total_correct_species += correct_species

            total_samples += species.shape[0]
            
            # get losses for backprop
            loss_r = criterion_r(r, data)
            loss_r = loss_r * cal_loss_mask
            loss_r = loss_r.squeeze().sum() / non_pad_eles

            
            h0_cdist = torch.cdist(h0, h0, p=2)
            loss_amr = criterion_amr(h0_cdist, amr_dissim)


            loss_species = criterion_species(h_species, species)


            losses_amr.append(loss_amr.detach().cpu().item())
            losses_r.append(loss_r.detach().cpu().item())
            losses_species.append(loss_species.detach().cpu().item())


        avg_loss_r = sum(losses_r) / len(losses_r)
        avg_loss_species = sum(losses_species) / len(losses_species)
        avg_loss_amr = sum(losses_amr) / len(losses_amr)

        accuracy_species = total_correct_species / total_samples
        
    return avg_loss_r, avg_loss_species, avg_loss_amr, accuracy_species


# TODO custom loss for synth dataset?

# for proprietary SNP
class SNPLoss(nn.Module):
    def __init__(self, threshold=15):
        super(SNPLoss, self).__init__()
        self.threshold = threshold

    def forward(self, inputs, targets):
        # return torch.mean((torch.log(inputs + 1) - torch.log(targets + 1)) ** 2)
        
        # # Hinge loss for values above the threshold
        hinge_loss = torch.relu(self.threshold - inputs) ** 2
        # hinge_loss_small = torch.relu(inputs - self.threshold) ** 2

        
        # Combine the two losses: use MSE where targets < threshold, hinge loss where targets >= threshold
        mse_loss = (inputs - targets) ** 2 # * (self.threshold / (1 + targets))
        mask = (targets <= self.threshold)
        valid_targets = targets[mask]
        len_targets = len(targets.flatten())
        len_valid_targets = len(valid_targets)

        # if len_valid_targets == 0:
        #     weight_mse = 0
        #     weight_hinge = 1
        # elif len_valid_targets == len_targets:
        #     weight_mse = 1
        #     weight_hinge = 0
        # else:
        #     weight = 0.5
        #     weight_mse = weight * len_targets / (len_valid_targets)
        #     weight_hinge = (1 - weight) * len_targets / (len_targets - len_valid_targets)
            
        # scale = 0 if len(valid_targets) == 0 else torch.var(valid_targets)
        # loss = torch.where(targets <= self.threshold, mse_loss * weight_mse, hinge_loss * weight_hinge).flatten()
        # loss = torch.where(targets <= self.threshold, mse_loss + hinge_loss_small, hinge_loss).flatten()
        loss = torch.where(targets <= self.threshold, mse_loss, hinge_loss).flatten()

        # inputs_binary = inputs.clone()
        # inputs_binary[inputs_binary <= self.threshold] = 0
        # inputs_binary[inputs_binary > self.threshold] = 1

        # targets_binary = targets.clone()
        # targets_binary[targets_binary <= self.threshold] = 0
        # targets_binary[targets_binary > self.threshold] = 1

        # binary_mse = (inputs_binary - targets_binary) ** 2
        # binary_loss = torch.mean(binary_mse)

        # if len(valid_targets) > 1:
        #     y = -torch.combinations(valid_targets).diff(dim=1).sign().squeeze()
        #     _outputs = torch.combinations(inputs[mask])
        
        #     # We split the columns so we have the right parameters for margin ranking
        #     x1 = _outputs[:, 0].squeeze()
        #     x2 = _outputs[:, 1].squeeze()
        
        #     # Now we have everything in the correct format to
        #     # compute the margin ranking loss
        #     mr_loss = nn.functional.margin_ranking_loss(x1, x2, y, margin=0, reduction='mean')
        # else:
        #     mr_loss = 0


        
        # Return the mean of the loss
        return torch.mean(loss)# + binary_loss # + mr_loss


def extract_letters(input_string):
    # Using regular expression to find all letters
    letters = re.findall(r'[A-Za-z]', input_string)
    # Joining the list of letters into a single string
    return ''.join(letters)


def join_keys_with_common_values(original_dict):
    # Initialize an empty dictionary to store the result
    result_dict = {}
    
    # Iterate through the original dictionary
    for key, value in original_dict.items():
        # If the value is already a key in the result dictionary, append the key to the list
        if value in result_dict:
            result_dict[value].append(key)
        # If the value is not a key in the result dictionary, create a new entry with the value as key and the key as the first element in the list
        else:
            result_dict[value] = [key]
    
    return result_dict


# for comparing proprietary clusters
def cutoff(snp_matrix, cutoff=15):
    A = snp_matrix.copy()
    A[A <= cutoff] = 1
    A[A > cutoff] = 0
    return A


def get_snp_matrix_species(all_specIDs, snp_data_path = '../multi_species_data/proprietary_RealTime_Year1_SNP_distances.csv'):
    # organize all_specIDs into a dictionary of specID: [specIDs] where the specIDs have the same prefix split by '_'
    specID_prefix_dict = {specID: specID.split('_')[0] for specID in all_specIDs}
    prefix_specID_dict = join_keys_with_common_values(specID_prefix_dict)
    # print(prefix_specID_dict)
    # assert 1==0

    specID_idx_dict = {specID: idx for idx, specID in enumerate(all_specIDs)}
    specID_idx_dict_inv = {idx: specID for idx, specID in enumerate(all_specIDs)}

    identifiers = [extract_letters(specID) for specID in all_specIDs]
    specID_id_dict = {i: j for i, j in zip(all_specIDs, identifiers)}
    joined_id_dict = join_keys_with_common_values(specID_id_dict)
    joined_id_idx_dict = {key: sorted([specID_idx_dict[specID] for specID in value]) for key, value in joined_id_dict.items()}

    snp_data = pd.read_csv(snp_data_path, index_col=0)

    # generate an SNP matrix for each species
    joined_snp_dict = {key: np.full((len(value), len(value)), -100) for key, value in joined_id_idx_dict.items()}
    for value in joined_snp_dict.values():
        np.fill_diagonal(value, 0)

    for i in track(range(len(snp_data)), description='generating SNP matrix for each species'):
        id1 = snp_data.iloc[i, 0]
        id2 = snp_data.iloc[i, 1]
        if id1 not in specID_idx_dict or id2 not in specID_idx_dict:
            continue
        key1 = extract_letters(id1)
        key2 = extract_letters(id2)
        if key1 != key2: 
            # print(key1, key2)
            continue
        assert key1 == key2   # assert that SNPs are only computed only between the same species
        # TODO: get ALL specids with prefix id1 and all specids with prefix id2

        g_idx1s = [specID_idx_dict[x] for x in prefix_specID_dict[id1]]
        g_idx2s = [specID_idx_dict[x] for x in prefix_specID_dict[id2]]

        l_idx1s = [joined_id_idx_dict[key1].index(g_idx1) for g_idx1 in g_idx1s]
        l_idx2s = [joined_id_idx_dict[key1].index(g_idx2) for g_idx2 in g_idx2s]
        # print(key1, g_idx1s, g_idx2s, l_idx1s, l_idx2s)
    # assert 1==0
        for l10, l11 in product(l_idx1s, l_idx1s):
            joined_snp_dict[key1][l10, l11] = 0
        for l20, l21 in product(l_idx2s, l_idx2s):
            joined_snp_dict[key1][l20, l21] = 0
        for l_idx1, l_idx2 in product(l_idx1s, l_idx2s):
            joined_snp_dict[key1][l_idx1, l_idx2] = snp_data.iloc[i, 2]
            joined_snp_dict[key1][l_idx2, l_idx1] = snp_data.iloc[i, 2]

    # post-process the SNP matrices
    for value in joined_snp_dict.values():
        max_value = np.max(value)
        value[value == -100] = max_value * 2
    
    joined_snp_dict_with_specID = {key: {'specIDs': [specID_idx_dict_inv[x] for x in joined_id_idx_dict[key]], 'SNP': value, 'cnc': connected_components(csgraph=csr_matrix(cutoff(value)), directed=False, return_labels=True)[1]} for key, value in joined_snp_dict.items() if len(value) > 1}

    # print(joined_snp_dict_with_specID['EC'])
    # assert 1==0
    return joined_snp_dict_with_specID


def get_sub_snp_matrix_from_specIDs(joined_snp_dict_with_specID, specIDs, device):
    # get the sub-dictionary of the SNP matrix with the given specIDs
    # step 1 organize the specIDs into a species: specIDs dictionary
    specID_species_dict = {specID: extract_letters(specID) for specID in specIDs}
    species_specIDs_dict = join_keys_with_common_values(specID_species_dict)
    # step 2 get the SNP matrix for each species in species_specIDs_dict
    sub_snp_dict = {}
    for species, specIDs in species_specIDs_dict.items():
        if species not in joined_snp_dict_with_specID:
            continue
        # specIDs, snp_matrix = joined_snp_dict_with_specID[species]
        all_specIDs = joined_snp_dict_with_specID[species]['specIDs']
        snp_matrix = joined_snp_dict_with_specID[species]['SNP']
        sub_idxs = []
        for specID in specIDs:
            if specID not in all_specIDs:
                continue
            else:
                sub_idxs.append(all_specIDs.index(specID))
        sub_snp_matrix = snp_matrix[sub_idxs][:, sub_idxs]
        sub_snp_dict[species] = {'specIDs': specIDs, 'SNP': torch.Tensor(sub_snp_matrix).to(device)}
    return sub_snp_dict


def get_predicted_components(nn_embeddings, cutoff):
    emb_dist_mat = cdist(nn_embeddings, nn_embeddings, 'euclidean')
    # emb_dist_mat_binary = emb_dist_mat.copy()
    # emb_dist_mat_binary[emb_dist_mat_binary <= cutoff] = 1
    # emb_dist_mat_binary[emb_dist_mat_binary > cutoff] = 0
    # emb_dist_mat_csr = csr_matrix(emb_dist_mat_binary)
    # n_components_nn, labels_nn = connected_components(csgraph=emb_dist_mat_csr, directed=False, return_labels=True)
    n_components_nn = AgglomerativeClustering(n_clusters=None, distance_threshold=cutoff, linkage='ward').fit(nn_embeddings).labels_
    return n_components_nn, emb_dist_mat

def get_predicted_components_from_dist(emb_dist_mat, cutoff):
    emb_dist_mat_binary = emb_dist_mat.copy()
    emb_dist_mat_binary[emb_dist_mat_binary <= cutoff] = 1
    emb_dist_mat_binary[emb_dist_mat_binary > cutoff] = 0
    emb_dist_mat_csr = csr_matrix(emb_dist_mat_binary)
    n_components_nn, labels_nn = connected_components(csgraph=emb_dist_mat_csr, directed=False, return_labels=True)
    return labels_nn


def get_predicted_components_num(nn_embeddings, num_clusters):
    # do hierarchical clustering
    n_components_nn = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward').fit(nn_embeddings).labels_
    return n_components_nn


def get_nonsingle_indices(cnc):
    values= []
    for i in range(cnc.max() + 1):
        if (cnc == i).sum() > 1:
            values.append(i)
    indices = np.where(np.isin(cnc, values))[0]
    return indices


def get_purity(cnc, pred):
    # get the purity of cnc clustering w.r.t. pred labels
    unique_clusters = np.unique(cnc)
    # get the indices of the majority class in each cluster
    purity = 0
    for cluster in unique_clusters:
        indices = np.where(cnc == cluster)[0]
        cluster_labels = pred[indices]
        majority_class = np.argmax(np.bincount(cluster_labels))
        purity += len(np.where(cluster_labels == majority_class)[0]) / len(indices)
    
    purity = 0 if len(unique_clusters) == 0 else purity / len(unique_clusters)
    return purity


def get_metrics(cnc_snp, cnc_nn):
    nonsingle_indices = get_nonsingle_indices(cnc_snp)
    rand_score = adjusted_rand_score(cnc_snp, cnc_nn)
    nmi_score = normalized_mutual_info_score(cnc_snp, cnc_nn)
    # ns_rand_score = adjusted_rand_score(cnc_snp[nonsingle_indices], cnc_nn[nonsingle_indices])
    # ns_nmi_score = normalized_mutual_info_score(cnc_snp[nonsingle_indices], cnc_nn[nonsingle_indices])
    # print(cnc_snp[nonsingle_indices], cnc_nn[nonsingle_indices])
    recall = get_purity(cnc_snp[nonsingle_indices], cnc_nn[nonsingle_indices])
    precision = get_purity(cnc_nn[nonsingle_indices], cnc_snp[nonsingle_indices])
    f1_score = 2 * recall * precision / (recall + precision + 1e-10)

    return recall, precision, f1_score, rand_score, nmi_score, len(set(cnc_snp[nonsingle_indices]))


# For DRIAMS data 
def get_binned_data(file_path, bins=np.arange(2000, 20001, 3)):
    with open(file_path, 'r') as handle:
        data_lines = [line.strip().split(' ') for line in handle.readlines()][3:]
        data = np.array(data_lines, dtype=float)
        data_binned = np.histogram(data[:, 0], bins=bins, weights=data[:, 1])[0]
    return data_binned, data


def read_csv_and_fetch(driams_preprocessed_root, driams_table, s, drug_names): #TODO
    data_matrix = []
    species_list = []
    amrs = []

    for i in track(range(len(driams_table)), description='Processing DRIAMS-{}'.format(s)):
        filename = driams_table.loc[i, 'code']
        file_path = osp.join(driams_preprocessed_root, filename+'.txt')
        if not osp.isfile(file_path):
            continue
        species = driams_table.loc[i, 'species']
        species_list.append(species.lower())
        data_binned, data_file = get_binned_data(file_path)
        data_matrix.append(data_binned)
        amr = driams_table.loc[i, drug_names].fillna('N')
        amrs.append(amr.values.tolist())
    
    data_matrix = np.array(data_matrix)
    print(f'data matrix shape: {data_matrix.shape}')
    return data_matrix, species_list, amrs


DRIAMS_LABEL_IDX_DICT = {'S': 0, 'I':1, 'R':2, '1':3, '0':4, 'N':5}
DRIAMS_SIM_TABLE = np.array([[1, 0, 0, 0, 1, 0], 
                             [0, 1, 0, 1, 0, 0],
                             [0, 0, 1, 1, 0, 0],
                             [0, 1, 1, 1, 0, 0],
                             [1, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0]], dtype=int)


def driams_sim(A, B):
    return sum([DRIAMS_SIM_TABLE[a, b] for a, b in zip(A, B)])


def get_driams_sim_mat(data): # list of strings
    data_np = np.zeros((len(data), len(data[0])))
    for i, d in enumerate(data):
        data_np[i] = np.array([DRIAMS_LABEL_IDX_DICT[x] for x in d])
    data_np = data_np.astype(int)
    result = cdist(data_np, data_np, metric=driams_sim)
    # correct diagonal
    np.fill_diagonal(result, len(data_np[0]))
    return result