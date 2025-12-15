import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import sys
import pandas as pd


def read_pdb_protein(pdbname):
    # Read information from PDB file
    fid1 = open(pdbname, 'r')
    row = 0
    while True:
        line = fid1.readline()
        if not line:
            break
        row += 1
    fid1.close()
    MAX = row

    residall = []
    resall = []
    coorall = []
    bexp = []

    fid = open(pdbname, 'r')
    for i in range(MAX):
        pdb = fid.readline()
        if len(pdb) <= 3:
            continue
        elif pdb[0:3] == 'END':
            break
        elif pdb[0:4] == 'ATOM' and pdb[13] == 'C' and pdb[14] == 'A':
            resid = int(pdb[22:26])
            res = pdb[17:20]
            b = float(pdb[60:66])
            coorx = float(pdb[30:38])
            coory = float(pdb[38:46])
            coorz = float(pdb[46:54])
            coor = [coorx, coory, coorz]
            residall.append(resid)
            resall.append(res)
            coorall.append(coor)
            bexp.append(b)
    fid.close()

    return residall, resall, coorall, bexp


# Protein files
filenames = ['data/1arb.pdb']
all_start_residues = [[50]]
select = 0
filename = filenames[select]

# Read and parse protein file
residall, resall, coorall, bexp = read_pdb_protein(filename)
n = len(bexp)
cutoff = 8.6
mode = None
N_state = 10
T = 50
start_residues = all_start_residues[select]
start_indexes = [residall.index(i) for i in start_residues]
residall = np.array(residall)
tau = 5

threshold_percentage = 0.95
output_path_num = 5

is_plot = True
is_write_file = False


def get_path_name(path):
    count = sum(1 for prev, current in zip(
        path, path[1:]) if abs(current - prev) > 10)
    return "path-0" if count == 0 else ("path-1" if count == 1 else "path-2")


def get_protein_name(filename):
    start_index = filename.rfind("/") + 1
    end_index = filename.rfind(".pdb")

    if start_index != -1 and end_index != -1:
        extracted_string = filename[start_index:end_index]
        return extracted_string
    else:
        return "unrecognized_protein_name"


def print_padded_header(header):
    desired_length = 100
    padding = "=" * ((desired_length - len(header)) // 2)
    added_header = header.center(desired_length, "=")
    print(added_header)


def plot_histogram(path, name, n):
    colors = ['#0074D9', '#2ECC40', '#FF851B', '#FF4136', '#B10DC9',
              '#FFDC00', '#001F3F', '#001F3F', '#F012BE', '#AAAAAA']
    # Calculate number of bins
    num_bins = (n - 0) // 2 + 1

    label = get_path_name(path)
    hist, bin_edges = np.histogram(
        path, bins=num_bins, range=(0, num_bins * 2), density=True)
    hist = hist / np.sum(hist)
    plt.bar(bin_edges[:-1], hist, width=2, alpha=0.5, edgecolor='k',
            align='edge', color=colors[random.randint(0, 9)])

    plt.xlabel('Residue Index')
    plt.ylabel('Frequency')
    plt.title(name)
    plt.show()


def save_to_excel(mat, mat_name):
    df = pd.DataFrame(mat)
    df.to_excel(mat_name + ".xlsx", index=False, header=False)


def get_hoff_mat(coorall, cutoff):
    print_padded_header("1. Calculate Hoff matrix")

    netmat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            dis = np.sqrt((coorall[i][0] - coorall[j][0]) ** 2 +
                          (coorall[i][1] - coorall[j][1]) ** 2 +
                          (coorall[i][2] - coorall[j][2]) ** 2)
            if dis <= cutoff:
                netmat[i, j] = -1
                netmat[j, i] = netmat[i, j]

    np.fill_diagonal(netmat, -(netmat.sum(axis=1) - np.diag(netmat)))
    N_state = int(np.mean(np.diagonal(netmat)))

    print("netmat:\n {} \n shape: {}".format(netmat, netmat.shape))
    print("N state: {}".format(N_state))
    if is_write_file:
        np.savetxt("netmat.txt", netmat, delimiter=",")

    return netmat


def get_inversed_hoff_mat(hoff_mat):
    print_padded_header("2. Calculate inverse of Hoff matrix, eigenvectors, eigenvalues")
    """
    [eigenvectors, eigenvalues, _] = np.linalg.svd(netmat) 
    print(eigenvectors.shape, _.shape)
    eigenvalues, eigenvectors = np.around(eigenvalues, decimals=4), np.around(eigenvectors, decimals=4)
    eigenvalues, eigenvectors =  _sort_eigenvalues(eigenvalues, eigenvectors)

    hoff_mat_inversed = (eigenvectors) * (np.linalg.pinv(np.diag(eigenvalues))) * (eigenvectors.T)
    hoff_mat_inversed = np.around(hoff_mat_inversed, decimals=4)
    """

    # """"
    eigenvalues, eigenvectors = np.linalg.eigh(hoff_mat)
    eigenvalues, eigenvectors = np.around(eigenvalues, decimals=4), np.around(eigenvectors, decimals=4)
    eigenvalues, eigenvectors = _sort_eigenvalues(eigenvalues, eigenvectors)

    hoff_mat_inversed = np.zeros_like(hoff_mat)
    for i in range(hoff_mat_inversed.shape[0]):
        for j in range(hoff_mat_inversed.shape[1]):
            for k in range(1, mode):
                hoff_mat_inversed[i, j] = hoff_mat_inversed[i, j] + \
                                          eigenvectors[i, k] * eigenvectors[j, k] / eigenvalues[k]
    hoff_mat_inversed = np.around(hoff_mat_inversed, decimals=4)
    # """

    print("hoff_mat_inversed:\n {} \n".format(hoff_mat_inversed))

    if is_write_file:
        np.savetxt("hoff_mat_inversed.txt", hoff_mat_inversed, delimiter=",")
        save_to_excel(hoff_mat_inversed, "hoff_mat_inversed")

    return hoff_mat_inversed, eigenvalues, eigenvectors


def _sort_eigenvalues(eigenvalues, eigenvectors):
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices[:: 1]]
    eigenvectors = eigenvectors[:, sorted_indices[:: 1]]

    return eigenvalues, eigenvectors


def get_A_ij(eigenvalues, eigenvectors):
    print_padded_header("3. Calculate A_ij")

    A_ij = {}

    for k in range(0, n):
        if eigenvalues[k] > 0.001:  #
            A_ij[k] = np.around((eigenvectors[:, k] * (np.array([eigenvectors[:, k]]).T) / eigenvalues[k]), decimals=4)
        else:
            A_ij[k] = np.zeros([n, n])

    print("A_ij[{}]: {} \n".format(k, A_ij[k]))
    if is_write_file:
        pd.DataFrame(A_ij[k]).to_excel("A_ij_{}.xlsx".format(k), index=True)

    return A_ij


def get_transfer_entropy(A_ij, tau, eigenvalues):
    print_padded_header("4. Calculate transfer entropy")

    # Initialize transfer entropy matrix TE with dimension (N,N) and complex data type
    trans_entropy = np.ones((n, n), dtype=np.complex_)

    for i in range(n):
        for j in range(n):
            aEk = [A_ij[k][j][j] for k in range(0, n)]
            bEk = [A_ij[k][i][j] for k in range(0, n)]
            cEk = [A_ij[k][j][j] for k in range(0, n)]
            dEk = [A_ij[k][i][j] for k in range(0, n)]
            eEk = [A_ij[k][i][i] for k in range(0, n)]

            aEk = aEk * np.exp(-eigenvalues * tau)
            bEk = bEk * np.exp(-eigenvalues * tau)

            a = np.sum(cEk) ** 2 - np.sum(aEk) ** 2
            b = (np.sum(eEk) * np.sum(cEk) ** 2)
            c = 2 * (np.sum(dEk)) * np.sum(aEk) * np.sum(bEk)
            d = -(((np.sum(bEk) ** 2) + (np.sum(dEk) ** 2)) *
                  (np.sum(cEk))) - ((np.sum(aEk) ** 2) * np.sum(eEk))
            f = np.sum(cEk)
            g = (np.sum(eEk) * np.sum(cEk)) - (np.sum(dEk) ** 2)
            if i == j:
                trans_entropy[i][j] = 0
            else:
                trans_entropy[i][j] = 0.5 * np.log(a) - 0.5 * np.log(
                    b + c + d) - 0.5 * np.log(f) + 0.5 * np.log(g)

    trans_entropy[trans_entropy < 0] = 0
    trans_entropy_float = np.abs(trans_entropy).astype(float)
    trans_entropy_float[trans_entropy_float < 0] = 0
    print("trans_entropy_float: {}".format(trans_entropy_float))

    if is_write_file:
        pd.DataFrame(trans_entropy_float).to_excel("trans_entropy_float.xlsx", index=False)

    # Normalize trans_entropy_float
    min_value = np.min(trans_entropy_float)
    max_value = np.max(trans_entropy_float)
    normalized_trans_entropy_float = (
                                             trans_entropy_float - min_value) / (max_value - min_value)

    # Create DataFrame with normalized data
    df = pd.DataFrame(normalized_trans_entropy_float)
    # Write DataFrame to Excel file
    df.to_excel('normalized_trans_entropy_float.xlsx', index=False, header=False)

    # Read data from Excel
    df = pd.read_excel('normalized_trans_entropy_float.xlsx', header=None)
    # Calculate sum of each row, axis=1 means sum by row
    sum_of_rows = df.sum(axis=1)
    # Convert sum results to DataFrame
    sum_df = pd.DataFrame(sum_of_rows, columns=['Sum'])
    # Write results to Excel
    sum_df.to_excel('sum_of_normalized_trans_entropy_float_rows.xlsx', index=False)

    if is_plot:
        plt.figure()
        m1 = 64
        map = plt.get_cmap('jet', m1)
        plt.pcolormesh(normalized_trans_entropy_float, cmap=map)
        plt.gca().set_aspect('equal')
        plt.colorbar()
        plt.xlabel('Residue Index')
        plt.ylabel('Residue Index')
        plt.title('normalized_trans_entropy_float')
        plt.show()

    return trans_entropy_float, normalized_trans_entropy_float


def get_net_trans_entropy_float(trans_entropy_float):
    print_padded_header("4.1 Calculate net transfer entropy")

    # Calculate symmetric difference of TE matrix (netTE) # Net entropy transfer from each residue
    net_TE = trans_entropy_float - trans_entropy_float.T
    print("net_TE: {}".format(net_TE))
    if is_write_file:
        pd.DataFrame(net_TE).to_excel("net_TE.xlsx", index=False)

    # Normalize net_TE
    min_value = np.min(net_TE)
    max_value = np.max(net_TE)
    normalized_net_TE = (net_TE - min_value) / (max_value - min_value) * 2 - 1
    print_padded_header("4.2 Calculate normalized net transfer entropy")
    print("normalized_net_TE: {}".format(normalized_net_TE))
    if is_write_file:
        pd.DataFrame(normalized_net_TE).to_excel("normalized_net_TE.xlsx", index=False)

    if is_plot:
        plt.figure()
        m1 = 64
        map = plt.get_cmap('jet', m1)
        plt.pcolormesh(normalized_net_TE, cmap=map)
        plt.gca().set_aspect('equal')
        plt.colorbar()
        plt.xlabel('Residue Index')
        plt.ylabel('Residue Index')
        plt.title('normalized_net_TE')
        plt.show()

    return net_TE, normalized_net_TE


if __name__ == "__main__":

    with open("output.txt", "w") as f:

        if is_write_file:
            original_stdout = sys.stdout
            sys.stdout = f

        # 1. Construct adjacency matrix
        netmat = get_hoff_mat(coorall, cutoff)

        # 2. Calculate inverse of Hoff adjacency matrix using eigenvalue decomposition
        hoff_mat_inversed, eigenvalues, eigenvectors = get_inversed_hoff_mat(hoff_mat=netmat)

        # 3. Calculate A_ij
        A_ij = get_A_ij(eigenvalues, eigenvectors)

        # 4. Calculate transfer entropy
        trans_entropy_float, normalized_trans_entropy_float = get_transfer_entropy(A_ij, tau, eigenvalues)
        trans_entropy_float = np.around(trans_entropy_float, decimals=4)

        # 5. HMM
        print_padded_header("5. Calculate HMM")
        adj_residues_indices = (netmat == -1)
        print("adj_residues_indices:\n {} \n".format(adj_residues_indices))
        if is_write_file:
            np.savetxt("adj_residues_indices.txt", adj_residues_indices, delimiter=",")

        # 5.1 Transition probability matrix

        adj_normalized_trans_entropy_float = normalized_trans_entropy_float * adj_residues_indices
        print("adj_normalized_trans_entropy_float:\n {} \n".format(adj_normalized_trans_entropy_float))
        if is_write_file:
            np.savetxt("adj_normalized_trans_entropy_float.txt", adj_normalized_trans_entropy_float, delimiter=",")
        df = pd.DataFrame(adj_normalized_trans_entropy_float)
        # Write DataFrame to Excel
        df.to_excel('adj_normalized_trans_entropy_float.xlsx', index=False, header=False)

        # Take top N mutual information of adjacency
        top_N_indices = np.argsort(adj_normalized_trans_entropy_float, axis=1)[:, -N_state:]
        bool_matrix = np.zeros_like(adj_normalized_trans_entropy_float, dtype=bool)
        bool_matrix[np.arange(bool_matrix.shape[0])[:, np.newaxis], top_N_indices] = True
        adj_topN_normalized_trans_entropy_float = adj_normalized_trans_entropy_float * bool_matrix
        print("top_N adj_normalized_trans_entropy_float:\n {} \n".format(adj_topN_normalized_trans_entropy_float))
        if is_write_file:
            np.savetxt("adj_topN_normalized_trans_entropy_float.txt", adj_topN_normalized_trans_entropy_float,
                       delimiter=",")
        df = pd.DataFrame(adj_topN_normalized_trans_entropy_float)
        df.to_excel("adj_topN_normalized_trans_entropy_float.xlsx", index=False)

        # Calculate transition probability matrix
        transition_proportionality_factor = np.sum(adj_normalized_trans_entropy_float, axis=1)
        transition_prob_mat = adj_normalized_trans_entropy_float / \
                              transition_proportionality_factor[:, np.newaxis]
        transition_prob_mat = np.around(transition_prob_mat, decimals=4)
        print("transition_prob_mat:\n {} \n".format(transition_prob_mat))
        if is_write_file:
            np.savetxt("transition_prob_mat.txt", transition_prob_mat, delimiter=",")

        # 5.2 Emission probability matrix
        # step 0 Calculate communication capability
        comm_capabilities_vec = np.sum(
            normalized_trans_entropy_float, axis=1)
        print("comm_capabilities:\n {} \n".format(comm_capabilities_vec))
        if is_write_file:
            np.savetxt("comm_capabilities.txt", comm_capabilities_vec, delimiter=",")

        # step1 Expand communication capability vector into NxN matrix
        comm_capabilities_mat = np.tile(
            comm_capabilities_vec, (comm_capabilities_vec.shape[0], 1))
        print("comm_capabilities_mat:\n {} \n".format(comm_capabilities_mat))
        if is_write_file:
            np.savetxt("comm_capabilities_mat.txt", comm_capabilities_mat, delimiter=",")

        # step2 Perform AND operation between the above matrix and adjacency matrix
        adj_comm_capabilities_mat = comm_capabilities_mat * adj_residues_indices
        print("adj_comm_capabilities_mat:\n {} \n".format(adj_comm_capabilities_mat))
        if is_write_file:
            np.savetxt("adj_comm_capabilities_mat.txt", adj_comm_capabilities_mat, delimiter=",")

        # step3 Keep top K largest values per row from the above result
        top_N_indices = np.argsort(adj_comm_capabilities_mat, axis=1)[:, -N_state:]
        bool_matrix = np.zeros_like(adj_comm_capabilities_mat, dtype=bool)
        bool_matrix[np.arange(bool_matrix.shape[0])[:, np.newaxis], top_N_indices] = True
        adj_topN_comm_capabilities_mat = adj_comm_capabilities_mat * bool_matrix
        print("adj_topN_comm_capabilities_mat:\n {} \n".format(adj_topN_comm_capabilities_mat))
        if is_write_file:
            np.savetxt("adj_topN_comm_capabilities_mat.txt", adj_topN_comm_capabilities_mat, delimiter=",")

        # step4 Calculate normalization denominator by summing each row; divide each value by row sum; get emission probability matrix
        emission_proportionality_factor = np.sum(
            adj_topN_comm_capabilities_mat, axis=1)
        emission_prob_mat = adj_topN_comm_capabilities_mat / \
                            emission_proportionality_factor[:, np.newaxis]
        emission_prob_mat = np.around(emission_prob_mat, decimals=4)
        print("emission_prob_mat:\n {} \n".format(emission_prob_mat))
        if is_write_file:
            np.savetxt("emission_prob_mat.txt", emission_prob_mat, delimiter=",")


        def get_cur_timestep_max_prob(deltas, pre_residues):
            max_value = -1
            max_pre_redidue = None
            for delta, pre_residue in zip(deltas, pre_residues):
                if np.max(delta) > max_value:
                    max_value = np.max(delta)
                    max_pre_redidue = pre_residue

            return max_value, max_pre_redidue


        # 5.3 Calculate possible paths from a node using Viterbi algorithm
        for start_index, start_residue in zip(start_indexes, start_residues):

            pre_deltas = [np.log(emission_prob_mat[start_index, :] + 1)]
            pre_residues = [start_index]
            pre_paths = [[start_index]]
            pre_paths_probs = [0]

            # Generate each step
            for t in range(1, T):
                cur_deltas = []
                cur_residues = []
                cur_paths = []
                cur_paths_probs = []

                for pre_delta, pre_path in zip(pre_deltas, pre_paths):
                    pre_delta[pre_path] = 0

                temps = []
                # todo: Take maximum value from all deltas in current step? Or take maximum from each delta in current step's deltas?
                for pre_delta, pre_residue in zip(pre_deltas, pre_residues):
                    """
                    Add a judgment
                    """
                    temps.append((pre_delta + np.log(transition_prob_mat[pre_residue, :] + 1)) \
                                 * adj_residues_indices[pre_residue, :])

                    max_value, max_pre_residue = get_cur_timestep_max_prob(temps,
                                                                           pre_residues)
                    threshold = np.around(max_value + np.log(threshold_percentage), decimals=4)

                # Iterate over possible next states max_i that meet the threshold in current step
                for temp, pre_delta, pre_residue, pre_path in zip(temps, pre_deltas, pre_residues, pre_paths):
                    max_i = np.where(temp > threshold)[0]
                    for i in max_i:

                        cur_delta = (temp + \
                                     np.log(emission_prob_mat[pre_residue, :] + 1)) * adj_residues_indices[pre_residue,
                                                                                      :]

                        if np.all(cur_delta == 0.0) or i == pre_residue:
                            continue

                        else:
                            cur_deltas.append(cur_delta)
                            cur_residues.append(i)
                            cur_paths_probs.append(pre_delta[i])
                            cur_paths.append(pre_path + [i])

                pre_deltas = copy.deepcopy(cur_deltas)
                pre_residues = copy.deepcopy(cur_residues)
                pre_paths = copy.deepcopy(cur_paths)
                pre_paths_probs = copy.deepcopy(cur_paths_probs)

            print(filename)
            print("Node {} generated {} paths: ".format(start_residue, len(pre_paths)))
            sorted_indices = sorted(
                range(len(pre_paths_probs)), key=lambda i: pre_paths_probs[i], reverse=True)
            for i, id in enumerate(sorted_indices):
                if i > output_path_num:
                    break
                path = residall[pre_paths[id]].tolist()
                print("===")

                print(path, "\t")
                if is_plot:
                    plot_histogram(path, get_protein_name(filename), n)

        if is_write_file:
            sys.stdout = original_stdout