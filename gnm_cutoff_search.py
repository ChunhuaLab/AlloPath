import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh


def gnm_find(filename='1arb.pdb'):
    # Read Cα atom information from PDB file
    posall, resall, bf = pdb_read(filename)
    N = len(resall)
    residue_number = N

    if N == 0:
        raise ValueError("No Cα atoms found, please check the PDB file")

    # Define cutoff radius range (6 to 15Å, step 0.5Å)
    r_c_values = np.arange(6.0, 15.1, 0.5)
    pmax = -np.inf
    r_c_select = 0.0
    best_bfactors = None


    dist_matrix = squareform(pdist(posall))

    for r_c in r_c_values:
        # Construct Laplacian matrix (network matrix)
        # Off-diagonal elements: -1 if distance ≤ cutoff radius, 0 otherwise
        netmat = np.where((dist_matrix <= r_c) & (dist_matrix != 0), -1, 0)
        # Diagonal elements: negative of row sums
        np.fill_diagonal(netmat, -np.sum(netmat, axis=1))

        # Eigen decomposition (extract non-zero eigenvalues; zero eigenvalues in GNM correspond to global translation)
        eig_vals, eig_vecs = eigh(netmat)

        # Check network connectivity (zero eigenvalue should be unique)
        zero_eig_mask = np.isclose(eig_vals, 0.0, atol=1e-6)
        if np.sum(zero_eig_mask) > 1:
            continue  

        # Exclude zero eigenvalues (take first non-zero eigenvalue and beyond)
        non_zero_idx = np.where(~zero_eig_mask)[0]
        if len(non_zero_idx) == 0:
            continue 

        eig_vals = eig_vals[non_zero_idx]
        eig_vecs = eig_vecs[:, non_zero_idx]

        # Calculate theoretical B-factors (sum of squared eigenvectors divided by eigenvalues)
        bfactors = np.sum(eig_vecs ** 2 / eig_vals, axis=1)

        # Calculate correlation coefficient between theoretical and experimental B-factors
        current_r = np.corrcoef(bfactors, bf)[0, 1]

        # Update optimal results
        if current_r > pmax:
            pmax = current_r
            r_c_select = r_c
            best_bfactors = bfactors

    return r_c_select, pmax, residue_number, best_bfactors


def pdb_read(filename):
    posall = []
    resall = []
    bf = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) < 76:
                continue

            # Filter ATOM or HETATM records
            record_type = line[0:6].strip()
            if record_type not in ('ATOM', 'HETATM'):
                continue

            # Filter Cα atoms (atom name should be 'CA')
            atom_name = line[12:16].strip()
            if atom_name != 'CA':
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue  

            try:
                resnum = int(line[22:26].strip())
            except ValueError:
                continue  

            try:
                bfactor = float(line[60:66].strip())
            except ValueError:
                continue  

            posall.append([x, y, z])
            resall.append(resnum)
            bf.append(bfactor)

    return np.array(posall), np.array(resall), np.array(bf)


# Example usage
if __name__ == "__main__":
    r_c, pcc, n_res, b_factors = gnm_find('1arb.pdb')
    print(f"Optimal cutoff radius: {r_c:.1f} Å")
    print(f"Maximum correlation coefficient: {pcc:.4f}")
    print(f"Number of residues: {n_res}")