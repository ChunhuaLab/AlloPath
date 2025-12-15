# AlloPath: a Method for Identifying Protein Allosteric Pathway Based on Transfer Entropy and Hidden Markov Model

Here, we develop AlloPath, an effective method for predicting protein allosteric pathways that integrates transfer entropy (TE), Hidden Markov Model (HMM), and the Viterbi dynamic programming algorithm.

Authors: Jingjie Su, Xinyu Zhang, Jilong Zhang and Chunhua Li.

The process includes two steps: optimal cutoff radius dertermination in GNM and allosteric pathway prediction.

## Step 1 optimal cutoff radius search

* Python version: 3.7.9

pip install numpy>=1.21.5

pip install scipy>=1.7.3

Run the "gnm_cutoff_search.py", please note that the Python program file and the downloaded PDB file should be located in the same folder.

a. PDB File Processing

pdb_read extracts Cα atom coordinates, residue numbers, and experimental B-factors by filtering ATOM/HETATM records with atom name "CA". The input protein structure (in PDB format) can be retrieved from the Protein Data Bank (PDB, https://www.rcsb.org/).

b. Distance Matrix

Compute pairwise distances between all Cα atoms using scipy.spatial.distance.pdist and convert them to a square matrix.

c. Cutoff radius Search Loop

Within the preset cutoff range (6.0–15.0 Å), each possible radius value is systematically traversed at a fixed step size (0.5 Å ). For each radius, the Pearson Correlation Coefficient (PCC) between the theoretical B-factors and experimental B-factors is calculated, and the corresponding PCC value for that radius is recorded.

d. Finds optimal cutoff radius

By comparing the PCCs corresponding to all the cutoff radius, the optimal cutoff radius that yields the maximum PCC between the theoretical and experimental B-factors is determined.

## Step 2 allosteric pathway prediction

* Python version: 3.7.9

pip install biopython >=1.78

pip install pandas >=1.3.5  

pip install numpy >=1.21.5

pip install matplotlib >=3.3.2

1. Prepare the PDB file of the target protein (e.g., 1arb.pdb) and place it in the "data" directory. 

2. Modify the configuration parameters in the te_hmm_nor.py file as needed:

filenames: List of PDB file paths

all_start_residues: List of starting residues (allosteric site) for pathway prediction

cutoff: The optimal cutoff radius calculated in the step 1

T: Number of steps for HMM pathway prediction

tau: Time lag parameter for transfer entropy calculation

threshold_percentage: Threshold for pathwy selection

output_path_num: Number of top pathways to output

3. Run the "te_hmm_nor.py"

a. Transfer Entropy Calculation

Compute transfer entropy between residues to measure information flow and scales it to the [0, 1] interval. 

b. HMM Pathway Prediction

Construct transition probability matrix from normalized transfer entropy data.

Communication capability of a residue is the sum of normalized transfer entropy values between it and all other residues.

Calculate emission probability  based on communication capabilities. 

c. Implements HMM and Viterbi algorithms to identify the most probable alloscteric pathway.

## Help

For any questions, please contact us by chunhuali@bjut.edu.cn and jingjiesu@emails.bjut.edu.cn.



