# TorqueClusteringPy
This is a work-in-progress to replicate the TorqueClustering algorithm described in Jie Yang and Chin-Teng Lin, “Autonomous clustering by fast find of mass and distance peaks,” IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), DOI: 10.1109/TPAMI.2025.3535743

It performs autonomous clustering by fast find of mass and distance peaks.

current status: core algorithm is mostly working but still dealing with issues in replicating all results for the complete dataset.

This fork of the original code strictly follows licensing instructions:

This repository is licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).

❌ Prohibition of Commercial Use
This software MAY NOT be used for any commercial purposes.

Any form of selling, paid services, SaaS deployment, or monetization based on this repository is strictly prohibited.

🔄 Forking Rules
You are allowed to fork this repository, but you MUST retain this exact license (CC BY-NC-SA 4.0).¨


**
# TorqueClustering Results (2025 March 16 status)

## Summary Table

| Dataset | Clusters Found | NMI | AC | AMI | Output File |
|---------|---------------|-----|----|----|-------------|
| Highly overlapping | 3 | 0.4924 | 0.2052 | 0.2028 | Fig_S3A.png |
| FLAME | 7 | 0.6068 | 0.4833 | 0.4198 | Fig_S3B.png |
| Spectral-path | 7 | 0.5272 | 0.4231 | 0.3551 | Fig_S3C.png |
| Unbalanced | 2 | 0.7132 | 0.9000 | 0.0000 | Fig_S3D.png |
| Noisy | 54 | - | - | - | Fig_S3E.png |
| Heterogeneous geometric | 4 | 0.8935 | 0.8550 | 0.6297 | Fig_S3F.png |
| Multi-objective 1 | 2 | 0.2816 | 0.3750 | NaN* | Fig_S3G.png |
| Multi-objective 2 | 15 | 0.7903 | 0.6490 | 0.6115 | Fig_S3H.png |
| Multi-objective 3 | 2 | 0.6386 | 0.6667 | NaN* | Fig_S3I.png |
| RNA-Seq | 2 | 0.0184 | 0.0087 | NaN* | - |
| CMU-PIE | 12 | 0.5739 | 0.1345 | 0.3339 | - |

*NaN values for AMI occurred when EMI was very small (< 2.22e-16); NMI was used instead.

## Hierarchical Clustering Details

For each dataset, the algorithm performed hierarchical clustering with these layer counts:

### Highly overlapping
- Layer 1: 1530 clusters
- Layer 2: 151 clusters
- Layer 3: 2 clusters
- Layer 4: 1 cluster

### FLAME
- Layer 1: 63 clusters
- Layer 2: 6 clusters
- Layer 3: 1 cluster

### Spectral-path
- Layer 1: 113 clusters
- Layer 2: 22 clusters
- Layer 3: 1 cluster

### Unbalanced
- Layer 1: 626 clusters
- Layer 2: 81 clusters
- Layer 3: 1 cluster

### Noisy
- Layer 1: 1217 clusters
- Layer 2: 150 clusters
- Layer 3: 1 cluster

### Heterogeneous geometric
- Layer 1: 126 clusters
- Layer 2: 16 clusters
- Layer 3: 1 cluster

### Multi-objective 1
- Layer 1: 342 clusters
- Layer 2: 57 clusters
- Layer 3: 3 clusters
- Layer 4: 1 cluster

### Multi-objective 2
- Layer 1: 332 clusters
- Layer 2: 59 clusters
- Layer 3: 2 clusters
- Layer 4: 1 cluster

### Multi-objective 3
- Layer 1: 164 clusters
- Layer 2: 20 clusters
- Layer 3: 2 clusters
- Layer 4: 1 cluster

### RNA-Seq
- Layer 1: 80 clusters
- Layer 2: 2 clusters
- Layer 3: 1 cluster

### CMU-PIE
- Layer 1: 799 clusters
- Layer 2: 92 clusters
- Layer 3: 2 clusters
- Layer 4: 1 cluster

## Errors

The following datasets encountered errors during processing:

1. **YTF dataset**: Cannot cast array data from dtype([('data', 'O'), ('datalabels', 'O')]) to dtype('float64') according to the rule 'unsafe'

2. **MNIST70k dataset**: Cannot cast array data from dtype([('data', 'O'), ('datalabels', 'O')]) to dtype('float64') according to the rule 'unsafe'

3. **Shuttle dataset**: Unable to allocate 50.1 GiB for an array with shape (3363957443, 2) and data type int64

4. **Haberman dataset**: data\haberman.txt not found

5. **Zoo dataset**: Cannot cast array data from dtype([('data', 'O'), ('datalabels', 'O')]) to dtype('float64') according to the rule 'unsafe'

6. **S.disease dataset**: Cannot cast array data from dtype([('data', 'O'), ('datalabels', 'O')]) to dtype('float64') according to the rule 'unsafe'

7. **Cell.track dataset**: Cannot cast array data from dtype([('data', 'O'), ('datalabels', 'O')]) to dtype('float64') according to the rule 'unsafe'

8. **CMU-PIE 11k dataset**: Cannot cast array data from dtype([('data', 'O'), ('datalabels', 'O')]) to dtype('float64') according to the rule 'unsafe'

9. **Reuters dataset**: Cannot cast array data from dtype([('data', 'O'), ('datalabels', 'O')]) to dtype('float64') according to the rule 'unsafe'

10. **COIL-100 dataset**: freezes and cannot proceed

Proper attribution to the original author is required when using, modifying, or redistributing this repository.
Any modifications, adaptations, or derivative works must also be non-commercial and remain under CC BY-NC-SA 4.0.



The code and Torque Clustering algorithm is not meant for commercial use. Please contact the author (jie.yang.uts@gmail.com) for licensing information.
