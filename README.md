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

# Comprehensive Comparison: TorqueClustering Results vs. Dataset Statistics from paper

| Dataset | Instances | Dimensions | Expected Clusters | Found Clusters | NMI | AC | AMI | Status |
|---------|-----------|------------|-------------------|----------------|-----|----|----|--------|
| Highly overlapping | 5000 | 2 | 15 | 3 | 0.4924 | 0.2052 | 0.2028 | ✅ Processed |
| FLAME | 240 | 2 | 2 | 7 | 0.6068 | 0.4833 | 0.4198 | ✅ Processed |
| Spectral-path | 312 | 2 | 3 | 7 | 0.5272 | 0.4231 | 0.3551 | ✅ Processed |
| Unbalanced | 2000 | 2 | 3 | 2 | 0.7132 | 0.9000 | 0.0000 | ✅ Processed |
| Noisy | 4000 | 2 | 5 | 54 | - | - | - | ✅ Processed (no ground truth) |
| Heterogeneous geometric | 400 | 2 | 3 | 4 | 0.8935 | 0.8550 | 0.6297 | ✅ Processed |
| Multi-objective 1 | 1000 | 2 | 4 | 2 | 0.2816 | 0.3750 | NaN* | ✅ Processed |
| Multi-objective 2 | 1000 | 2 | 4 | 15 | 0.7903 | 0.6490 | 0.6115 | ✅ Processed |
| Multi-objective 3 | 1500 | 2 | 6 | 2 | 0.6386 | 0.6667 | NaN* | ✅ Processed |
| OFD-F100 | 100 | 10304 | 10 | - | - | - | - | ❌ Not in results |
| MNIST | 10000 | 4096 | 10 | - | - | - | - | ❌ Data type error (MNIST70k) |
| COIL-100 | 7200 | 49152 | 100 | - | - | - | - | ❌ Incomplete processing |
| Shuttle | 58000 | 9 | 7 | - | - | - | - | ❌ Memory allocation error |
| RNA-Seq | 801 | 20531 | 5 | 2 | 0.0184 | 0.0087 | NaN* | ✅ Processed |
| Haberman | 306 | 3 | 2 | - | - | - | - | ❌ File not found |
| Zoo | 101 | 16 | 7 | - | - | - | - | ❌ Data type error |
| S.disease | - | - | - | - | - | - | - | ❌ Data type error |
| Cell.track | - | - | - | - | - | - | - | ❌ Data type error |
| CMU-PIE | - | - | - | 12 | 0.5739 | 0.1345 | 0.3339 | ✅ Processed (not in provided stats) |
| CMU-PIE 11k | - | - | - | - | - | - | - | ❌ Data type error |
| Reuters | - | - | - | - | - | - | - | ❌ Data type error |
| YTF | - | - | - | - | - | - | - | ❌ Data type error |
| Atom | 800 | 3 | 2 | - | - | - | - | ❌ Not in results |

*NaN values for AMI occurred when EMI was very small (< 2.22e-16); NMI was used instead.

## Summary of Findings

### Successfully Processed Datasets (10/20)

- **Matching cluster count**: None of the processed datasets found the exact number of expected clusters
- **Close match**: Heterogeneous geometric (found 4 vs. expected 3), Unbalanced (found 2 vs. expected 3)
- **Largest discrepancies**: 
  - Highly overlapping: Found 3 vs. expected 15
  - Noisy: Found 54 vs. expected 5
  - Multi-objective 2: Found 15 vs. expected 4

### Failed Processing (10/20)

- **Data type casting errors**: 8 datasets (MNIST70k, Zoo, S.disease, Cell.track, CMU-PIE 11k, Reuters, YTF, COIL-100)
- **Memory allocation error**: 1 dataset (Shuttle)
- **File not found**: 1 dataset (Haberman)

### Performance Metrics

- **Best NMI**: Heterogeneous geometric (0.8935)
- **Best AC**: Unbalanced (0.9000)
- **Worst performer**: RNA-Seq (NMI: 0.0184, AC: 0.0087)

### Additional Datasets in Results

Your TorqueClustering run included several datasets not mentioned in the provided statistics:
- CMU-PIE
- S.disease
- Cell.track
- CMU-PIE 11k
- Reuters
- YTF

### Missing Datasets from  Results

Datasets in the provided statistics but not processed in your results:
- OFD-F100
- Atom

6. **S.disease dataset**: Cannot cast array data from dtype([('data', 'O'), ('datalabels', 'O')]) to dtype('float64') according to the rule 'unsafe'

7. **Cell.track dataset**: Cannot cast array data from dtype([('data', 'O'), ('datalabels', 'O')]) to dtype('float64') according to the rule 'unsafe'

8. **CMU-PIE 11k dataset**: Cannot cast array data from dtype([('data', 'O'), ('datalabels', 'O')]) to dtype('float64') according to the rule 'unsafe'

9. **Reuters dataset**: Cannot cast array data from dtype([('data', 'O'), ('datalabels', 'O')]) to dtype('float64') according to the rule 'unsafe'

10. **COIL-100 dataset**: freezes and cannot proceed

Proper attribution to the original author is required when using, modifying, or redistributing this repository.
Any modifications, adaptations, or derivative works must also be non-commercial and remain under CC BY-NC-SA 4.0.



The code and Torque Clustering algorithm is not meant for commercial use. Please contact the author (jie.yang.uts@gmail.com) for licensing information.
