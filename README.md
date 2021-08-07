# MN-PCA
MN-PCA (matrix normal principal component analysis) is a powerful and intuitive PCA method through modeling the graphical noise by the matrix normal distribution. MN-PCA obtains a **low-rank representation** of data and the **structure of the correlated noise** simultaneously. 
- `illustrative_example.m` is an illustractive example to show the projections of PCA and MN-PCA on the synthetic dataset.
- `./MN-PCA-MRL` contains the main function of the maximizing regularized likelihood algorithm of `MnPCA.m`
- `./MN-PCA-w2` contains the matlab wrapper function `MnPCAq1_wrapper.m` for the minizing Wasserstein distance algorithm. The algorithm is written with PyTorch (`MnPCAq1.py`). 

## Citation

```
@article{Zhang2019,
  title={Matrix normal PCA for interpretable dimension reduction and graphical noise modeling},
  author={Zhang, Chihao and Gai, Kuo and Zhang, Shihua},
  journal={arXiv preprint arXiv:1911.10796},
  year={2019}
}
```

