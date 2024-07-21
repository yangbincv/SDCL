# Shallow-Deep Collaborative Learning for Unsupervised Visible-Infrared Person Re-Identification

The *official* repository for [Towards Grand Unified Representation Learning for Unsupervised Visible-Infrared Person Re-Identification](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_Shallow-Deep_Collaborative_Learning_for_Unsupervised_Visible-Infrared_Person_Re-Identification_CVPR_2024_paper.pdf). We achieve state-of-the-art performances on **unsupervised visible-infrared person re-identification** task.

**Our unified framework**
![framework](figs/overview.pdf)

# Highlight

1. We propose a novel unsupervised learning framework that adopts a bottom-up domain learning strategy with cross-memory association embedding. This enables the model to learn unified representation which is robust against hierarchical discrepancy.
2. We design a cross-modality label unification module to propagate and smooth labels between two modalities with heterogeneous affinity matrix and homogeneous structure matrix, respectively, unifying the identities across the two modalities.
3. Extensive experiments on the SYSU-MM01 and RegDB datasets demonstrate that our GUR framework significantly outperforms existing USL-VI-ReID methods, and even surpasses some supervised counterparts, further narrowing the gap between supervised and unsupervised VI-ReID. 

# Prepare Datasets
Put SYSU-MM01 and RegDB dataset into data/sysu and data/regdb, run prepare\_sysu.py and prepare\_regdb.py to prepare the training data (convert to market1501 format).

# Training

We utilize 2 A100 GPUs for training.

**examples:**

SYSU-MM01:

1. Train:
```shell
sh train_cc_vit_sysu.sh
```


2. Test:
```shell
sh test_cc_vit_sysu.sh
```

RegDB:

1. Train:
:
```shell
sh train_cc_vit_regdb.sh
```

2. Test:
```shell
sh test_cc_vit_regdb.sh
```


# Citation
This code is based on previous work [ADCA](https://github.com/yangbincv/ADCA.). 
If you find this code useful for your research, please cite our papers.

```
@inproceedings{yang2024shallow,
  title={Shallow-Deep Collaborative Learning for Unsupervised Visible-Infrared Person Re-Identification},
  author={Yang, Bin and Chen, Jun and Ye, Mang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16870--16879},
  year={2024}
}


@article{yang2023dual,
  title={Dual Consistency-Constrained Learning for Unsupervised Visible-Infrared Person Re-Identification},
  author={Yang, Bin and Chen, Jun and Chen, Cuiqun and Ye, Mang},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2023},
  publisher={IEEE}
}


@InProceedings{Yang_2023_ICCV,
    author    = {Yang, Bin and Chen, Jun and Ye, Mang},
    title     = {Towards Grand Unified Representation Learning for Unsupervised Visible-Infrared Person Re-Identification},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {11069-11079}
}

@inproceedings{adca,
  title={Augmented Dual-Contrastive Aggregation Learning for Unsupervised Visible-Infrared Person Re-Identification},
  author={Yang, Bin and Ye, Mang and Chen, Jun and Wu, Zesen},
  pages = {2843–2851},
  booktitle = {ACM MM},
  year={2022}
}

@article{yang2023translation,
  title={Translation, association and augmentation: Learning cross-modality re-identification from single-modality annotation},
  author={Yang, Bin and Chen, Jun and Ma, Xianzheng and Ye, Mang},
  journal={IEEE Transactions on Image Processing},
  year={2023},
  publisher={IEEE}
}

```

# Contact
yangbin_cv@whu.edu.cn; yemang@whu.edu.cn.



