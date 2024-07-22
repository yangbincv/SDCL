# Shallow-Deep Collaborative Learning for Unsupervised Visible-Infrared Person Re-Identification

The *official* repository for [Shallow-Deep Collaborative Learning for Unsupervised Visible-Infrared Person Re-Identification](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_Shallow-Deep_Collaborative_Learning_for_Unsupervised_Visible-Infrared_Person_Re-Identification_CVPR_2024_paper.pdf). We achieve state-of-the-art performances on **unsupervised visible-infrared person re-identification** task.


# Contributions

1. We propose a shallow-deep collaborative learning framework based on the transformer architecture. This framework facilitates the learning of robust representation, effectively countering the cross-modality discrepancy through the collaboration of shallow and deep features.
2. We propose a collaborative neighbor learning module to formulate dependable intra-modality and cross-modality neighbor learning, enabling the model to capture modality-invariant and discriminative features. 
3. We propose a collaborative ranking association module to explore intra-modality and cross-modality ranking consistencies, unifying the cross-modality labels and providing invaluable cross-modality supervision.
4. Extensive experiments validate that our SDCL framework surpasses existing methods on two mainstream VI-ReID benchmarks, consistently improving the unsupervised cross-modality retrieval performance.



# Prepare Datasets
Put SYSU-MM01 and RegDB dataset into data/sysu and data/regdb, run prepare\_sysu.py and prepare\_regdb.py to prepare the training data (convert to market1501 format).( See previous work [ADCA](https://github.com/yangbincv/ADCA) or [GUR](https://github.com/yangbincv/GUR). )

# Prepare Pre-trained model
We adopt the self-supervised pre-trained models (ViT-B/16+ICS) from [Self-Supervised Pre-Training for Transformer-Based Person Re-Identification](https://github.com/damo-cv/TransReID-SSL?tab=readme-ov-file).
Download link:https://drive.google.com/file/d/1ZFMCBZ-lNFMeBD5K8PtJYJfYEk5D9isd/view

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



