## [Adjustment and Alignment for Unbiased Open Set Domain Adaptation (CVPR-23)](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Adjustment_and_Alignment_for_Unbiased_Open_Set_Domain_Adaptation_CVPR_2023_paper.html)

[[Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Adjustment_and_Alignment_for_Unbiased_Open_Set_Domain_Adaptation_CVPR_2023_paper.html)] [[Video Presentation](https://www.youtube.com/watch?v=hFZn16ntyXw)]

By [Wuyang Li](https://wymancv.github.io/wuyang.github.io/)

## Quick Summary

The main idea comes from open-set object detection, where the novel objects are hidden in the background. In OSDA, we do not separate objects from the background since both are out-of-base-class distributions and can be treated as unknown.

- Even though the source domain only contains the base-class image, we discover novel-class regions hidden in the image to generate unknown signals. This enables unbiased learning in the source domain.
- With a fine-grained perspective, each image can be treated as the base-class and novel-class regions, regardless of the image-level label. Hence, we align the base and novel class distribution, enabling an unbiased domain transfer.
- We use the causal theory to guide the method design.

## Experimental Environment
- cudatoolkit == 10.0
- torch == 1.6
- torchvision == 0.7.0
- numpy == 1.21.4
- scikit-learn == 1.0.2


## Get Start
- Download the Officehome dataset.
- Change the data root in train.py: --data-root
- Run run.sh for all sub-tasks.
- Generate final results in latex format.

Reproduced resuts by us:
|  A $\rightarrow$ C   | A $\rightarrow$ P   |A $\rightarrow$ R    | C $\rightarrow$ A  |C $\rightarrow$ P   | C $\rightarrow$ R |P $\rightarrow$ A   | P $\rightarrow$ C  |P $\rightarrow$ R   | R $\rightarrow$ A  |R $\rightarrow$ C   | R $\rightarrow$ P  |Avg  |
|  ----  | ----  | ----  | ----  | ----  | ----  | ----  | ----  | ----  | ----  | ----  | ----  |----  |
| 69.3  | 73.2 | 76.3 | 64.7 |68.6  | 72.7 |65.9  | 63.9 |76.0  | 70.6 |68.1  | 78.7 | 70.7|


- The performance of each sub-task is slightly different from the paper due to different seeds, environments, and warm-up iterations. 

- The average performance is the same.

## Limitations and Disussions
- The hyperparameters, e.g., top-K and gradient scalers, are relatively sensitive to the dataset properties (especially for tiny datasets with homogeneous scenes) and warm-up stages.
- We tried to avoid introducing extra parameters in the inference stage, which is sub-optimal. Using a different classification head and introduce other designs for the unknown prediction will be better.
- The idea of dicovering unknown components in a base-class image can be transferred to other tasks.

## Contact

If you have any questions or ideas you would like to discuss with me, feel free to let me know through wuyangli2-c @ my.cityu.edu.hk. Except for the main experiment on Officehome, other tiny-scaled benchmark settings will be released later if needed.

## Citation

If this work is helpful for your project, please give it a star and citation. Thanks~

```BibTeX
@InProceedings{Li_2023_CVPR,
    author    = {Li, Wuyang and Liu, Jie and Han, Bo and Yuan, Yixuan},
    title     = {Adjustment and Alignment for Unbiased Open Set Domain Adaptation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {24110-24119}
}
```

## Abstract 

Open Set Domain Adaptation (OSDA) transfers the model from a label-rich domain to a label-free one containing novel-class samples. Existing OSDA works overlook abundant novel-class semantics hidden in the source domain, leading to a biased model learning and transfer. Although the causality has been studied to remove the semantic-level bias, the non-available novel-class samples result in the failure of existing causal solutions in OSDA. To break through this barrier, we propose a novel causalitydriven solution with the unexplored front-door adjustment theory, and then implement it with a theoretically grounded framework, coined Adjustment and Alignment (ANNA), to achieve an unbiased OSDA. In a nutshell, ANNA consists of Front-Door Adjustment (FDA) to correct the biased learning in the source domain and Decoupled Causal Alignment (DCA) to transfer the model unbiasedly. On the one hand, FDA delves into fine-grained visual blocks to discover novel-class regions hidden in the base-class image. Then, it corrects the biased model optimization by implementing causal debiasing. On the other hand, DCA disentangles the base-class and novel-class regions with orthogonal masks, and then adapts the decoupled distribution for an unbiased model transfer. Extensive experiments show that ANNA achieves state-of-the-art results. 

![image](./assets/mot.png)
