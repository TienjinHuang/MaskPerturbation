# Instructions

## The examples of running code can be found in RN18_MaskPerturb directory
## hypyerparameter rho denotes the magnitude for scaling perturbations on relaxed mask, i.e. mask scores.
## hypyerparameter prun_rate denotes the sparsity.
## The MaskRangeRate on the printed information denotes the ratio of perturbed binary mask, i.e. (Mask_perutrb U Mask_unperturb).sum()/Mask_unperturb.sum()
## The LossGap on the printed infomation denotes the changes of loss between unperturbed mask and perturbed mask.
## This version only includes layer-wise pruning.

# Experiments

## I am doing experiments on CIFAR-10 with RN-18 for exploring the effect of rho on the performence with varying sparsity.

## I also plan to test the effect of rho on the performence under global prunning setting. 

## if you have any idea for analysis, just do it as you think.  We can share the experimental/analysis setting and results in slack or here in time. 

## If you have no idea for analysis, you could also do some testing for global-wise prunning setting if you would like.
