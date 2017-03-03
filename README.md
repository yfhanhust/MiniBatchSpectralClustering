# README #

This repository contains codes to reproduce one of the experimental results in the paper:

[1] Yufei HAN and Maurizio Filippone. Mini-Batch Spectral Clustering, 2017, accepted by IJCNN 2017. A longer and detailed version can be found here: https://arxiv.org/abs/1607.02024

The code implements the proposed mini-batch spectral clustering algorithm and a Nystrom approximation based spectral embedding algorithm used as warm start of the proposed clustering method. The Nystrom approximation based spectral embedding method was proposed by:

[2] Charless Fowlkes, Serge Belongie, Fan Chung and Jitendra Malik, Spectral Grouping Using the Nystrom Method, IEEE Transactions on Pattern Analysis and Machine Learning, vol.26, no.2, 2004

The function implements the proposed mini-batch spectral clustering algorithm in Python is as illstrated:

StochasticRiemannianOpt(C,X,ndim,master_stepsize,auto_corr,outer_max_iter,nsamples,nrounds)

*       C:                   Symmetric Laplacian matrix generated from pairwise similarity measurements between training data points 
*       X:                   Training data matrix, with each row corresponding to one data point
*       master_stepsize:     Master Stepsize of AdaGrad 
*       auto_corr:           Auto correlation coefficient of AdaGrad
*       outer_max_iter:      The maximum number of iterations for AdaGrad based optimisation
*       nsamples:            The number of columns sampled from the Laplacian matrix C each time 
*       nrounds:             The number of sampling rounds

nsamples * nrounds equals to the size of the mini-batch used for stochastic gradient computation

The Nystrom approximation based baseline clustering algorithm is also implemented in Python, as you can find here: 

nystromSP(train_data,nsample,sigma,num_clusters)

*       train_data:          Training data matix, with each row corresponding to one data point 
*       nsample:             The number of sampled data points from the training data 
*       gamma_value:         Scailing parameter in RBF similarity measurement between data points 
*       num_clusters:        The expected number of clusters, just as K in K-means

## Example 

X_sto1,nnz_list,X_sto_list = StochasticRiemannianOpt(laplacian,X,ndim,master_stepsize,auto_corr,outer_iter,ncols,nsampleround)

*       X_sto_list:          List of spectral embeddings derived from the Laplacian matrix, generated per iteration. 

X = nystromSP(train_data,10,gamma_value,nclass)

*       X:                   The approximated spectral embedding derived from the Laplacian matrix 

## Note ##

The Nystrom approximation based spectral embedding algorithm is also used as a baseline in the comparative experiments of our paper 

## Shuttle data used in this example code ## 

You can download the shuttle data from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#shuttle

We use both training data shuttle.scale and the testing data shuttle.scale.t. We merge them into one single training data matrix for clustering. Due to limit of space, we don't share the data directly in this repo. 
