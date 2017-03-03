## Copyright 2017 Yufei HAN, Maurizio Filippone
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.


import numpy as np
import scipy as sp
from numpy.linalg import *
from scipy.linalg import *
from numpy.random import *

#### stochastic approximation to matrix product
def StochasticAverageMPFast(A,B,nelem,sampleIter,sample_mat):
    rowA = A.shape[0]
    colA = A.shape[1]
    rowB = B.shape[0]
    colB = B.shape[1]
    AB_prod = np.zeros((rowA,colB),dtype=float)
    ### generate sampling matrix
    #rvec = np.random.random_integers(0,high=(colA-1),size=(nelem,sampleIter))
    ### count unique elements
    #unique_rvec = np.unique(rvec)
    ### avoid repeated calculation
    rvec_list =[]
    #rvec_ind = list(range(colA))
    p = float(nelem) / float(colA)
    for iter in range(0,sampleIter):
        #np.random.shuffle(rvec_ind)
        rvec = sample_mat[:,iter].ravel().astype(int)
        #print rvec
        #rvec = rvec_ind[:nelem]
        rvec_list.extend(list(rvec))
        AB_prod += np.dot(A[:,list(rvec)],B[list(rvec),:]) * (1./p) * (1./float(sampleIter))
        #AB_prod += np.dot(A[:,rvec[:,iter]],B[rvec[:,iter],:]) * (1./p) * (1./float(sampleIter))

    return AB_prod,len(np.unique(rvec_list))
    
#### Riemannian Gradient Calculation using stochastic approximation 
def RiemannianGrad(C_off_diag,X,nsamples,nrounds,sample_mat):
    #### -(I-XX^T)CX
    #### Step.1 stochastic approximation to CX
    CX,nnz_len = StochasticAverageMPFast(C_off_diag,X,nsamples,nrounds,sample_mat)
    CX = CX - X
    #### Step.2 XCX
    XCX = np.dot(X.T,CX) ## do we need to use stochastic approximation to XCX
    #### Step.3 XXCX'
    XXCX = np.dot(X,XCX)
    G = CX - XXCX
    G = -1. * G
    return G,nnz_len

#### Spectral decomposition using Nystrom approximation
def nystromSP(train_data,nsample,sigma,num_clusters):
    num_clusters = np.unique(train_label).shape[0]
    data_ind = np.array(range(train_data.shape[0]))
    np.random.shuffle(data_ind)
    sampled_ind = data_ind[:nsample]
    other_ind = data_ind[nsample:]

    A = rbf_kernel(train_data[sampled_ind,:],train_data[sampled_ind,:],gamma=sigma)
    B = rbf_kernel(train_data[sampled_ind,:],train_data[other_ind,:],gamma=sigma)
    d1 = np.sum(A,axis=1) + np.sum(B,axis=1)
    d2 = np.sum(B.T,axis=1) + np.dot(B.T,np.dot(np.linalg.inv(A),np.sum(B,axis=1)))
    dhat = np.reshape(np.sqrt(1./np.concatenate([d1,d2])),[train_data.shape[0],1])
    A = np.multiply(A,np.dot(dhat[0:nsample],dhat[0:nsample].T))
    m = train_data.shape[0] - nsample
    B1 = np.dot(dhat[0:nsample,:],dhat[nsample:(nsample+m),:].T)
    B = np.multiply(B,B1)
    Asi = sp.linalg.sqrtm(np.linalg.inv(A))
    BBT  = np.dot(B,B.T)
    W = np.zeros((A.shape[0]+B.shape[1],A.shape[1]))
    W[0:A.shape[0],:] = A
    W[A.shape[0]:,:] = B.T
    R = A + np.dot(Asi,np.dot(BBT,Asi))
    R = (R + R.T)/2
    S,U = np.linalg.eigh(R) ### ascending order of eigenvalues
    S = np.diag(S[::-1])
    U = U[:,::-1]

    W = np.dot(W,Asi)
    V = np.dot(np.dot(W,U[:,:num_clusters]),np.linalg.inv(np.sqrt(S[:num_clusters,:][:,:num_clusters])))

    sq_sum = np.sqrt(np.sum(np.multiply(V,V),axis=1))+1e-20
    sq_sum_mask = np.zeros((len(sq_sum),num_clusters),dtype=float)
    for k in range(num_clusters):
        sq_sum_mask[:,k] = sq_sum

    Umat = np.divide(V,sq_sum_mask)
    X = np.zeros((Umat.shape[0],Umat.shape[1]))
    X[data_ind,:] = Umat
    return X
    
#### Mini-batch based spectral decomposition using Stochastic Gradient on Manifold    
def StochasticRiemannianOpt(C,X,ndim,master_stepsize,auto_corr,outer_max_iter,nsamples,nrounds):
    ### AdaGrad
    ### obj func X = argmin_{X} (0.5) * X.T L X, s.t. X.TX = I
    ### Semi-sotchastic gradient descent to control variance reduction
    fudge_factor = 1e-6
    historical_grad = 0.
    k = ndim
    n = C.shape[0]
    ### orthonormalisation
    X = Orthonormalisation(X)
    ### extract diagonal part
    #C_diag = np.diag(np.diag(C))
    #C_off_diag = C - C_diag
    for k in range(C.shape[0]):
        C[k,k] = 0.

    nnz_list = [] ## recording how many no-zero entries sampled each iteration
    X_sto_list = []
    
    print 'generate sampling templates'
    sample_mat_list = []
    rvec_idx = list(range(C.shape[0]))
    for itr in range(0,outer_max_iter):
        rvec_mat = np.zeros((nsamples,nrounds),dtype=float)
        for k in range(nrounds):
            np.random.shuffle(rvec_idx)
            rvec = rvec_idx[:nsamples]
            rvec_mat[:,k] = rvec

        sample_mat_list.append(rvec_mat)
    #print rvec_mat

    print 'the iteration begins '
    for itr in range(0,outer_max_iter):
        ### stochastic gradient
        stoG,nnz = RiemannianGrad(C,X,nsamples,nrounds,sample_mat_list[itr])
        nnz_list.append(nnz)
        historical_grad += auto_corr * historical_grad + (1.-auto_corr) * np.power(stoG,2)
        adjusted_grd = stoG / (fudge_factor + np.sqrt(historical_grad))
        X = X - master_stepsize * adjusted_grd
        X = Orthonormalisation(X)
        X_sto_list.append(X)
        print 'iteration: ' + str(itr)

    return X,nnz_list,X_sto_list

#### objective function of the trace optimisation problem 
def objeval(C,X):
    fvalue = 0.
    fvalue = -0.5*np.trace(np.dot(np.dot(X.T,C),X))
    return fvalue

#### Exact Gradient 
def ExactRiemannianGrad(C,X):
    CX = np.dot(C,X)
    XCX = np.dot(X.T,CX)
    XXCX = np.dot(X,XCX)
    G = CX - XXCX
    G = -1. * G ### -CX + XXCX
    return G


def gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X,mode='reduced')
    return Q

def Orthonormalisation(X):
    #### orthonormalisation
    if X.shape[1] == 1:
       target_norm = np.linalg.norm(X)
       X_hat = X / (target_norm + 1e-6)
    else:
        X_hat = gram_schmidt_columns(X)

    return X_hat

### Soving spectral decomposition using the exact gradient descent 
def ExactRiemannianOpt(C,master_stepsize,inner_max_iter,dim):
    ### AdaGrad
    fudge_factor = 1e-6
    historical_grad = 0.
    #auto_corr = 1.0
    k = dim
    n = C.shape[0]
    X = np.random.randn(n,k)
    ### orthonormalisation
    X = Orthonormalisation(X)
    #inner_max_iter = 30
    objval = []
    X_list = []
    for itr in range(0,inner_max_iter):
        G = ExactRiemannianGrad(C,X) ### stochastic gradient on Riemannian manifold
        historical_grad += np.power(np.linalg.norm(G),2)
        adjusted_grd = G / (fudge_factor + np.sqrt(historical_grad))
        X = X - master_stepsize * adjusted_grd
        #### orthonormalisation
        X = Orthonormalisation(X)
        X_list.append(X)
        if itr % 10 == 0:
           objval.append(objeval(C,X)) ### -1/2 * tr(X.TCX)
        #grad.append(G)

    return X,objval,X_list

#### Baseline: Power iteration based spectral decomposition
def PowerMethod(normalized_W,nclass,p):
    S = np.random.randn(normalized_W.shape[0],nclass)
    B = np.dot(normalized_W,S) ### dot

    for i in range(p):
        B = np.dot(normalized_W,B)
        B = np.dot(normalized_W,B)

    ### svd
    leftU,s,rightU = svd(B)
    return leftU[:,:nclass]
    
    
train_data_file = open("shuttle.scale",'r')
train_data_strings = train_data_file.readlines(300000000000)
train_data_file.close()

train_data = []
train_label = []
for i in range(len(train_data_strings)):
    tmp = train_data_strings[i].split(' ')
    tmpfeature = np.zeros((9,),dtype=float)
    for k in range(1,len(tmp)):
        if tmp[k] == '\n':
            continue

        single_line = tmp[k].split(':')
        feaidx = int(single_line[0])
        feaval = float(single_line[1])
        tmpfeature[feaidx-1] = feaval

    train_data.append(tmpfeature)
    train_label.append(float(tmp[0]))

train_data = np.array(train_data)
train_label = np.array(train_label)

test_data_file = open("shuttle.scale.t",'r')
test_data_strings = test_data_file.readlines(300000000000)
test_data_file.close()

test_data = []
test_label = []
for i in range(len(test_data_strings)):
    tmp = test_data_strings[i].split(' ')
    tmpfeature = np.zeros((9,),dtype=float)
    for k in range(1,len(tmp)):
        if tmp[k] == '\n':
            continue

        single_line = tmp[k].split(':')
        feaidx = int(single_line[0])
        feaval = float(single_line[1])
        tmpfeature[feaidx-1] = feaval

    test_data.append(tmpfeature)
    test_label.append(float(tmp[0]))

test_data = np.array(test_data)
test_label = np.array(test_label)

train_data = np.concatenate((train_data,test_data))
train_label = np.concatenate((train_label,test_label))

print 'nsample: ' + str(train_data.shape[0])
print 'nclass: ' + str(np.unique(train_label).shape[0])

gamma_value = 5.0
affinity_matrix = rbf_kernel(train_data,gamma = gamma_value)
from sklearn.utils.graph import graph_laplacian
from sklearn.utils.extmath import _deterministic_vector_sign_flip
### calculate laplacian matrix
laplacian,dd = graph_laplacian(affinity_matrix,normed=True,return_diag=True)
laplacian *= -1.

nclass = np.unique(train_label).shape[0]
nsample = train_data.shape[0]

#### Configuring AdaGrad
print 'mini batch size = 100'
master_stepsize = 0.0025
outer_iter = 600
nsampleround = 50
ncols = 2
auto_corr = 0.0
ndim = nclass
print 'nsampleround: ' + str(nsampleround)
print 'ncols: ' + str(ncols)
print 'master_stepsize: ' + str(master_stepsize)

nmi_sgd_set = []
num_repeat_exp = 10
for repeatExp in range(num_repeat_exp):
    print 'iteration id: ' + str(repeatExp)
    X = nystromSP(train_data,10,gamma_value,nclass)
    X_sto1,nnz_list,X_sto_list = StochasticRiemannianOpt(laplacian,X,ndim,master_stepsize,auto_corr,outer_iter,ncols,nsampleround)
    nmi_sgd = []
    for i in range(len(X_sto_list)):
        if i % 5 ==0:
           X_sto_tmp = X_sto_list[i].T * dd
           X_sto_tmp = _deterministic_vector_sign_flip(X_sto_tmp)
           cluster_id = KMeans(n_clusters = nclass, n_init = 50).fit(X_sto_tmp.T).labels_
           nmi = normalized_mutual_info_score(train_label,cluster_id) ### measuring NMI score per iteration
           nmi_sgd.append(nmi)

    nmi_sgd_set.append(nmi_sgd)

nmi_sgd_set=np.array(nmi_sgd_set)
nrow = nmi_sgd_set.shape[0]
ncol = nmi_sgd_set.shape[1]
records_file = open('sgd_cost_nmi_2_50_60k_ada_warmstart.csv','w')
for i in range(ncol):
    tmpstr = ''
    for j in range(nrow):
        tmpstr = tmpstr + str(nmi_sgd_set[j][i]) + ' '
    tmpstr += '\n'
    records_file.write(tmpstr)
