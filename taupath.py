import numpy as np
from numpy import random
from random import sample 
import scipy
from sklearn.metrics import pairwise_distances
from scipy.stats import rankdata
from scipy.stats import kendalltau
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import time
from itertools import combinations 

def Diff_distance_fun(x1,x2):
    diff_dist = sum(x1-x2)
    return diff_dist

def General_corr_fun(x,y,method):
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    if method == "Kendall":
        xx_mat = np.sign(pairwise_distances(x,x.copy(),metric=Diff_distance_fun))
        yy_mat = np.sign(pairwise_distances(y,y.copy(),metric=Diff_distance_fun))
    elif method == "Spearman":
        rank_x = rankdata(x).reshape(-1,1)
        rank_y = rankdata(y).reshape(-1,1)
        xx_mat = pairwise_distances(rank_x,rank_x.copy(),metric=Diff_distance_fun)
        yy_mat = pairwise_distances(rank_y,rank_y.copy(),metric=Diff_distance_fun)
    elif method == "Pearson":
        xx_mat = pairwise_distances(x,x.copy(),metric=Diff_distance_fun)
        yy_mat = pairwise_distances(y,y.copy(),metric=Diff_distance_fun)
    num = np.sum(xx_mat*yy_mat)
    denom1 = np.sum(xx_mat*xx_mat)
    denom2 = np.sum(yy_mat*yy_mat)
    corr = num/np.sqrt(denom1*denom2)
    return corr,num,denom1,denom2

def General_corr_Update_fun(x,y,results,xnew,ynew,method):
    if method == "Kendall":
        denom1_new = results[2]+2*np.sum(np.sign(xnew-x)*np.sign(xnew-x))
        denom2_new = results[3]+2*np.sum(np.sign(ynew-y)*np.sign(ynew-y))
        num_new = results[1]+2*np.sum(np.sign(xnew-x)*np.sign(ynew-y))
    elif method == "Pearson":
        denom1_new = results[2]+2*np.sum((xnew-x)**2)
        denom2_new = results[3]+2*np.sum((ynew-y)**2)
        num_new = results[1]+2*np.sum((xnew-x)*(ynew-y))
    elif method == "Spearman":
        rank_x = rankdata(x)
        rank_y = rankdata(y)
        
        info_x = (x>xnew)+0
        rank_x_now = rank_x+info_x
        rank_xnew = np.array([np.sum(1-info_x)+1])
        rank_x_updated = np.concatenate([rank_x_now,rank_xnew])
        
        info_y = (y>ynew)+0
        rank_y_now = rank_y+info_y
        rank_ynew = np.array([np.sum(1-info_y)+1])
        rank_y_updated = np.concatenate([rank_y_now,rank_ynew])
        
        cov = np.cov(rank_x_updated,rank_y_updated)
        denom1_new = cov[0,0]
        denom2_new = cov[1,1]
        num_new = np.cov(rank_x_updated,rank_y_updated)[0,1]
    
    corr_new = num_new/np.sqrt(denom1_new*denom2_new)
        
    return corr_new,num_new,denom1_new,denom2_new


def General_corr_Delete_fun(x,y,results,xdel,ydel,method):
    if method == "Kendall":
        denom1_new = results[2]-2*np.sum(np.sign(xdel-x)*np.sign(xdel-x))
        denom2_new = results[3]-2*np.sum(np.sign(ydel-y)*np.sign(ydel-y))
        num_new = results[1]-2*np.sum(np.sign(xdel-x)*np.sign(ydel-y))
    elif method == "Pearson":
        denom1_new = results[2]-2*np.sum((xdel-x)**2)
        denom2_new = results[3]-2*np.sum((ydel-y)**2)
        num_new = results[1]-2*np.sum((xdel-x)*(ydel-y))
    elif method == "Spearman":
        rank_x = rankdata(x)
        rank_y = rankdata(y)
        
        info_x = (x>xdel)+0
        rank_x_now = rank_x-info_x
        idx = np.where(x==xdel)
        rank_x_updated = np.delete(rank_x_now,idx)
        
        
        info_y = (y>ydel)+0
        rank_y_now = rank_y-info_y
        idy = np.where(y==ydel)
        rank_y_updated = np.delete(rank_y_now,idy)
        
        cov = np.cov(rank_x_updated,rank_y_updated)
        denom1_new = cov[0,0]
        denom2_new = cov[1,1]
        num_new = np.cov(rank_x_updated,rank_y_updated)[0,1]
    
    corr_new = num_new/np.sqrt(denom1_new*denom2_new)
        
    return corr_new,num_new,denom1_new,denom2_new
        
    

def gen_corr(x,y,method):
    if method == "Kendall":
        corr, _ = kendalltau(x, y)
    elif method == "Spearman":
        corr,_ = spearmanr(x, y)
    elif method == "Pearson":
        corr, _ = pearsonr(x, y)
    return corr    


def compute_score(x,method="Kendall"):
    p = x.shape[1]
    score = 0
    for i in range(p):
        for j in range(i+1,p):
            score += (2/(p*(p-1)))*abs(gen_corr(x[:,i],x[:,j],method))
    return score


class CorrState(object):
    
    def __init__(self, score, results):
        
        self.results = results   
        self.score = score
        
        
        
def compute_score_initial(X,ind,method):
    x = X[ind]
    corr=[]
    num=[]
    denom1=[]
    denom2=[]
    it = 0
    score = 0
    p = X.shape[1]
    for i in range(p):
        for j in range(i+1,p):
            corr0, num0, denom10,denom20 = General_corr_fun(x[:,i],x[:,j],method='Kendall')
            corr.append(corr0)
            score += abs(corr0)
            num.append(num0)
            denom1.append(denom10)
            denom2.append(denom20)
    score = (2./(p*(p-1)))*score
    stat = CorrState(score,list(zip(corr,num,denom1,denom2)))
    return stat




def compute_score_add(X,ind_old,ind_new,state_old,method):
    xnew = X[ind_new]
    x = X[ind_old]
    it = 0
    score = 0
    corr=[]
    num=[]
    denom1=[]
    denom2=[]
    p = X.shape[1]
    for i in range(p):
        for j in range(i+1,p):
            results = state_old.results[it]
            it += 1
            corr0, num0, denom10,denom20 = General_corr_Update_fun(
                x[:,i],x[:,j],results,xnew[i],xnew[j],method='Kendall')
            score += abs(corr0)
            corr.append(corr0)
            num.append(num0)
            denom1.append(denom10)
            denom2.append(denom20)
    score = (2./(p*(p-1)))*score
    stat = CorrState(score,list(zip(corr,num,denom1,denom2)))
    return stat


def compute_score_del(X,ind_old,ind_del,state_old,method):
    xdel = X[ind_del]
    x = X[ind_old]
    it = 0
    score = 0
    corr=[]
    num=[]
    denom1=[]
    denom2=[]
    p = X.shape[1]
    for i in range(p):
        for j in range(i+1,p):
            results = state_old.results[it]
            it += 1
            corr0, num0, denom10,denom20 = General_corr_Delete_fun(
                x[:,i],x[:,j],results,xdel[i],xdel[j],method='Kendall')
            score += abs(corr0)
            corr.append(corr0)
            num.append(num0)
            denom1.append(denom10)
            denom2.append(denom20)
    score = (2./(p*(p-1)))*score
    stat = CorrState(score,list(zip(corr,num,denom1,denom2)))
    return stat




def compute_score_with_state(X,ind_old,ind,method,past_score):
    # ind = ind_old+[ind_new]
    idx = ','.join([str(ii) for ii in ind])
    idx_old = ','.join([str(ii) for ii in ind_old]) #dictionary index
    
    if past_score.get(idx) is None:
        if len(ind_old)>len(ind):
            ind_del = list(set(ind_old)-set(ind))[0]
            if idx_old not in past_score:
                past_score_bk = past_score
            state_old = past_score[idx_old]
            state_new = compute_score_del(X,ind_old,ind_del,state_old,method)
        else:
            ind_new = list(set(ind)-set(ind_old))[0]
            state_old = past_score[idx_old]
            state_new = compute_score_add(X,ind_old,ind_new,state_old,method)
        past_score[idx] = state_new
        return past_score[idx]
    else:
        return past_score[idx].score




# mutate function
def mutate(ind, row, low, high, total = 2):
    '''
    mutate function: insert or delete sample rows from candidate ind
    
    inputs:
        ind   vector   indices for initial rows
        row   int      total number of samples
        low   int      minimum size for candidate set
        high  int      maximum size for sandidate set 
        
    output:
        list of mutated sequences
    '''
    l = len(ind)
    new_cand = list()
    
    count = 0
    while count<total:
        
        if random.randn()<0:
            # deletion 
            if l>low:
                #new_cand.append(ind[random.choice(l,l-1)])
                new_cand.append([ind[ii] for ii in random.choice(l,l-1,replace=False)])
                count += 1
        else:
            # insertion
            if l<high:
                valid_ind = list(set(range(row))-set(ind))
                chosen = valid_ind[int(random.choice(len(valid_ind),1))]
                new_cand.append(ind+[chosen])
                count += 1
                
    return new_cand


def selection(cand_score,top,kp):
    '''
    selection function: keep the best kp candidate scores, for the rest keep with prob 
    
    inputs:
        cand_score array    all candidate scores
        top        int      the first top should be kept
        kp         int      total number of candidate to keep
        prob       float    keep less competitive candidates with prob 
    '''
    sorted_ind = np.argsort(cand_score)[::-1]
    keep = np.zeros([len(cand_score),],dtype=np.bool)
    keep[:top] = True
    l = len(cand_score)-top
    r = kp - top
    for ii in random.choice(l,r,replace=False):
        keep[top+ii] = True
    return sorted_ind[keep]   




def sub_pop_state(X,cand_size,low,high,top,kp,rep,method="Kendall"):
    past_score = dict()
    row,column = X.shape
    n = random.randint(low, high, size=cand_size)
    # initialize candidate 
    cand =  [list(random.choice(row, ni, replace=False)) for ni in n]
    cand = [sorted(can) for can in cand]
    cand = [list(x) for x in set(tuple(x) for x in cand)]
    # initialize
    for ind in cand:
        ind = sorted(ind)
        idx = ','.join([str(ii) for ii in ind])
        past_score[idx] = compute_score_initial(X,ind,method)
    # Replication
    for t in range(rep):
        # mutation
        new = [mutate(ind, row, low, high, total = 2) for ind in cand]
        
        for i,new_i in enumerate(new):
            ind = sorted(cand[i])
            cand += new_i
            for ind_new in new_i:
                ind_new = sorted(ind_new)
                state = compute_score_with_state(X,ind,ind_new,method,past_score)

        cand = [sorted(can) for can in cand]
        cand = [list(x) for x in set(tuple(x) for x in cand)] 
        score = [past_score[','.join([str(ii) for ii in ind])].score for ind in cand]


        # selection
        ind = selection(score,top,kp)
        cand = [cand[idx] for idx in ind]
        score = [score[idx] for idx in ind]
        out = np.array(cand)[np.where([sc == score[0] for sc in score])[0]]
        #print(t,score[0],cand[0])
        out_sort = [sorted(can) for can in out]
        unique_out = [list(x) for x in set(tuple(x) for x in out_sort)]
    #return unique_out,score[0]
    print("GA_Subpopulation:",unique_out,"Best Score:",score[0])
    return unique_out,score[0]


def compute_step_add(X, ind, state, states, method):
    row,col = X.shape
    to_add = list(set(range(row)) - set(ind))
    for add in to_add:
        new_ind = sorted(ind+[add])
        idx = ','.join([str(ii) for ii in new_ind])
        if idx in states:
            continue
        states[idx] = compute_score_add(X,ind,add,state,method)
        

def compute_step_del(X, ind, state, states, method):
    row,col = X.shape
    to_del = list(ind)
    for d in to_del:
        new_ind = ind.copy()
        new_ind.remove(d)
        idx = ','.join([str(ii) for ii in new_ind])
        if idx in states:
            continue
        states[idx] = compute_score_del(X,ind,d,state,method)

def forward_select(X,low,method):
    row,col = X.shape
    for step in range(3,low+1):
        states = dict()
        if step == 3:
            comb = combinations(range(row), step)
            for ind in comb:
                ind = sorted(ind)
                idx = ','.join([str(ii) for ii in ind])
                states[idx] = compute_score_initial(X,ind,method)
        else:
            for idx,state in cand.items():
                ind = [int(i) for i in idx.split(',')]
                compute_step_add (X, ind, state, states, method)
        score = [v.score for k,v in states.items()]
        best_score = max(score)   
        cand = dict()
        for k,v in states.items():
            if v.score==best_score: cand[k] = v
    for k,v in cand.items():
        print("Forward_Subpopulation:",k,"Best Score:",v.score)
    return cand         

def backward_select(X,low,method):
    row,col = X.shape
    h = row
    for step in range(h,low-1,-1):
        states = dict()
        if step == h:
            ind = list(range(h))
            idx = ','.join([str(ii) for ii in ind])
            states[idx] = compute_score_initial(X,ind,method)
        else:
            for idx,state in cand.items():
                ind = [int(i) for i in idx.split(',')]
                compute_step_del (X, ind, state, states, method)
        score = [v.score for k,v in states.items()]
        best_score = max(score)   
        cand = dict()
        for k,v in states.items():
            if v.score==best_score: cand[k] = v
        if step < low: break   
    for k,v in cand.items():
        print("Backward_Subpopulation:",k,"Best Score:",v.score)
    return cand