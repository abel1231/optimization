import numpy as np
# from qpsolvers import solve_qp
# import cvxpy as cp
import pandas as pd

def data_preprocessing(file_path):
    '''
    Return expected return and covariance matrix of the dataset
    '''
    ## excel --> array
    df = pd.read_excel(file_path)
    data = df.values
    ## closing price --> rate of return
    RoR = np.zeros([data.shape[0]-1,data.shape[1]])   # Rate of return
    for i in range(data.shape[0]-1):
        RoR[i,:] = (data[i+1,:]-data[i,:])/data[i,:]
    ## Calculate the expected return and covariance matrix
    data_df = pd.DataFrame(RoR)
    exp_ret = data_df.mean()
    cov_mat = data_df.cov()
    return exp_ret, cov_mat
    
if __name__=="__main__":
    # Preprocessing Step
    file_path = "./data/stock_data.xlsx"
    exp_ret, cov_mat = data_preprocessing(file_path)
    # Original optimization using qpsolvers
    risk = 5
    P = risk * cov_mat.values
    q = -1 * exp_ret.values
    G = np.vstack((np.diag(np.ones(len(exp_ret))),np.diag(-np.ones(len(exp_ret)))))
    h = np.array([1.*np.ones(len(exp_ret)),np.zeros(len(exp_ret))]).reshape(len(exp_ret)*2,)
    A = np.ones(len(exp_ret))
    b = b = np.array([1.]).reshape((1,))
    x= solve_qp(P, q, G, h, A=A, b=b)
    x = x.reshape(len(x),1)
    print('Original weight solution:\n', x.T )
    print('Original optimal value:\n',  np.dot(x.T,exp_ret.values.reshape(len(exp_ret),1)) - 2.5* np.dot(np.dot(x.T,cov_mat.values),x))
    # Integer programming using cvxpy
    n = 6
    G = 16
    w = cp.Variable(n,integer=True)
    gamma = 5/2
    ret = exp_ret.values@(w/G)
    var = gamma*cp.quad_form((w/G),cov_mat.values)
    prob2 = cp.Problem(cp.Maximize(ret-var),[cp.sum(w)==G, w>=0])
    prob2.solve()
    print('\nThe integer solution is \n',w.value,'\n',w.value/G)
    print("The optimal value is \n", prob2.value)
    # Paper formulation
    theta3 = 1/1800
    w2 = cp.Variable(n,integer=True)
    ret2 = exp_ret.values@(w2/G)
    var2 = gamma*cp.quad_form((w2/G),cov_mat.values)
    lag = cp.quad_form(w2-G,np.diag(np.ones(n)))
    prob3 = cp.Problem(cp.Maximize(ret2-var2-theta3*lag))
    prob3.solve()
    print('\nPaper formulation solution is\n',w2.value,'\n',w2.value/G)
    k = w2.value/16
    print("The optimal value is",)
    print(np.dot(k.T,exp_ret.values.reshape(len(exp_ret),1)) - 2.5* np.dot(np.dot(k.T,cov_mat.values),k))
    
    
    
    