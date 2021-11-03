import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

#Reading the file 
ratings = pd.read_csv('ratings.dat', engine='python', sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'])
RT=ratings
#splitting the dataset into training and test sets
KF = KFold(n_splits=5, shuffle=True)
c = 2

i = 5
#Kfold
for train_index, test_index in KF.split(RT):
    RT_train, RT_test = RT.loc[train_index], RT.loc[test_index]
    Row_df = RT_train.pivot(index = 'user_id', columns ='movie_id', values = 'rating')
    u_mean = Row_df.mean(axis=1)
    Row_df_array = Row_df.to_numpy()
    u_mean = u_mean.to_numpy()
    normal = Row_df_array - u_mean.reshape(-1,1)
    N = normal
    u = np.full((normal.shape[0],2), 1)
    v = np.full((2,normal.shape[1]), 1)
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    uv = np.dot(u,v)
    print("TRAIN:", train_index, "TEST:", test_index)
 

#for updating u and v ---- if you arent using kfold  do not use the Kfold for loop
    for iterations in range(i):
        for r in range(6040):
  #update u
            for s in range(c):
                sums = 0
                u_rk = u[r,:]
                v_kj = v[:,:]
                u_rk_del = np.delete(u_rk, s, 0)
                v_kj_del = np.delete(v_kj, s, 0)
                v_sj = v[s,:]
                v_sj_squared = v_sj ** 2

                u_rk_v_kj = np.dot(u_rk_del, v_kj_del)
                m_rj = N[r,:]

                error = m_rj - u_rk_v_kj

                vsj_dot_er = v_sj * error
                sums = np.nansum(vsj_dot_er)
                v_sj_ssum = np.nansum((v_sj_squared) * (~np.isnan(m_rj)))
                newval_u = sums / v_sj_ssum
                u[r,s] = u[r,s] + ((newval_u - u[r,s]))
        #update v 
        for r in range(c):
            for s in range(Row_df_array.shape[1]):
                sums = 0
             

                u_ik = u[:,:]
                v_ks = v[:,s]
                u_ik_del = np.delete(u_ik, r, 1)

                v_ks_del = np.delete(v_ks, r, 0)
                u_ir = u[:,r]
                u_ir_squared = u_ir ** 2

                u_ik_v_ks = np.dot(u_ik_del, v_ks_del)
                m_is = N[:,s]
                error = m_is - u_ik_v_ks

                uir_dot_er = u_ir * error
                sumsv = np.nansum(uir_dot_er)
                u_ir_ssum = np.nansum(u_ir_squared * (~np.isnan(m_is)))
                newval_v = sumsv / u_ir_ssum
                v[r,s] = v[r,s] + ((newval_v - v[r,s]))

        uv = np.dot(u,v)
        dif = uv -normal
        print("Iteration Number: ",iterations )
      
      #for mean absoulte error
        dif_abs= (np.absolute(dif))
        dif_abs_0s = np.nan_to_num(dif_abs)
        dif_abs_sum = np.sum(dif_abs_0s,axis=0)
        sum_dif = dif_abs_sum.sum()
        non_0_count = np.count_nonzero(dif_abs_0s)
        MAE=sum_dif/non_0_count
        print('MAE',MAE)
  #for Root mean square error
        dif_sqr = dif ** 2
        dif_sqr_0s = np.nan_to_num(dif_sqr)
        dif_sqr_total= np.sum( dif_sqr_0s ,axis=0)
        sumz = dif_sqr_total.sum()
        non_0_count_sqr = np.count_nonzero( dif_sqr_0s )
        RME = sumz/ non_0_count_sqr
        rme_list=[RME]
        print('RMSE=',RME)
        

        
