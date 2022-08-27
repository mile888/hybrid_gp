#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from sklearn.linear_model import LinearRegression


# In[3]:


def linear_model_IEEE9(dc_dataset, num_samples=180):
    X_dc_train = dc_dataset.loc[:num_samples, ['Pg0', 'Pg1', 'Pg2', 'Load_P1', 'Load_P2', 'Load_P3', 'RS_P1', 'RS_P2']]
    Y_dc_train = dc_dataset.loc[:num_samples, ['P_34', 'P_45', 'P_56', 'P_67', 'P_78', 'P_83']]
    
    lr = LinearRegression()
    lr.fit(X_dc_train, Y_dc_train)
    
    A_coef = lr.coef_
    b_inter = lr.intercept_
    
    return A_coef, b_inter


def linear_model_IEEE39(dc_dataset, num_samples=180):
    X_dc_train = dc_dataset.loc[:num_samples, ['Pg0', 'Pg1', 'Pg2', 'Pg3', 'Pg4', 'Pg5', 'Pg6', 'Pg7', 'Pg8', 'Pg9',
                                               'Load_P1', 'Load_P2', 'Load_P3', 'Load_P4', 'Load_P5', 'Load_P6', 'Load_P7', 'Load_P8', 
                                               'Load_P9', 'Load_P10', 'Load_P11', 'Load_P12', 'Load_P13', 'Load_P14', 'Load_P15',
                                               'Load_P16', 'Load_P17', 'Load_P18', 'Load_P19', 'Load_P20', 'Load_P21', 
                                               'RS_P1', 'RS_P2', 'RS_P3', 'RS_P4', 'RS_P5', 'RS_P6']]
    Y_dc_train = dc_dataset.loc[:num_samples, ['P_0_1', 'P_0_38', 'P_1_2', 'P_1_24', 'P_2_3', 'P_2_17', 'P_3_4', 'P_3_13', 'P_4_5', 'P_4_7', 
                                               'P_5_6', 'P_5_10', 'P_6_7', 'P_7_8', 'P_8_38', 'P_9_10', 'P_9_12', 'P_12_13', 'P_13_14', 
                                               'P_14_15', 'P_15_16', 'P_15_18', 'P_15_20', 'P_15_23', 'P_16_17', 'P_16_26', 'P_20_21', 
                                               'P_21_22','P_22_23', 'P_22_35', 'P_24_25', 'P_25_26', 'P_25_27', 'P_25_28', 'P_27_28']]
    
    lr = LinearRegression()
    lr.fit(X_dc_train, Y_dc_train)
    
    A_coef = lr.coef_
    b_inter = lr.intercept_
    
    return A_coef, b_inter


# In[ ]:




