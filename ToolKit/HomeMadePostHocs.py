# -*- coding: utf-8 -*-
"""
Created on Mon May 17 16:34:16 2021

@author: Angel.BAUDON
"""
import numpy as np, pandas as pd

def Dunnett_table(number_of_observations, number_of_groups):
    '''
    Arguments:
        - n: number of observations
        - grp: number of groups
    '''
    n = number_of_observations - number_of_groups
    grp = number_of_groups
    
    if n <= 5:
        if grp == 2: return 4.03, 2.57
        if grp == 3: return 4.63, 3.03
        if grp == 4: return 4.98, 3.29        
        if grp == 5: return 5.22, 3.48
        if grp == 6: return 5.41, 3.62
        if grp == 7: return 5.56, 3.73        
        if grp == 8: return 5.69, 3.82

    if n == 6:
        if grp == 2: return 3.71, 2.45        
        if grp == 3: return 4.21, 2.86
        if grp == 4: return 4.51, 3.1        
        if grp == 5: return 4.71, 3.26
        if grp == 6: return 4.87, 3.39
        if grp == 7: return 5, 3.49        
        if grp == 8: return 5.1, 3.57

    if n == 7:
        if grp == 2: return 3.5, 2.36        
        if grp == 3: return 3.95, 2.75
        if grp == 4: return 4.21, 2.97        
        if grp == 5: return 4.39, 3.12
        if grp == 6: return 4.53, 3.24
        if grp == 7: return 4.64, 3.33        
        if grp == 8: return 4.74, 3.41

    if n == 8:
        if grp == 2: return 3.36, 2.31        
        if grp == 3: return 3.77, 2.67
        if grp == 4: return 4, 2.88        
        if grp == 5: return 4.17, 3.01
        if grp == 6: return 4.29, 3.13
        if grp == 7: return 4.4, 3.22        
        if grp == 8: return 4.48, 3.29

    if n == 9:
        if grp == 2: return 3.25, 2.26        
        if grp == 3: return 3.63, 2.61
        if grp == 4: return 3.85, 2.81        
        if grp == 5: return 4.01, 2.95
        if grp == 6: return 4.12, 3.05
        if grp == 7: return 4.22, 3.14        
        if grp == 8: return 4.3, 3.2

    if n == 10:
        if grp == 2: return 3.17, 2.23        
        if grp == 3: return 3.53, 2.57
        if grp == 4: return 3.74, 2.76        
        if grp == 5: return 3.88, 2.89
        if grp == 6: return 3.99, 2.99
        if grp == 7: return 4.08, 3.07        
        if grp == 8: return 4.16, 3.14
    
    if n == 11:
        if grp == 2: return 3.11, 2.2        
        if grp == 3: return 3.45, 2.53
        if grp == 4: return 3.65, 2.72        
        if grp == 5: return 3.79, 2.84
        if grp == 6: return 3.89, 2.94
        if grp == 7: return 3.98, 3.02        
        if grp == 8: return 4.05, 3.08
    
    if n == 12:
        if grp == 2: return 3.05, 2.18        
        if grp == 3: return 3.39, 2.5
        if grp == 4: return 3.58, 2.68        
        if grp == 5: return 3.71, 2.81
        if grp == 6: return 3.81, 2.9
        if grp == 7: return 3.89, 2.98        
        if grp == 8: return 3.96, 3.04

    if n == 13:
        if grp == 2: return 3.01, 2.16        
        if grp == 3: return 3.33, 2.48
        if grp == 4: return 3.52, 2.65        
        if grp == 5: return 3.65, 2.78
        if grp == 6: return 3.74, 2.87
        if grp == 7: return 3.82, 2.94        
        if grp == 8: return 3.89, 3

    if n == 14:
        if grp == 2: return 2.98, 2.14        
        if grp == 3: return 3.29, 2.46
        if grp == 4: return 3.47, 2.63        
        if grp == 5: return 3.59, 2.75
        if grp == 6: return 3.69, 2.84
        if grp == 7: return 3.76, 2.91        
        if grp == 8: return 3.83, 2.97

    if n == 15:
        if grp == 2: return 2.95, 2.13        
        if grp == 3: return 3.25, 2.44
        if grp == 4: return 3.43, 2.61        
        if grp == 5: return 3.55, 2.73
        if grp == 6: return 3.64, 2.82
        if grp == 7: return 3.71, 2.89        
        if grp == 8: return 3.78, 2.95
            
    if n == 16:
        if grp == 2: return 2.92, 2.2        
        if grp == 3: return 3.22, 2.42
        if grp == 4: return 3.39, 2.59        
        if grp == 5: return 3.51, 2.71
        if grp == 6: return 3.6, 2.8
        if grp == 7: return 3.67, 2.87        
        if grp == 8: return 3.73, 2.92
       
    if n == 17:
        if grp == 2: return 2.9, 2.11        
        if grp == 3: return 3.19, 2.41
        if grp == 4: return 3.36, 2.58        
        if grp == 5: return 3.47, 2.69
        if grp == 6: return 3.56, 2.78
        if grp == 7: return 3.63, 2.85        
        if grp == 8: return 3.69, 2.9

    if n == 18:
        if grp == 2: return 2.88, 2.1        
        if grp == 3: return 3.17, 2.4
        if grp == 4: return 3.33, 2.56        
        if grp == 5: return 3.44, 2.68
        if grp == 6: return 3.53, 2.76
        if grp == 7: return 3.6, 2.83        
        if grp == 8: return 3.66, 2.89
    
    if n == 19:
        if grp == 2: return 2.86, 2.09
        if grp == 3: return 3.15, 2.39
        if grp == 4: return 3.31, 2.55
        if grp == 5: return 3.4, 2.66
        if grp == 6: return 3.5, 2.75
        if grp == 7: return 3.57, 2.81
        if grp == 8: return 3.63, 2.87
    
    if n >= 20 and n < 30:
        if grp == 2: return 2.85, 2.09
        if grp == 3: return 3.13, 2.38
        if grp == 4: return 3.29, 2.54
        if grp == 5: return 3.4, 2.65
        if grp == 6: return 3.48, 2.73
        if grp == 7: return 3.55, 2.8
        if grp == 8: return 3.6, 2.86
    
    if n >= 30 and n < 40:
        if grp == 2: return 2.75, 2.04
        if grp == 3: return 3.01, 2.32
        if grp == 4: return 3.15, 2.47
        if grp == 5: return 3.25, 2.58
        if grp == 6: return 3.33, 2.66
        if grp == 7: return 3.39, 2.72
        if grp == 8: return 3.44, 2.77
    
    else:
        if grp == 2: return 2.7, 2.02
        if grp == 3: return 2.27, 2.29
        if grp == 4: return 3.09, 2.44
        if grp == 5: return 3.19, 2.54
        if grp == 6: return 3.26, 2.62
        if grp == 7: return 3.32, 2.68
        if grp == 8: return 3.37, 2.73

       
def posthoc_dunnett(data, val_col='Values', group_col='Groups'):
    
    import scipy
    '''
    Arguments:
        - data: pd.DataFrame that contains at lest two columns
        - val_col: the name of the column that contains the dependent variable
                   !!! At least 1 column sould contain the name "ctrl" !!!
        - group_col: the name of the column that contains the independent variable
    '''
    
    group_names = sorted(list(set(data[group_col])))
    try: ctrl_name = [x for x in group_names if 'trl' in x][0]
    except IndexError: ctrl_name = group_names[0]
    ctrl_grp = np.asarray(data[data[group_col] == ctrl_name][val_col])
    
    
    values = np.asarray(data[val_col])
    
    Tds = Dunnett_table(len(data[val_col]), len(group_names))

    SSw = sum([(d-np.mean(values))**2 for d in values])
    df_within = len(values) - len(group_names)
    MSw = SSw / df_within
    
    pvalz = {}
    for grp in [x for x in group_names if x != ctrl_name]:
        val_grp = np.asarray(data[data[group_col] == grp][val_col])
        
        Du1, Du5 = [Td * np.sqrt((2*MSw)/len(val_grp)) for Td in Tds]
        
        Du = abs(np.nanmean(val_grp)-np.nanmean(ctrl_grp))
        
        pval = .01 if Du > Du1 else .05 if Du > Du5 else 1
        print('\n\n pval1 : ', pval)
        
        #Find exact pval
        '''!!! I'm not sure of this part, this seems theorically correct
                and it pass the test of random data but it need confirmation !!! '''
        
        #Find sigma of the Dunnett distribution by dividing the border at 5%
        #which correspond to 1.96 * sigma
        sigma_dunnett = Du5/1.96
        
        #Normalize the Du statistic by sigma to normalise it and create a z-score
        z_score = Du / sigma_dunnett
        print(z_score)
        
        #Find the area that correspond to this z-score border (*2 because of bilaterality)
        pval = scipy.stats.norm.sf(abs(z_score))*2
        print('pval2 : ', pval, '\n\n')

        pvalz[f'{ctrl_name} vs {grp}'] = pval
    return pvalz



def posthoc_dunn_sidak(data, val_col='Values', group_col='Groups'):
    from scikit_posthocs import posthoc_dunn
    
    group_names = sorted(list(set(data[group_col])))
    try: ctrl_name = [x for x in group_names if 'trl' in x][0]
    except IndexError: ctrl_name = group_names[0]
    n_comparisons = len(group_names) - 1
    
    ph = posthoc_dunn(data, val_col=val_col, group_col=group_col)
    
    comp_vs_ctrl = [x for x in group_names if x != ctrl_name]
    
    pvalz = {}
    for grp in comp_vs_ctrl:
        unc_pval = ph.loc[ctrl_name, grp]
        pval = 1 - (1 - unc_pval) ** n_comparisons
        pvalz[f'{ctrl_name} vs {grp}'] = pval
    return pvalz

    

if __name__ == '__main__':
    values = np.random.randn(40)
    df = pd.DataFrame({'Groups':list('ABCD')*10, 'Values':values})
    print(posthoc_dunnett(df))







'''
References

To choose the test:
https://www.unistat.com/guide/nonparametric-tests-kruskal-wallis-one-way-anova/
https://stats.stackexchange.com/questions/500723/dunn-test-and-specific-comparisons

For Dunett test:
https://www.statisticshowto.com/dunnetts-test/
https://www.statology.org/dunnetts-test/

    
For sidak correction:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3045855/ (doi: 10.1037/a0012850)
https://www.statology.org/dunns-test/

'''