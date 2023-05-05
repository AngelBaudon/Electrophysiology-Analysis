# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 13:05:09 2021

@author: Angel.BAUDON


To do on this code :
    - WARNING ! Every pval are corrected with yhe bonferoni method wich is the most conservative !
    - QQ plots (& unbiased methods to estimate the fit ?)
    - Test sphericity ?
    
"""
def IntraGrpStat(data, group_name=False, Paired=False):
    
    import scipy, numpy as np, pandas as pd, pingouin as pg
    from statsmodels.stats.anova import AnovaRM
    from scikit_posthocs import posthoc_dunn, posthoc_conover, posthoc_tukey
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from ToolKit.HomeMadePostHocs import posthoc_dunnett, posthoc_dunn_sidak
    

    
    norm, residuals, groups = [], [], []

    if isinstance(data, pd.DataFrame):
        df = data
        group_col, data_col = df.columns
        for trt in list(set(df[group_col])):
            data = df[df[group_col] == trt][data_col]
            residuals.extend([x-np.nanmean(data) for x in data])
            groups.append([d for d in data if str(d) != 'nan'])
        group_name = sorted(list(set(df[group_col])))

    if isinstance(data, list) and not group_name:
        group_name, groups = [f'Time {x}' for x,y in enumerate(data)], data
    
    
    '''               Test the criteria for parametric tests         '''
    rest1 = [[x-np.nanmean(group) for x in group] for group in groups]
    rest2 = [x for y in rest1 for x in y]
    
    
    #Normality
    S, p_val_s = scipy.stats.shapiro(rest2)
    norm.append(False) if p_val_s < 0.05 else norm.append(True)
    
    #Equality of variances
    L, p_val_l = scipy.stats.levene(*rest1)
    norm.append(False) if p_val_l < 0.05 else norm.append(True)
    
    
    
    '''                                   Decision tree                 '''
    #T-test familly
    if len(groups) == 2:
        #Parametric test
        if not False in norm:
            if not Paired:
                stat, pval = scipy.stats.ttest_ind(*groups)
                test = 'Paired t-Test'
            if Paired:
                stat, pval = scipy.stats.ttest_rel(*groups)
                test = 'Unpaired t-Test'
                
                
        #Non parametric test
        if False in norm:
            if not Paired:
                stat, pval = scipy.stats.wilcoxon(*groups)
                test = 'Wilcoxon'
            if Paired:
                stat, pval = scipy.stats.mannwhitneyu(*groups)
                test = 'Mann-Whitney'
          
    
    #Anova familly
    elif len(groups) > 2:

        comp_name = []
        for g, group in enumerate(group_name):
            for d in range(len(group_name)-g-1):
                comp_name.append(f'{g} vs {group_name[d+g+1]}')

        
        if not Paired:
            if not False in norm:
                
                stat, pval = scipy.stats.f_oneway(*groups)
                ph = posthoc_dunnett(df, val_col=data_col, group_col=group_col)
                
                result = {'Test': "One-Way ANOVA & Dunnett's MC",
                          'Stat AOV': stat, 'Pval AOV': pval}
                result = dict(result, **ph)
                # print('Dunn : ', ph, '\n\n')
                
                
                # ph = posthoc_tukey(df, val_col=data_col, group_col=group_col)
                # result = {'Test': "One-Way ANOVA & Tukey's MC",
                #           'Stat AOV': stat, 'Pval AOV': pval}
                
                # for c in range(len(ph.columns)):
                #     for d in range(1, len(ph.columns)-c):
                #         result[comp_name[0]] = ph.iat[c,d+c]
                #         comp_name.pop(0)
                # print('Tukey : ', results, '\n\n')
                    
            else:
                stat, pval = scipy.stats.kruskal(*groups)
                ph = posthoc_dunn_sidak(df, val_col=data_col, group_col=group_col)
                
                result = {'Test': "Kruskal-Wallis AOV & Dunn's MC with Sidak correction",
                          'Stat AOV': stat, 'Pval AOV': pval}
                result = dict(result, **ph)



        # if Paired:
        #     if not False in norm:
        #         aovrm = AnovaRM(df, depvar=data_name, subject='Cell', within=[group_name])
        #         res = aovrm.fit().summary().tables[0]
        #         stat, pval = float(res['F Value']), float(res['Pr > F'])
    
        #         # ph = pairwise_ttests(data=df, dv='Values', within='Time',
        #         #                      subject='Cell', padjust='bonf')
    
        #         ph = pairwise_tukeyhsd(df[data_name], df[group_name])
                
        #         ph = pd.DataFrame(data=ph._results_table.data[1:],
        #                           columns=ph._results_table.data[0])
        #         ph_out = {'Test': 'RM ANOVA & paired Tukey'}
        #         for x, y, z in zip(*[list(ph[x]) for x in ['group1', 'group2', 'p-adj']]):
        #             ph_out[f'{x} vs {y}'] = z
    
            
        #     else: 
        #         stat, pval = scipy.stats.friedmanchisquare(*groups)
        #         ph = posthoc_conover(groups, p_adjust = 'bonferroni')
                
        #         ph_out = {'Test': 'Friedman & Wilcoxon with Bonferoni correction'}
        #         for x, y, c1, c2 in zip([1,1,2], [2,3,3], comp1, comp2):
        #             ph_out[f'{c1} vs {c2}'] = float(ph.loc[x, y])
    
    return(pd.DataFrame(result.items(), columns=('Test', 'Result')))
    # try: return(round(stat, 4), round(pval, 4), ph_out)
    # except NameError: return(round(stat, 4), round(pval, 4), test)
