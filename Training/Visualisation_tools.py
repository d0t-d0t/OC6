import seaborn as sns

def plot_pred_per_type(X_test,
                       cible = 'prediction_type',
                       bin_cible = 'score_probability',
                       color_palette={
                           'TP': 'green',
                           'TN': 'blue',
                           'FP': 'orange',
                           'FN': 'red',
                       },
                       not_to_show=[]):
    X_filtered = X_test[~X_test[cible].isin(not_to_show)]
    # fig, ax = plt.subplots()
    g = sns.displot(data=X_filtered,
                x=bin_cible,
                hue=cible,
                palette='viridis' ,#color_palette,
                kind='hist',
                bins=20,
                multiple="stack",
                
                
                )
    g.figure.legend(loc='upper right')
    return g.figure
    