import pandas as pd
import numpy as np
import warnings
from scipy.stats import ttest_ind, f_oneway
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings(action='ignore')


cluster_data = pd.read_csv('./stage_result.csv')
CombinedNames = cluster_data['CombinedName']
Clusters = cluster_data['Cluster']

cd_data = pd.read_csv('./.csv')  #Neuropsychological scale score data
features = cd_data.columns.tolist()
features.remove('CombinedName')

matched_cd_data = cd_data[cd_data['CombinedName'].isin(CombinedNames)].copy()

for feature in features:
    non_nan_data = matched_cd_data[feature].dropna().values.reshape(-1, 1)
    if len(non_nan_data) > 0:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(non_nan_data).flatten()
        matched_cd_data.loc[matched_cd_data[feature].notna(), feature] = scaled_data

result_filename = './Result_NSD.csv'
significant_count = 0

with open(result_filename, 'w') as file:
    file.write('Feature,ANOVA_p_value,Between-group_df,Within-group_df,p12,p13,p23,Significant\n')

    for feature_name in features:
        feature_values = np.array(
            [matched_cd_data.loc[matched_cd_data['CombinedName'] == name, feature_name].values[0]
             if name in matched_cd_data['CombinedName'].values else np.nan
             for name in CombinedNames]
        )

        cluster1 = feature_values[Clusters == 1]
        cluster2 = feature_values[Clusters == 2]
        cluster3 = feature_values[Clusters == 3]
        cluster1 = cluster1[~np.isnan(cluster1)]
        cluster2 = cluster2[~np.isnan(cluster2)]
        cluster3 = cluster3[~np.isnan(cluster3)]

        k = 3
        N = len(cluster1) + len(cluster2) + len(cluster3)
        between_group_df = k - 1
        within_group_df = N - k

        if len(cluster1) > 1 and len(cluster2) > 1 and len(cluster3) > 1:
            anova_pvalue = f_oneway(cluster1, cluster2, cluster3).pvalue
            p12 = ttest_ind(cluster1, cluster2).pvalue
            p13 = ttest_ind(cluster1, cluster3).pvalue
            p23 = ttest_ind(cluster2, cluster3).pvalue

            is_significant = (anova_pvalue < 0.05) and all(p < 0.05 for p in [p12, p13, p23])
            file.write(
                f'{feature_name},{anova_pvalue:.30f},{between_group_df},{within_group_df},{p12:.30f},{p13:.30f},{p23:.30f},{int(is_significant)}\n')

            if is_significant:
                significant_count += 1

print(f'The number of significant features between all pairings of subtypes is: {significant_count}')

