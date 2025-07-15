#produce MCI stage label
import pandas as pd


file_1 = f'./cluster_result.csv'

df1 = pd.read_csv(file_1)

df1 = df1.iloc[:-4]

file_cd = './.csv' #MOCA data
df_cd = pd.read_csv(file_cd)

df_merged = pd.merge(df1, df_cd[['CombinedName', 'MOCA']], on='CombinedName', how='left')

cluster_moca_means = df_merged.groupby('Cluster')['MOCA'].mean()

sorted_clusters = cluster_moca_means.sort_values().index
cluster_mapping = {cluster: new_cluster for new_cluster, cluster in enumerate(sorted_clusters[::-1], start=1)}

df1['Cluster'] = df1['Cluster'].map(cluster_mapping)


print(f" Modify the mapping relationship: {cluster_mapping}")


output_file = f'./Final_cluster_result.csv'
df1.to_csv(output_file, index=False, header=True)


print(f"The modified file has been saved as {output_file}.")

