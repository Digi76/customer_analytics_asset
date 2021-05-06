import matplotlib.pyplot as plt
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

cfg_file_nm = "../config/eda_config.json"
with open(cfg_file_nm, "r") as f:
    cfg_dict = json.load(f)

file_nm = cfg_dict.get('input_file')
cat_col_list = cfg_dict.get('categorical_col_list')
target_col = cfg_dict.get('target_var')
data_dict = cfg_dict.get('data_dict')

def plot_cat_var(df, explore_col):
    #target_col = 'ACCEPT_INDICATOR'

    rr_overall = 29

    #df = pd.read_csv(path + file)
    df_size = df.shape[0]

    df_count = df[explore_col].value_counts().reset_index()
    df_count.rename(columns={explore_col: 'count', 'index': explore_col}, inplace=True)
    df_count['Coverage%'] = (df_count['count'] / df_size) * 100

    df_grp = df.groupby(explore_col)[target_col].sum().reset_index()

    df_final = pd.merge(df_count, df_grp, on=explore_col, how='inner')
    df_final['Response Rate%'] = (df_final[target_col] / df_final['count']) * 100
    df_final.drop(['count', target_col], 1, inplace=True)
    df_final.sort_values(explore_col, inplace=True)
    df_final['RR_Overall%'] = rr_overall

    # create figure and axis objects with subplots()
    fig, ax = plt.subplots()
    # make a plot
    # ax.plot(df_final[explore_col], df_final['Coverage%'], color="red", marker="o")
    ax.bar(df_final[explore_col], df_final['Coverage%'], color="b", width=0.25, label='Coverage%')
    # set x-axis label
    ax.set_xlabel(explore_col, fontsize=14)
    # set y-axis label
    ax.set_ylabel("Coverage%", color="black", fontsize=14)

    # twin object for two different y-axis on the sample plot
    ax2 = ax.twinx()
    # make a plot with different y-axis using second axis object
    ax.plot(df_final[explore_col], df_final['Response Rate%'], color="red", marker="o", label='Response Rate%')
    ax.plot(df_final[explore_col], df_final['RR_Overall%'], color="green", linestyle='dashed', label='RR_Overall%')
    ax2.set_ylabel('Response Rate%', color="black", fontsize=14)
    ax.legend()
    plt.show()

def plot_cont_var(data,col_list):
    plt.xticks(rotation=90)

    sns.boxplot(x="variable", y="value", data=pd.melt(data[col_list]))

    plt.show()


if __name__ == '__main__':

    col_list = cat_col_list

    df = pd.read_csv(file_nm)

    for explore_col in col_list:
        plot_cat_var(df,explore_col)


    for tbl in data_dict:
        file_nm = tbl.get("file_nm")
        cont_col_list = tbl.get("cont_col_list")

        #print(file_nm, col_list)
        data = pd.read_csv(file_nm)
        plot_cont_var(data,cont_col_list)