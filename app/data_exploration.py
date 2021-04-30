import matplotlib.pyplot as plt
import pandas as pd


def plot_cat_var(explore_col):
    target_col = 'ACCEPT_INDICATOR'

    rr_overall = 29

    df = pd.read_csv(path + file)
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

if __name__ == '__main__':

    path = 'C:/project/DG_thesis/'
    path = 'C:/Users/DebasishGuha/DG/Personal/MTech/Sem4/Project/work/'
    file = 'EU_OFFER_PROPENSITY_VIEW_new.csv'

    col_list = ['EDUCATION_ID', 'LIFESTYLE_CATEGORY_ID']

    col_list = ['EDUCATION_ID', 'LIFESTYLE_CATEGORY_ID', 'FAMILY_CYCLE_ID', 'SOCIAL_CLASS_ID', 'HOUSE_PROPERTY_ID',
                'AGE_GROUP_ID', 'INCOME_LEVEL_ID', 'BULDING_TYPE_ID', 'BUILDING_VOLUME_LEVEL']

    for explore_col in col_list:
        plot_cat_var(explore_col)

