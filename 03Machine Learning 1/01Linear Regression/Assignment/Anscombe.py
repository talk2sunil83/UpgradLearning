# %%
import seaborn as sns
# sns.set_theme(style="ticks")

# Load the example dataset for Anscombe's quartet
df = sns.load_dataset("anscombe")

# %%
df.pivot(index='dataset')
# values=["x", "y"], 
# %%
df.pivot(columns='dataset')
# %%
df.describe()

# Show the results of a linear regression within each dataset
sns.lmplot(x="x", y="y", col="dataset", hue="dataset", data=df,
           col_wrap=2, ci=None, palette="muted", height=4,
           scatter_kws={"s": 50, "alpha": 1})
# %%
