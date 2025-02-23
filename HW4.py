# %% [markdown]
# # Week 4 - Univariate Analysis, part 2

# %% [markdown]
# # 1. Lesson - None

# %% [markdown]
# # 2. Weekly graph question

# %% [markdown]
# Below are a histogram and boxplot representation of the same data. A pharmacy is keeping a record of the prices of the drugs that it sells, and an administrator wants to know how much the more expensive drugs tend to cost, in the context of the other prices.
# 
# Please write a short explanation of the pros and cons of these two representations. Which would you choose? How would you modify the formatting, if at all, to make it more visually interesting, clear, or informative?

# %%
import numpy as np
import pandas as pd
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
import os

np.random.seed(0)
num_data = 100
data = np.exp(np.random.uniform(size = num_data) * 4)
df = pd.DataFrame(data.T, columns = ["data"])

# %%
print("The 75th percentile is:", df.quantile(q = 0.75))
df.plot.hist()

# %%
df.plot.box()

# %% [markdown]
# # 3. Homework - working on your datasets

# %% [markdown]
# This week, you will do the same types of exercises as last week, but you should use your own datasets that you found last semester.
# 
# ### Here are some types of analysis you can do:
# 
# - Draw histograms and histogram variants for each feature or column.  (Swarm plot, kde plot, violin plot).
# 
# - Draw grouped histograms.  For instance, if you have tree heights for both maple and oak trees, you could draw histograms for both.
# 
# - Draw a bar plot to indicate total counts of each categorical variable in a given column.
# 
# - Find means, medians, and modes.
# 
# ### Conclusions:
# 
# - Explain what conclusions you would draw from this analysis: are the data what you expect?  Are the data likely to be usable?  If they are not useable, find some new data!
# 
# - What is the overall shape of the distribution?  Is it normal, skewed, bimodal, uniform, etc.?
# 
# - Are there any outliers present?  (Data points that are far from the others.)
# 
# - If there are multiple related histograms, how does the distribution change across different groups?
# 
# - What are the minimum and maximum values represented in each histogram?
# 
# - How do bin sizes affect the histogram?  Does changing the bin width reveal different patterns in the data?
# 
# - Does the distribution appear normal, or does it have a different distribution?

# %%
# set variables
kaggle_dataset_path = "ramyhafez/bank-customer-churn"
kaggle_dataset_file_name = "Bank_Churn.csv"
print(f"Path to kaggle dataset: {kaggle_dataset_path}")
print(f"Kaggle dataset file name: {kaggle_dataset_file_name}")

# download the data set
kaggle_dataset_local_path = kagglehub.dataset_download(kaggle_dataset_path)
print(f"Path to downloaded file: {kaggle_dataset_local_path}")

# read csv file to pandas dataframe
kaggle_dataset_local_path_to_file = os.path.join(kaggle_dataset_local_path, kaggle_dataset_file_name)
kaggle_dataset_raw = pd.read_csv(kaggle_dataset_local_path_to_file)

# copy dataframe for EDA
dataset = kaggle_dataset_raw.copy()

# show top rows
dataset.head()


# %%
# Your code here:

# %% [markdown]
# # 4. Storytelling With Data graph

# %% [markdown]
# Reproduce any graph of your choice in p. 52-68 of the Storytelling With Data book as best you can.  (The second half of chapter two).  You do not have to get the exact data values right, just the overall look and feel.

# %%



