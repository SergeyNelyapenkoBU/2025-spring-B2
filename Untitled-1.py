# %%


# %% [markdown]
# # Week 1 - Preprocessing
# 
# ## Please run the cells of the notebook as you get to them while reading

# %%
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# %% [markdown]
# # 1. Lesson on how to search for Python commands
# 
# Let's consider a few possible ways to learn about Python programming.  Let's suppose you want to learn how to produce a short summary of the information in your DataFrame.
# 
# 1. Your **instructor** could provide the information.
# 
# You could be provided with a lesson about functions like info() and describe().  If you have a pandas DataFrame called df, then you can summarize its contents using df.info() or df.describe().  df.info() provides a list of column names with their counts and data types.  df.describe() will provide information such as the mean, min, max, standard deviation, and quantiles.  Thus:

# %%
df = pd.DataFrame([[1, 4], [2, 5], [3, 6], [4, 7]], columns = ['A', 'B'])
df.describe()

# %% [markdown]
# In this describe() result, we see that the two columns A and B each have four elements.  The means and other statistics are shown.
# 
# 2. You could look up the information on **Google**.
# 
# If I Google the question "how do I briefly summarize the contents of a dataframe using Python," I receive the following link (among others), which discusses the describe() command mentioned above:
# 
# https://www.w3schools.com/python/pandas/ref_df_describe.asp
# 
# It also provide the complete usage information:
# 
# dataframe.describe(percentiles, include, exclude, datetime_is_numeric)
# 
# It explains that "percentiles" is set by default to [0.25, 0.5, 0.75] but we could change that.  Let's try it!  Since there are three intervals here rather than four, it might be more meaningful to ask about a 33rd and 67th percentile rather than 25, 50, and 75.  We can use 1/3 for 0.33 and 2/3 for 0.67 to get the exact percentile values.

# %%
df = pd.DataFrame([[1, 4], [2, 5], [3, 6], [4, 7]], columns = ['A', 'B'])
df.describe(percentiles = [1/3, 2/3])

# %% [markdown]
# Apparently, the 50% value (the median) stays even though we did not specifically request it.
# 
# 3. You could look up the official **documentation**.
# 
# Now that we know we want the pandas describe() function, try Googling: pandas documentation describe.
# 
# Here is the general documentation page for pandas:
# 
# https://pandas.pydata.org/docs/index.html
# 
# Here is the specific page for the describe() function:
# 
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html
# 
# When I look at this, it appears to be showing the most recent (currently 2.2) version of pandas; this is shown in the upper right corner.
# 
# 4. You could also ask **ChatGPT**.
# 
# Let's try it.  ChatGPT, "how do I briefly summarize the contents of a dataframe using Python"
# 
# When I do this, ChatGPT mentions describe() among other options, but does not go into detail.  However, I could ask it.  ChatGPT, "tell me more about describe() in Python for summarizing dataframes."
# 
# Then, I get a good explanation of describe(), although it does not mention the percentiles option.  One advantage of using Google or the documentation in addition of ChatGPT is that these sources may provide interesting information that does not directly answer our question.  Thus, we might not have known about the various arguments, such as percentiles, if we only used ChatGPT.  A second issue is that ChatGPT sometimes hallucinates (it makes up information).  In general, by examining multiple sources - Google, documentation, and ChatGPT - we can get more information.

# %% [markdown]
# # 2. Weekly graph question

# %% [markdown]
# In Storytelling With Data, on page 1: examine the pie chart graph in the upper left corner of the graphs.  Please write a short explanation of the pros and cons of this graph.  What do you think of the choice of pie chart as a format?  The color scheme?  The legend?  The title?  How would you draw it differently if you were creating this graph?

# %% [markdown]
# # 3. Homework - Bank Customers
# 
# I will begin by creating a file for you to analyze.  I will show you all of the steps I used to create it.  Please run this code in order to create and save a file about bank customers.
# 
# ### The numbered problems are for you to solve.

# %%
num_customers = 100
np.random.seed(0)

# %%
df_bank = pd.DataFrame(columns = ["CustomerID"])

# %%
df_bank["CustomerID"] = [str(x) for x in np.arange(num_customers)]

# %%
start = datetime(1950, 1, 1)
end = datetime(2024, 1, 1)
numdays = (end - start).days
random_days = np.random.randint(0, numdays, size = num_customers)
df_bank["BirthDate"] = start + pd.to_timedelta(random_days, unit='D')
df_bank["BirthDate"] = df_bank["BirthDate"].dt.strftime('%Y-%m-%d')

# %%
def make_ssn_string(num):
    ssn_str = f'{num:09}'
    return ssn_str[0:3] + "-" + ssn_str[3:5] + "-" + ssn_str[5:9]
ssn_vector_func = np.vectorize(make_ssn_string)
df_bank["SSN"] = ssn_vector_func(np.random.randint(0, 999999999, size = num_customers))

# %%
df_bank["AccountID"] = np.random.randint(0, num_customers, size = num_customers)

# %%
random_days = np.random.randint(0, 365 * 80, size = num_customers)
df_bank["AccountOpened"] = (pd.to_datetime(df_bank["BirthDate"]) + pd.to_timedelta(random_days, unit='D')).dt.strftime('%Y-%m-%d')

# %%
df_bank.loc[0, "BirthDate"] = "1980"
df_bank.loc[1, "BirthDate"] = "no date"

# %%
df_bank.loc[2, "AccountID"] = np.nan

# %%
df_bank["AccountType"] = np.random.choice(["checking", "savings", "cd"], size = num_customers)

# %% [markdown]
# Load the bank_customers.csv file.  (There is no practical reason to save it, then load it - we're just demonstrating how this would be done.)
# I am calling the loaded df by a new name, df_bank_loaded, to make clear why it's not the same variable as the old df.  Of course, in actuality the two contain the exact same data!  But it's good to get in the habit of naming things carefully.

# %%
df_bank.loc[num_customers - 1] = df.loc[0]
df_bank.to_csv("bank_customers.csv", index=False)

# %%
df_bank_loaded = pd.read_csv("bank_customers.csv")

# %% [markdown]
# 1. Use describe() and info() to analyze the data.   Also, look at the first few rows.

# %% [markdown]
# Suggested Google Search or ChatGPT prompt: "how do I use the describe function in python"
# 
# Example Google result: https://www.w3schools.com/python/pandas/ref_df_describe.asp

# %%
# The first few rows
df_bank_loaded.iloc[0:5]

# %% [markdown]
# If you used describe() and info(), you now know that BirthDate and AccountOpened are strings.  But we want them to be dates.  Let's convert them to dates (or Timestamps in pandas).  When we try this, we get a ValueError.

# %%
try:
    df_bank_loaded["BirthDate"] = pd.to_datetime(df_bank_loaded["BirthDate"], format='%Y-%m-%d')
    print("It worked!")
except ValueError as e:
    print(f"ValueError for BirthDate: {e}")

# %%
try:
    df_bank_loaded["AccountOpened"] = pd.to_datetime(df_bank_loaded["AccountOpened"], format='%Y-%m-%d')
    print("It worked!")
except ValueError as e:
    print(f"ValueError for AccountOpened: {e}")

# %% [markdown]
# The simple way to fix this is to remove the rows that have bad dates for BirthDate.  I Googled:
# 
# "How to remove rows from a dataframe that have poorly formatted dates using python"
# 
# https://stackoverflow.com/questions/21556744/pandas-remove-rows-whose-date-does-not-follow-specified-format
# 
# This recommends that I verify that the date is a string of length 10, because YYYY-MM-DD has that length:
# 
# df1\[df1.BirthDate.str.len() !=10]

# %%
len(df_bank_loaded[df_bank_loaded.BirthDate.str.len() == 10])

# %%
df_bank_loaded[df_bank_loaded.BirthDate.str.len() != 10].iloc[0:5]

# %% [markdown]
# Now we can make this permanent, creating a new DataFrame df_bank_datefix.
# I am making a copy in order to ensure that df_bank_datefix is a new DataFrame rather than being a slice of the old one.

# %%
df_bank_datefix = df_bank_loaded[df_bank_loaded.BirthDate.str.len() == 10].copy()


df_bank_loaded[df_bank_loaded.BirthDate.str.len() == 10]


# %% [markdown]
# Test again:

# %%
try:
    df_bank_datefix["BirthDate"] = pd.to_datetime(df_bank_datefix["BirthDate"], format='%Y-%m-%d')
    print("It worked!")
except ValueError as e:
    print(f"ValueError: {e}")

# %% [markdown]
# 2. To check that it worked, use a summary function that will tell you if the BirthDate field is now a datetime type

# %%
df_bank_datefix.info()

# %% [markdown]
# 3. Check whether there are any null values in the DataFrame.  If so, remove those rows or (if you prefer) fill in the value with an appropriate number.
# 
# First try at a Google search or ChatGPT prompt: "how do I find out if there are any null values in a pandas DataFrame?"
# 
# This page gives an answer.  Unfortunately, it took my request too literally: it tells me only if there are any, and not which rows have them.  On reflection, that's not really what I want - I think I asked the wrong question.  I want to see the rows, not just _whether_ there are any.
# 
# https://stackoverflow.com/questions/29530232/how-to-check-if-any-value-is-nan-in-a-pandas-dataframe
# 
# ChatGPT likewise doesn't give the answer I want - because I asked the wrong question.
# 
# Next try at a Google search or ChatGPT prompt: "how do I check which rows have null values in a pandas DataFrame?"
# 
# This page gives an answer:
# 
# https://stackoverflow.com/questions/36226083/how-to-find-which-columns-contain-any-nan-value-in-pandas-dataframe
# 
# ChatGPT also gives a good answer.  I recommend looking at both of them!
# 
# Now try it on your own:
# 
# Suggested Google search or ChatGPT prompt: "how do I remove rows with null values in a pandas DataFrame?"
# 
# Suggested Google search or ChatGPT prompt: "how do I fill in null values in a pandas DataFrame?"

# %%
df_bank_datefix.isna().any()

# %%
df_bank_nafix_drop = df_bank_datefix.dropna()
df_bank_nafix_drop.isna().any()

# %%
df_bank_nafix_fill = df_bank_datefix.fillna(-1)
df_bank_nafix_fill.isna().any()

# %% [markdown]
# 4. Find out if there are any duplicate rows (two rows exactly the same).  List their row numbers.  Then remove the duplicates

# %% [markdown]
# Suggested Google search or ChatGPT prompt: "how can I find out if there are any duplicate rows in a DataFrame using Python"
# 
# Again, Google provides me with a page that addresses the question:
# 
# https://saturncloud.io/blog/how-to-find-all-duplicate-rows-in-a-pandas-dataframe/
# 
# To remove the duplicates, do this search: "how can I remove the duplicate rows in a DataFrame using Python"
# 
# This leads me to the following documentation.
# 
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html

# %%
df_bank_duplicates = df_bank_nafix_drop[df_bank_nafix_drop.duplicated()]
print("Duplicate Rows:\n", df_bank_duplicates)

# %%
df_bank_duplicate_indices = df_bank_duplicates.index.tolist()
print("Indices of duplicate rows:", df_bank_duplicate_indices)

# %%
# Remove duplicate rows
df_bank_nafix_no_duplicates = df_bank_nafix_drop.drop_duplicates()

df_bank_nafix_no_duplicates

# %% [markdown]
# 5. Check whether the customers all have unique AccountIDs.  If not, provide the first example of a non-unique AccountId.

# %% [markdown]
# Suggested Google search or ChatGPT prompt: "how can I find the first non-unique item from a pandas Series in python"
# 
# By the way: why didn't I ask the question "how can I check whether the customers all have unique AccountIDs"?
# 
# The problem would be that Google and ChatGPT don't know what "customers" you are talking about.  It's important to understand that the AccountIDs are a column of a DataFrame, and as such they are a Series.  Therefore, we should use the correct vocabulary and ask about a Series.  If you mess up and ask about a "list" instead of a Series, you _might_ get an answer that still works.  But it's better to get the vocabularly right.
# 
# It's important to add "in python" because this task could be performed in many languages.
# 
# ChatGPT gave me this suggestion: data[data.isin(data[data.duplicated()])].iloc[0]
# However, ChatGPT did not explain how this code worked and even claimed (falsely) that it was going to use the value_counts() function in the solution.  So although the code is correct, I personally found ChatGPT's answer very confusing.  You could, perhaps, ask ChatGPT to explain further how this code works.
# 
# ChatGPT, "How does this code work: data[data.isin(data[data.duplicated()])].iloc[0]"
# 
# On the other hand, Google leads me to the documentation for the duplicated() function:
# 
# https://pandas.pydata.org/docs/reference/api/pandas.Series.duplicated.html
# 
# Here, I can see that when I really need is data.duplicated(keep = False), where "data" should be the Series in question.  However, this just gives me a Series of boolean values indicating which ones are duplicates.  I have to somehow know that extracting the numerical values instead of a Series of booleans involves boolean indexing: data\[data.duplicated(keep = False)].
# 
# So as usual, I'd suggest that a combination of Google, documentation, and ChatGPT will give you the best information.

# %%
df_bank_duplicated_by_account_id = df_bank_nafix_fill[df_bank_nafix_fill.duplicated(subset=['AccountID'])]
df_bank_duplicated_by_account_id.sort_values(by=['AccountID']).head(10)

# %% [markdown]
# 6. Count how many distinct AccountIDs there are.

# %% [markdown]
# Suggested Google search or ChatGPT prompt: "how can I find out how many distinct items there are in a pandas Series using python"
# 
# This time Google provides me with a page that's specifically made to answer this question:
# 
# https://www.geeksforgeeks.org/how-to-count-distinct-values-of-a-pandas-dataframe-column/

# %%
distinct_count_account_id = len(df_bank_nafix_drop['AccountID'].unique())

print("Number of distinct items in column 'AccountID':", distinct_count_account_id)

# %%
distinct_count_all_columns = df_bank_nafix_drop.nunique()
print("Number of distinct items in each column: \n", distinct_count_all_columns)

# %% [markdown]
# 7. Remove the duplicate AccountIDs so that each AccountID appears only once.
# 
# This will involve using data.duplicated() but this time without keep = False.  We don't want to drop all duplicates; we want to leave one example of each value.

# %%
df_bank_nafix_no_duplicates_by_account_id = df_bank_nafix_drop[~df_bank_nafix_drop['AccountID'].duplicated(keep='first')]

df_bank_nafix_no_duplicates_by_account_id.nunique()

# %% [markdown]
# 8. What are the mean, median, and mode customer age in years?  (Rounding down to the next lower age.)
# Are there any outliers?  (Customers with very large or very small ages, compared with the other ages?)

# %% [markdown]
# Suggested Google search or ChatGPT prompt: "how can I find out the mean, median, and mode of a pandas Series"

# %%
def calculate_age(birth_date):
    today = datetime.today()
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    return age

df_bank_nafix_no_duplicates_by_account_id.loc[:, "Age"] = df_bank_nafix_no_duplicates_by_account_id["BirthDate"].apply(lambda x: calculate_age(x))

df_bank_nafix_no_duplicates_by_account_id.loc[:, "AccountAge"] = df_bank_nafix_no_duplicates_by_account_id["AccountOpened"].apply(lambda x: calculate_age(x))

df_bank_nafix_no_duplicates_by_account_id.head()

# %%
import math

max_age = df_bank_nafix_no_duplicates_by_account_id["Age"].max()
min_age = df_bank_nafix_no_duplicates_by_account_id["Age"].min()
print(f"Max age: {max_age}; Min age: {min_age}")

mean_age = math.floor(df_bank_nafix_no_duplicates_by_account_id["Age"].mean())
median_age = math.floor(df_bank_nafix_no_duplicates_by_account_id["Age"].median())
mode_age = math.floor(df_bank_nafix_no_duplicates_by_account_id["Age"].mode().iloc[0])
std_age = df_bank_nafix_no_duplicates_by_account_id["Age"].std()

print(f"Mean age: {mean_age}; Median: {median_age}; Mode: {mode_age}; Standard deviation: {std_age}")

# %%
lower_bound = 18 # leagal bound 
upper_bound = 80 # high risk bound
outliers_rule = df_bank_nafix_no_duplicates_by_account_id[(df_bank_nafix_no_duplicates_by_account_id["Age"] < lower_bound) | (df_bank_nafix_no_duplicates_by_account_id["Age"] > upper_bound)]

print("Outliers using Rule-Based Method and domain knowlage:")
outliers_rule.head()

# %%
df_bank_nafix_no_duplicates_by_account_id.loc[:, "Z_score"] = (df_bank_nafix_no_duplicates_by_account_id["Age"] - mean_age) / std_age
print(f"Z-scores: \n{df_bank_nafix_no_duplicates_by_account_id['Z_score'].head()}")

outliers_z = df_bank_nafix_no_duplicates_by_account_id[(df_bank_nafix_no_duplicates_by_account_id["Z_score"] > 1.5) | (df_bank_nafix_no_duplicates_by_account_id["Z_score"] < -1.5)]

outliers_z

# %%
account_oppened_before_birth = df_bank_nafix_no_duplicates_by_account_id[(df_bank_nafix_no_duplicates_by_account_id["AccountAge"] < 0)]
account_oppened_before_birth.head()

# %% [markdown]
# 9. One-hot encode the AccountType column.  This means creating a new "checking," "savings", and "cd" columns so that you can run machine learning algorithms.

# %%
df1 = df_bank_nafix_no_duplicates_by_account_id.copy()
df2 = df_bank_nafix_no_duplicates_by_account_id.copy()
one_hot = pd.get_dummies(df1["AccountType"])
df2 = df2.join(one_hot)
df2.iloc[0:5]

# %% [markdown]
# Now, change the cd, checking, and savings columns into integers.

# %%
df2["cd"] = df2["cd"].astype(int)
df2["checking"] = df2["checking"].astype(int)
df2["savings"] = df2["savings"].astype(int)
df2.head()

# %%
df2.info()

# %% [markdown]
# 10. Are there any other data values that do not seem right?  If not, give an example?

# %% [markdown]
# I don't think Google or ChatGPT alone will help you here.  To answer the question, look at the columns and think about what relationships they should have with each other.  For example, it seems reasonable to expect that BirthDate would be no earlier than 120 years ago (it's unlikely that a customer would be this old.)  Now we can ask Google:
# 
# "How can I find out how long ago a pandas date is"
# 
# Google provides this helpful link, although it is not exactly the solution - you'll have to work with it a bit:
# 
# https://stackoverflow.com/questions/26072087/pandas-number-of-days-elapsed-since-a-certain-date
# 
# If you check, I think you'll find that all dates are more recent than 120 years ago.  What about the AccountOpened columns?  I see some obviously wrong dates there just by looking at the first few rows.
# 
# Along those same lines, are there any birth dates that are too recent?  Do we think that any two year olds will have opened bank accounts?  How common do you think this is in real life?  How common is it in our data set?  Can you detect the two year olds opening bank accounts using just one column, or do you need two columns?

# %% [markdown]
# 11. Use Matplotlib and/or Seaborn to analyse the ages at which customers open their account.  Is there a connection between the year they are born vs. the age at which they open the account?  Graph this in whatever way you think is best.

# %% [markdown]
# I asked Google and ChatGPT: "How can I plot dates vs. dates in Matplotlib".  This gave me a hard time at first - I had to tell ChatGPT it was giving me the wrong information because it tried to plot dates vs. numbers.  Eventually, I found out that you plot dates vs. dates in the same way you'd plot numbers vs. numbers.
# 
# Think in terms of Storytelling With Data to plot these as best you can.  Once you've seen the result, try to think of the best way to plot the data so as to show the user what you want them to see.  Title the graph so as to display the lesson that you want the user to take away.
# Here are some options for the axes:
# 
# 1. A scatter or line plot: On the x-axis, the date they are born.  On the y-axis, the date they open the account.
# 2. A scatter or line plot: On the x-axis, the date they are born.  On the y-axis, the age in years at which they open the account.
# 3. A scatter or line plot: On the x-axis, they year (integer) they are born.  On the y-axis, the age in years at which they open the account.
# 4. A histogram: on the x-axis, the age at which they open the account.
# 
# Here is an example:

# %%
import matplotlib.pyplot as plt

ax = plt.gca() # get an "Axes" object to draw on; gca stands for "get current Axes"
ax.scatter(df2["BirthDate"], df2["AccountOpened"]) # create a scatter plot based on these two dates
ax.set_ylabel("Account Opened") # label the y axis
ax.set_xlabel("Birth Date") # label the x axis

# %% [markdown]
# # 4. Storytelling With Data graph

# %% [markdown]
# Choose any graph in the Introduction of Storytelling With Data.  Using matplotlib to reproduce it in a rough way.  I don't expect you to spend an enormous amount of time on this; I understand that you likely will not have time to re-create every feature of the graph.  However, if you're excited about learning to use matplotlib, this is a good way to do that.  You don't have to duplicate the exact values on the graph; just the same rough shape will be enough.  If you don't feel comfortable using matplotlib yet, do the best you can and write down what you tried or what Google searches you did to find the answers.

# %%
# Account Opened by Account Type
plt.figure(figsize=(8, 6))
for account_type in df2['AccountType'].unique():
    filtered = df2[df2['AccountType'] == account_type]
    plt.scatter(filtered['AccountOpened'], filtered['Age'], label=account_type)

plt.title("Account Opened vs Age by Account Type", fontsize=14)
plt.xlabel("Account Opened", fontsize=12)
plt.ylabel("Age", fontsize=12)
plt.legend(title="Account Type")
plt.grid(True)

plt.tight_layout()
plt.show()


