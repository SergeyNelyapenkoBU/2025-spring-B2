# %% [markdown]
# # Week 3 - Univariate Analysis
# 
# ## Please run the cells of the notebook as you get to them while reading

# %%
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import matplotlib.pyplot as plt

# %% [markdown]
# # 1. Lesson: Bar charts and univariate graphs

# %% [markdown]
# Let's make a dataset (in this case, just a series) which is weighted to have more small values than large values.  By squaring a random number between 0 and 1, we ensure that half (those whose initial value is below 0.5) are below 0.25, while the other half are between 0.25 and 1.  This means that most of the values are small, and it's more interesting than analyzing a perfectly uniform dataset.  This kind of trick - transforming one random variable to get another - can generate a variety of random datasets for you.  We then multiply by 100 to get a number between 0 and 100.

# %%
np.random.seed(0)
lesson_series = np.round(np.random.random(size = 1000)**2 * 100, 2)
lesson_series[0:10] # check the first ten values.  Are they mostly on the small side?

# %%
import seaborn as sns

# %% [markdown]
# In the plot below, you can see a histogram of the values in the series.  For some reason, it decided to have exactly 11 bins (we allowed it to choose the number of bins.)  Most values - about 300 of them - are between 0 and 9, and the next most likely bin is between 9 and 18.  Since there are 1000 values, the total of the bars should be 1000.

# %%
sns.histplot(lesson_series)

# %% [markdown]
# There are 11 bins or bars, a number which by default is chosen by seaborn.  We can reproduce this manually to (hopefully) see the same values as numbers. I'm not sure that this second histogram is guaranteed to be exactly the same, but it looks the same to me:

# %%
np.histogram(lesson_series, bins = 11)[0]

# %% [markdown]
# What happens if we override seaborn and choose the number of bins ourselves?  We could choose a much larger number of bins:

# %%
sns.histplot(lesson_series, bins = 50)

# %% [markdown]
# One disadvantage of this 50 bin picture is that the outliers are worse.  That is, the graph wobbles up and down a bit more randomly.  That's because there are fewer values in each bin, so there's more of a role for chance to take effect.  If we had many more data points and/or fewer bins, we could get rid of this wobble.

# %%
lesson_series_2 = np.round(np.random.random(size = 10000)**2 * 100, 2)
sns.histplot(lesson_series_2, bins = 50)

# %% [markdown]
# Here you can see that because the number of bins is the same as in the second graph above, but for more data, the histogram is a smoother graph.  Why does more data make for a smoother graph?  Something for you to think about.  I said it's because a larger number of data points in each bin reduces the role of chance - but why is that?

# %% [markdown]
# Here is a KDE (Kernel Density Estimate) plot.  It's just the same histogram, but drawn smoothly.  The KDE plot doesn't have a "number of bins."  It's always drawn the same way.  In this case, because of the smoothness of the curve, it seems that x-values less then zero and above 100 are still plotted, even though there were no such values in the dataset.  This seems like a drawback of the KDE plot, especially if the viewer is unprepared for this aspect of the plot.

# %%
sns.kdeplot(lesson_series)

# %% [markdown]
# We could also draw a box plot.  This time, to construct the data I used a fourth power rather than squaring, with only 100 data points, so that the points are even more concentrated toward the small numbers.  It turns out that this will create a more interesting boxplot.  The top and bottom edges of the box are the 75th and 25th percentile, respectively, and the top and bottom "whiskers" show a larger range which is a multiple of 1.5 times the the box height.  (The bottom whisker cannot be see because it's pushed against the bottom of the graph.)  The filled-in box shows that half of the values are between about 0 and 30 on the y-axis.  Is that what you'd expect?  The 25th and 75th percentile of the original uniform random variable are at 0.25 and 0.75.  Taken to the fourth power and multiplied by 100 (remember, that's how we constructed our sample), that's 0.25\*\*4 * 100 = 0.4 and 0.75\*\*4 * 100 = 32.  It's plausible that those are the height of the bottom and top of the box.  We can see that a small number of samples are above the top whisker; they are shown as individual dots.

# %%
lesson_series_3 = np.round(np.random.random(size = 100)**4 * 100, 2)
sns.boxplot(lesson_series_3)

# %% [markdown]
# If we go back to the original lesson_series with the squared values, there will be two whiskers, because it isn't so strongly weighted toward small values:

# %%
sns.boxplot(lesson_series)

# %% [markdown]
# Going back to the fourth power series, another histogram variant is the violin plot.  This simply combines a kde plot (turned on its side and forming two side of the violin) with a boxplot:

# %%
sns.violinplot(lesson_series_3)

# %% [markdown]
# Finally, a swarm plot shows the histogram (turned on its side and doubled, as with the violin plot) but showing each individual point.

# %%
sns.swarmplot(lesson_series_3)

# %% [markdown]
# # 2. Weekly graph question

# %% [markdown]
# Below are a histogram and table representation of the same data.  A species of bird is being analyzed, and each individual's body length in inches has been measured.
# 
# Please write a short explanation of the pros and cons of these two representations.  Which would you choose?  How would you modify the formatting, if at all, to make it more visually interesting, clear, or informative?

# %%
import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(0)
num_data = 10000
data = np.random.normal(size = num_data) + 6
df = pd.DataFrame(data.T, columns = ["data"])

# %%
histnums = np.histogram(df["data"])
histcounts = histnums[0]
histmins = histnums[1][0:-1]
histmaxes = histnums[1][1:]

# %%
pd.DataFrame(np.array([histcounts, histmins, histmaxes]).T, columns = ["count", "minval", "maxval"])

# %%
df.plot.hist()

# %% [markdown]
# # 3. Homework - Amusement Park Rides

# %% [markdown]
# Now let's imagine we have some data about how many times different visitors to an amusement park used each ride, as well as how much money they spend at the amusement park.  Each sample represents a single visit by a single visitor on a given date.

# %%
num_visits = 10000
np.random.seed(0)

# %%
df = pd.DataFrame(columns = ["VisitDate"])

# %%
start = datetime(2010, 1, 1)
end = datetime(2024, 1, 1)
numdays = (end - start).days
random_days = np.random.randint(0, numdays, size = num_visits)
s = start + pd.to_timedelta(random_days, unit='D')
s = s.sort_values()
df["VisitDate"] = s

# %%
df["IsAdult"] = np.random.choice([True, True, False], size = num_visits)

# %%
df["MartianRide"] = np.random.choice([0] * 8 + [1] * 3 + [2] * 3 + [3] * 1 + [10], size = num_visits) * df["IsAdult"]

# %%
df["TeacupRide"] = np.random.choice([0] * 2 + [1] * 5 + [2] * 3 + [5] * 2, size = num_visits) * ~df["IsAdult"]

# %%
df["RiverRide"] = np.random.choice([0] * 8 + [1] * 3 + [2] * 2, size = num_visits) * df["IsAdult"] + np.random.randint(1, 5, size = num_visits) * ~df["IsAdult"]

# %%
df["MoneySpent"] = np.round(np.random.random(size = num_visits)**2 * 100, 2)

# %%
df.iloc[0:5]

# %% [markdown]
# 1. Find the mean, median, and mode for how many times visitors rode each ride.  See Week 1 for Google advice on this.

# %%
df.describe()

# %%
print("Note: The 50 percentile is the same as the median.")

# %% [markdown]
# 2. Use groupby() to find the mean, median, and mode for how many times each ride was ridden on each given day.

# %% [markdown]
# Suggested Google search or ChatGPT prompt: 
# I first tried: "How do I use groupby to find the mean over each day in my DataFrame?" but Google didn't help me.
# 
# Then I tried: "How do I find the mean over each date in a dataframe?"  Sometimes, you have to try multiple searches.
# 
# This gives me a very helpful site, where someone is doing the same thing we are:
# 
# https://stackoverflow.com/questions/40788530/how-to-calculate-mean-of-some-rows-for-each-given-date-in-a-dataframe

# %%
df.groupby('VisitDate').mean()

# %%
df.groupby('VisitDate').median()

# %%
mode_of_visitdate = df['VisitDate'].mode().iloc[0]
print(f"Mode of VisitDate: {mode_of_visitdate}")

# %%
df.groupby('VisitDate')[['MartianRide','TeacupRide','RiverRide']].mean()

# %%
df.groupby('VisitDate')[['MartianRide','TeacupRide','RiverRide']].median()

# %%

numeric_columns = df.select_dtypes(include=['number'])
results = {}
for col in numeric_columns:
    mean_value = df[col].mean()
    median_value = df[col].median()
    mode_result = stats.mode(df[col].dropna(), nan_policy='omit', keepdims=True)
    mode_value = mode_result.mode[0] if mode_result.mode.size > 0 else None
    results[col] = {
        'Mean': mean_value,
        'Median': median_value,
        'Mode': mode_value
    }

summary_df = pd.DataFrame(results).T
print(summary_df)

# %% [markdown]
# 3. Find the standard deviation and variance of the count for each ride.
# 
# Suggested Google search or ChatGPT prompt: "How do I find the standard deviation of a Series in pandas documentation?" and similar query for variance.
# 
# I included the word "documentation" because this task likely involves applying one single function for standard deviation and another for variance, so I expected that I could find a single documentation page for each that would cover my needs.  I get these:
# 
# https://pandas.pydata.org/pandas-docs/dev/reference/api/pandas.Series.var.html
# 
# https://pandas.pydata.org/docs/reference/api/pandas.Series.std.html

# %%
df.std()

# %% [markdown]
# 4. Find the 90th percentile count for each ride.  That is, if the customers are ordered by their number of rides, and there are 100 customers, how many rides does the 90th person take?  There is a function in pandas that easily does this calculation.
# 
# Suggested Google search or ChatGPT prompt: "How do I find a percentile value for a Series in pandas documentation?"
# 
# This time, I will let you find the page!  Hint: it's not actually called the "percentile" function.

# %%
# 90th Percentile
percentile_90 = df[["MartianRide", "TeacupRide", "RiverRide"]].quantile(0.9)
print(percentile_90)


# %% [markdown]
# 5. Plot a histogram of the ride count, using each day as data element as you did in #2.  Use both the total ride count for each day as well as, separately, the mean ride count for each day.
# 
# Suggested Google search or ChatGPT prompt: "How do I plot a histogram for a Series in pandas?"
# 
# I found this: https://pandas.pydata.org/docs/reference/api/pandas.Series.plot.hist.html
# 
# Note: the "by" keyword will not help you plot a count for each day; if you try to use it for this, it will try to draw a separate histogram for each day.
# 
# You've already learned how to find the mean ride count for each day.  Can you use a similar idea to find the total ride count for each day?

# %%
# Histograms
# df.groupby("VisitDate")["MartianRide"].sum().plot.bar()

# %% [markdown]
# 6. Make a bar chart showing the total number of Adult and Child participants.
# 
# Try to formulate your own Google search or ChatGPT prompt.
# 
# Here's one page I found:
# 
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html
# 
# which shows some interesting examples for you to use.  I recommend clicking on the first, basic example to see how to use the function.
# 
# You can also look up the pandas documentation, which might lead to an entirely different method.

# %%


# %% [markdown]
# 7. Make a stacked bar chart showing how many Adult and Child visits took the River Ride, with the x-axis showing the number of rides and with two stacked bars (adult, child).  That is, there could be a column for the number of Adults (and Children) who took 0 rides, 1 ride, 2 rides, and so on.
# 
# Try to formulate your own Google search or ChatGPT prompt.
# 
# For me, the pandas documentation was easier than the matplotlib documentation.
# 
# * If you use the pandas method, you might have to one-hot encode the IsAdult column.  You might want names for the new columns other than True and False.  Use df.join or pd.concat to attach the pd.get_dummies table you've created, and then groupby the RiverRide value.
# 
# * You will need to end up with a small table with two columns (IsAdult True and False) and an index (RiverRide) as well as a small number of rows (the values of RiverRide)
# 
# * That said, if you can come up with another approach, it's fine.

# %% [markdown]
# 8. Other tasks
# 
# Use seaborn (import seaborn as sns) to create a kernel density estimation (kde) plot.  Here is a tutorial you can read about seaborn:
# 
# https://seaborn.pydata.org/tutorial/introduction.html
# 
# Seaborn often makes nicer looking graphs than pandas or matplotlib.
# 
# Now use seaborn to make a swarm plot, a violin plot, and a box plot.
# 
# Which plots are the best for showing this data?

# %% [markdown]
# # 4. Storytelling With Data graph
# 
# Try to make a scatterplot using amusement park dataset that is similar to the one on page 45, where the two axes are the Teacup Ride count and the River Ride count.  If you want to see individual dots and not a dense swarm of dots, you'll have to cut the dataset down to a small number of points (say, 100 points or so).  You can remove the other points, for instance, and focus on the first 100 points.
# 
# Here are some things you could do (you don't have to do all of them):
# 
# * Draw a dashed line that roughly separates the adult from child points.
# 
# * Draw the points in the adult vs. child region of the graph in different colors.
# 
# * Choose the right size and number of dots to make the graph look good.
# 
# * Write a word that appears on the dashed line (like AVG in the plot on page 45).
# 
# If there are any other graphs in the Storytelling With Data chapter that look interesting, and you want more practice, you can try to reproduce them too.

# %%
sample_df = df.sample(100, random_state=0)
sns.scatterplot(x=sample_df["TeacupRide"], y=sample_df["RiverRide"], hue=sample_df["IsAdult"].map({True: "Adult", False: "Child"}))
plt.axvline(sample_df["TeacupRide"].mean(), linestyle='dashed', color='red', label='AVG')
plt.legend()
plt.title("Scatter Plot: Teacup Ride vs River Ride")
plt.show()



