'''Real Python Data Science Article
https://realpython.com/python-for-data-analysis
We will use a file in VS Code instead of a Jupiter notebook
This will mean some slight changes to what was in the tutorial and
some other minor differences also'''
import pandas as pd

# Get the data
james_bond_data = pd.read_csv("james_bond_data_rp.csv").convert_dtypes()
# side part of the tutorial to scrape a web site
import ssl
# needed to prevent a certificate error
ssl._create_default_https_context = ssl._create_unverified_context
url = "https://en.wikipedia.org/wiki/List_of_James_Bond_novels_and_short_stories"
bond_books = pd.read_html(url)
book_data = bond_books[1].convert_dtypes()
print(james_bond_data.head())
print(book_data.head())

# Clean the data
# Rename columns
old_to_new_names = {
    "Release":"release_date",
    "Movie":"movie_title",
    "Bond":"bond_actor",
    "US_Gross":"income_usa",
    "World_Gross":"income_world",
    "Budget ($ 000s)":"budget_usd",
    "Film_Length":"film_length",
    "Avg_User_IMDB":"imdb_avg",
    "Avg_User_Rtn_Tom":"rotten_tomatoes_avg",
    "Martinis":"martinis_consumed",
    "Kills_Bond":"bond_kills"
}
data = james_bond_data.rename(columns=old_to_new_names)
print(data.columns)
# any missing data?
print(data.info())
print(data.loc[data.isna().any(axis="columns")])
# these ratings values were withheld to provide an example of missing
#  data, so they can be looked up in the original dataset and fixed
data.loc[10, "imdb_avg"] = 7.1
data.loc[10, "rotten_tomatoes_avg"] = 6.8
# I updated these a bit different to the tutorial so I don't have to
#  recreate the whole data frame
# Fix the types for currency amounts
data = data.assign(income_usa=lambda data: (
    data["income_usa"].replace("[$,]", "", regex=True).astype("Float64")))
data = data.assign(income_world=lambda data: (
    data["income_world"].replace("[$,]", "", regex=True).astype("Float64")))
data = data.assign(budget_usd=lambda data: (
    data["budget_usd"].replace("[$,]", "", regex=True).astype("Float64")) * 1000)
# the budget was in $1,000s so we fixed that at the same time
# TODO consider moving all this to the front of the pipeline
# Next fix the run length and retype
data = data.assign(film_length=lambda data: (
    data["film_length"].str.removesuffix(" mins").astype("Int64")))
# Now move to date types for release date
data = data.assign(release_date=lambda data: pd.to_datetime(
    data["release_date"], format="%B, %Y"))
data["release_year"] = data["release_date"].dt.year.astype("Int64")
# have a look
print(data.info())
print(data.head())
'''Fix the actor names. Again, only two items here that were probably
introduced intentionally to the practice dataset. If this were real, but
similarly limited to known good answers, we could try things like a K-nearest
with known good values as seeds or maybe a BURT vector or a structured LLM
responce as was outlined in the Data Sceptic podcast in December 'I LLM and
you can too'. For now, we will just correct with .replace as it is probably
valid to replace everywhere these occur'''
data.replace("Shawn Connery", "Sean Connery", inplace=True)
data.replace("Roger MOORE", "Roger Moore", inplace=True)
# check
print(data["bond_actor"].value_counts())
# car types
print(data["Bond_Car_MFG"].value_counts())
data.replace("Astin Martin", "Aston Martin", inplace=True)
# Test for reasonable values, fix and test again
print(data[["film_length", "martinis_consumed"]].describe())
data["martinis_consumed"].replace(-6, 6, inplace=True)
data["film_length"].replace(1200, 120, inplace=True)
print(data[["film_length", "martinis_consumed"]].describe())
# check for duplicates
data.loc[data.duplicated(keep=False)]
# drop them in place without needing another data frame and re-indexing
data.drop_duplicates(inplace=True, ignore_index=True)
# we should only reindex if we are not dependant on ordering after this
# Store the clean data to a new file without the index column
data.to_csv("james_bond_data_cleansed.csv", index=False)

# Performing analysis
import matplotlib.pyplot as plt
# first a plot
fig, ax = plt.subplots()
ax.scatter(data["imdb_avg"], data["rotten_tomatoes_avg"])
ax.set_title("Scatter Plot of Ratings")
ax.set_xlabel("Average IMDb Rating")
ax.set_ylabel("Average Rotten Tomatoes Rating")
fig.show()
# regression analysis
from sklearn.linear_model import LinearRegression
x = data.loc[:, ["imdb_avg"]]
y = data.loc[:, "rotten_tomatoes_avg"]
model = LinearRegression()
model.fit(x, y)
r_squared = f"R-Squared: {model.score(x, y):.2f}"
best_fit = f"y = {model.coef_[0]:.4f}x{model.intercept_:+.4f}"
y_pred = model.predict(x)
fig, ax = plt.subplots()
ax.scatter(x, y)
ax.plot(x, y_pred, color="red")
ax.text(7.25, 5.5, r_squared, fontsize=10)
ax.text(7.25, 7, best_fit, fontsize=10)
ax.set_title("Scatter Plot of Ratings")
ax.set_xlabel("Average IMDb Rating")
ax.set_ylabel("Average Rotten Tomatoes Rating")
fig.show()
# bar plot the run times
fig, ax = plt.subplots()
length = data["film_length"].value_counts(bins=7).sort_index()
length.plot.bar(
    ax = ax,
    title = "Film Length Distribution",
    xlabel = "Time Range (mins)",
    ylabel = "Count")
fig.show()
# print other run time stats
data["film_length"].agg(["min", "max", "mean", "std"])
# kills vs ratings
fig, ax = plt.subplots()
ax.scatter(data["imdb_avg"], data["bond_kills"])
ax.set_title("Scatter Plot of Kills vs Ratings")
ax.set_xlabel("Average IMDb Rating")
ax.set_ylabel("Kills by Bond")
fig.show()
# no clear relationship here
