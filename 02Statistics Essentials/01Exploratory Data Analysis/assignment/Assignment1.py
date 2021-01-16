# %% [markdown]
'''
## Spark Funds Data Analysis
Spark Funds wants to make investments in a few companies. The CEO of Spark Funds wants to understand the global trends in investments so that she can take the investment decisions effectively.
### Constraints for investments
- 5 to 15 million USD
- English-speaking countries
- Invest where most other investors are investing. This pattern is often observed among early stage startup investors
### The objective is to identify the best(where most investors are investing):
- sectors : eight 'main sectors'
- countries : most heavily invested
- a suitable investment type for making investments : investment amounts in the venture, seed, angel, private equity etc
### Investment amount increasing order
Seed/angel (Startup) ==> venture ==> Private equity

*Spark Funds wants to choose one of these four investment types for each potential investment they will make.*
'''

# %%
import csv
import functools
import operator
import string
from typing import Collection, Iterable, List, Sequence, Tuple

# %%
import enchant
import matplotlib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from colorama import Back, Fore, Style
from fuzzywuzzy import fuzz
from IPython.display import display
from matplotlib.pyplot import title
from nltk.corpus import words
from numpy.lib.arraysetops import isin, unique
from tqdm import tqdm

sns.set(style="whitegrid")
%matplotlib inline


# %%

default_ecoding = 'iso-8859-1'
dollar_formatting = '${0:,.2f}'
float_formatting = '{0:,.2f}'
int_formatting = '{0:,.0f}'

# Ref: https://www.kaggle.com/maunish/osic-super-cool-eda-and-pytorch-baseline/notebook
# used for changing color of text in print statement
y_ = Fore.YELLOW
r_ = Fore.RED
g_ = Fore.GREEN
b_ = Fore.BLUE
m_ = Fore.MAGENTA
sr_ = Style.RESET_ALL
# %%


def read_nonutf8_encoded_file(file_name, encoding='iso-8859-1'):
    return pd.read_csv(file_name, encoding=encoding)


# %%
companies_raw = read_nonutf8_encoded_file("companies.csv")
# %%
companies_raw.shape
# %%
companies_raw.head()
# %%
companies_raw.iloc[422]
# %% [markdown]
'''
We need to remove special chars
'''
# %%
companies_raw.isnull().any()
# %%
companies_raw.isnull().sum().sum()

# %% [markdown]
'''
Column wise nulls
'''
# %%
companies_raw.isnull().sum()
# %% [markdown]
'''
Nulls percentage
'''
# %%
((companies_raw.isnull().sum()*100)/companies_raw.shape[0])
# %% [markdown]
'''
 - founded_at is column with highest missing percentage which is ~23%
 - state_code,region and city are in ~12-13%
 - country_code,homepage_url, category_list and name as well have missing values
 - We will remove founded at, state code, region and country columns
 - We will remove the rows with missing values in name, category list and home URL columns because they have around 10% missing values

'''
# %%
companies_raw.info()

# %%
# column names
name_c = 'name'
name_c_clean = 'name_clean'
permalink = 'permalink'
permalink_clean = 'permalink_clean'
homepage_url = 'homepage_url'
category_list = 'category_list'
category_list_clean = 'category_list_clean'
status_c = 'status'
country_code_c = 'country_code'
country_code_c_clean = 'country_code_clean'
primary_sector = 'primary_sector'
main_sector = 'main_sector'


company_permalink = 'company_permalink'
company_permalink_clean = 'company_permalink_clean'
funding_round_type = 'funding_round_type'
raised_amount_usd = 'raised_amount_usd'

# %%
companies_raw[name_c].value_counts()
# %%
companies_raw[permalink].value_counts()

# %%
companies_raw[permalink].apply(lambda x: str(x).split("/")[1]).unique()
# %% [markdown]
'''
As we can see that we have only one prefix "Organization" we can remove it if required
'''
# %%
companies_raw[status_c].unique()

# %% [markdown]
'''
We need to remove closed and acquired companies as we can not invest into them
'''
# %%
rounds2_raw = read_nonutf8_encoded_file("rounds2.csv")

# %%
rounds2_raw.head()

# %% [markdown]
'''
we'll keep only company_permalink, funding_round_type, raised_amount_usd and drop funding_round_permalink, funding_round_code, funded_at as these are not required in our analysis
'''
# %%
rounds2_raw.info()
# %%
rounds2_raw.shape
# %%
rounds2_raw.isnull().sum().sum()

# %% [markdown]
'''
Column wise nulls
'''
# %%
rounds2_raw.isnull().sum()
# %% [markdown]
'''
Nulls percentage
'''
# %%
((rounds2_raw.isnull().sum()*100)/rounds2_raw.shape[0])
# %% [markdown]
'''
 - funding_round_code column should be removed and the rows with null values in raised_amount_usd column
'''
# %%
rounds2_raw.dtypes
# %%
rounds2_raw[raised_amount_usd].describe()
# %% [markdown]
'''
 - We can see that all values are positive
'''
# %%
rounds2_raw[raised_amount_usd].plot(kind='box')
# %% [markdown]
'''
 - Values are extremely skewed
'''

# %% [markdown]
# ## Checkpoint 1: Data Cleaning 1

# %% [markdown]
'''
Cleaning the compamies dataframes
'''
# %%
alphanumeric = string.ascii_letters + string.digits


def clean_permalink(permalink: str) -> str:
    """cleans and standardize the permalink
    Args:
        permalink (str): raw permalink

    Returns:
        str: cleaned and lowercase permalink
    """
    valid_chars = alphanumeric+"/-"
    return (''.join([ch for ch in permalink if ch in valid_chars])).lower()
    # return permalink.encode('utf-8').decode('ascii', 'ignore').lower()


def clean_company_name(company_name: str) -> str:
    """cleans and standardize the company name

    Args:
        permalink (str): raw company name

    Returns:
        str: cleaned and lowercase company name
    """
    # Ref: https://companieshouse.blog.gov.uk/2019/02/14/symbols-and-characters-in-a-company-name/
    valid_chars = alphanumeric+".-(),&@£$€¥#♥;: "
    return (''.join([ch for ch in company_name if ch in valid_chars])).lower()


def clean_category_list(category_list: str) -> List[str]:
    """separates, cleans and standardize the category list

    Args:
        permalink (str): raw category list

    Returns:
        str: separated, cleaned and lowercase category list
    """
    # strip and split
    category_list = [c.strip() for c in str(category_list).strip().split("|")]
    # Remove unwanted characters
    category_list = list(map(lambda cn: ''.join(
        [ch for ch in cn.lower() if ch in string.ascii_letters]), category_list))
    return category_list


def clean_companies_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans companies dataframe

    Args:
        df (pd.DataFrame): Raw companies dataframe

    Returns:
        pd.DataFrame: Cleaned companies dataframe
    """
    # Fix columns
    # dropping homepage_url as I don't see significance
    df = df[[
        permalink, name_c, category_list, status_c, country_code_c]]

    # Fix rows with missing values
    df = df[df[name_c].notnull()]
    df = df[df[category_list].notnull()]
    df = df[df[country_code_c].notnull()]
    df = df[df[status_c].isin(['operating', 'ipo'])]

    # Standardize and invalid values
    df[permalink_clean] = df[permalink].apply(clean_permalink)
    df[name_c_clean] = df[name_c].apply(clean_company_name)
    # Lets make country list a clean list
    df[category_list_clean] = df[category_list].apply(
        clean_category_list)
    return df


# %%
companies = clean_companies_data(companies_raw)
# companies_frame.to_csv("companies_frame.csv", encoding=default_ecoding, quoting=csv.QUOTE_ALL, index=False)

# How many unique companies are present in companies?
# %%
unique_companies = companies[permalink_clean].unique()
print(
    f"Companies:Unique companies before cleaning:{len(companies_raw[permalink].unique())}")
print(f"Companies:Unique companies after cleaning:{len(unique_companies)}")

# %%
companies.info()
# %% [markdown]
'''
Cleaning the rounds2 dataframe
'''
# %%


def clean_rounds2_frame(df: pd.DataFrame) -> pd.DataFrame:
    """ Cleans Rounds2 dataframe

    Args:
        df (pd.DataFrame): Raw Rounds2 dataframe

    Returns:
        pd.DataFrame: Cleaned Rounds2 dataframe
    """
    df = df[[company_permalink, funding_round_type, raised_amount_usd]]
    df[company_permalink_clean] = df[company_permalink].apply(clean_permalink)
    # Filtering
    df = df[df[raised_amount_usd].notnull()]
    # pick only compamies which are present in cleaned compamies frame
    df = df[df[company_permalink_clean].isin(unique_companies)]
    return df


# %%
rounds2 = clean_rounds2_frame(rounds2_raw)
# rounds2.to_csv("rounds2_frame.csv", encoding=default_ecoding,quoting=csv.QUOTE_ALL, index=False)


# %%


def print_information() -> None:
    """Prints answer to few basic questions"""
    print(
        f"Rounds2:Unique companies before cleaning:{len(rounds2_raw[company_permalink].unique())}")
    print(
        f"Rounds2:Unique companies after cleaning:{len(rounds2[company_permalink_clean].unique())}")
    t1 = companies_raw[permalink].str.lower().unique().shape[0]
    t2 = rounds2_raw[company_permalink].str.lower().unique().shape[0]
    print(f"Total companies in original companies file: {t1}")
    print(f"Total companies in original rounds2 file: {t2}")
    print(f"Compamies count comparision before cleaning: {abs(t1-t2)}")
    t1 = companies[permalink_clean].str.lower().unique().shape[0]
    t2 = rounds2[company_permalink_clean].str.lower().unique().shape[0]
    print(f"Total companies in cleaned companies data: {t1}")
    print(f"Total companies in cleaned rounds2 data: {t2}")
    print(f"Compamies count comparision after cleaning: {abs(t1-t2)}")
    print(
        f"companies count diff in companies frame but not in rounds2 frame: {len(set(companies[permalink_clean]) - set(rounds2[company_permalink_clean]))}")
    print(
        f"companies count diff in rounds2 frame but not in companies frame: {len(set(rounds2[company_permalink_clean])-set(companies[permalink_clean]))}")


# %%
print_information()
# %% [markdown]
'''
- We can see that all the companies which are present in rounds2 are also present in companies frame
- In the companies data frame, which column can be used as the unique key for each company? Write the name of the column.
  - **permalink**
- How many unique companies are present in rounds2?
  - **Unique companies before cleaning:90247**
  - **Unique companies after cleaning:40470**
- Are there any companies in the rounds2 file which are not present in companies? Answer yes or no: Y/N
  - **Y**
'''
# %%


def flat_2d_list(lst: Iterable[Iterable]) -> Iterable:
    return functools.reduce(operator.iconcat, lst, [])


def plot_series(series: pd.Series, title: str, xlabel: str, ylabel: str, display_format: str = '{0:,.0f}', x_rotation=0, y_rotation=0) -> None:
    """Bar plots a interger series

    Args:
        series (pd.Series): interger series
    """
    ax = sns.barplot(x=series.index, y=series)
    plt.xticks(rotation=x_rotation)
    plt.yticks(rotation=y_rotation)
    # Ref: https://github.com/mwaskom/seaborn/issues/1582
    for i, p in enumerate(ax.patches):
        ax.annotate(display_format.format(series.iloc[i]), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 4), textcoords='offset points')
    plt.show()


def plot_and_print_companies_information():
    company_duplication_count_by_name = companies[name_c_clean].value_counts(
    ).value_counts()
    display(company_duplication_count_by_name)
    plot_series(company_duplication_count_by_name, "Company duplicate count by name",
                "Duplicate Count", "Company Count")
    flattened_categories = pd.Series(flat_2d_list(
        companies[category_list_clean])).value_counts().nlargest(5)
    display(flattened_categories)
    plot_series(flattened_categories, "Top 5 categories", "Category", "Count")
    top10_countries_forcompany_count = companies[country_code_c].value_counts(
    ).nlargest(10)
    display(top10_countries_forcompany_count)
    plot_series(top10_countries_forcompany_count,
                "Top 10 countries with most companies", "Country", "Company Count")


# %%
plot_and_print_companies_information()
# %% [markdown]
'''
### In above results and plots we can see that
 - in companies file (for name field)
   - 48034 companies have single entry
   - 143 companies are duplicated once
   - 7 companies are duplicated thrice
   - 3 companies are duplicated four times
     - Note: We can analyse why and which companies are duplicated names
 - software(6635), mobile(4126), biotechnology(3784), ecommerce(3284), socialmedia(2040) are the top fields where companies are working now a days
 - USA(29845), GBR(3109), CAN(1567), IND(1440), CHN(1413), FRA(952), DEU(878), ISR(789), ESP(646), AUS(428) are top 10 contries with highest companies
'''

# %% [markdown]
'''
### Company name duplication investigation
'''
# %%
company_names_with_duplicated_entries = (
    companies[name_c_clean].value_counts() > 1).index
duplicated_companies = companies[companies[name_c_clean].isin(
    company_names_with_duplicated_entries)]
duplicated_companies.to_csv(
    "duplicated_companies.csv", index=False, quoting=csv.QUOTE_ALL)

# %% [markdown]
'''
- It turns out those are really different companies
'''
# %%
rounds2.columns
# %%


def print_stats(ser_data: pd.Series) -> None:
    # https://www.kaggle.com/maunish/osic-super-cool-eda-and-pytorch-baseline/notebook
    print(f"{y_}Max value: {ser_data.max()} \n{r_}Min value: {ser_data.min()}\n{g_}Mean: {round(ser_data.mean(),2)}\n{b_}Standard Deviation: {round(ser_data.std(),2)}")


def plot_investmet_count_distribution(ser_data: pd.Series, color: str, title: str, xlabel: str, ylabel: str, show_stats=True) -> None:
    plt.figure(dpi=100)
    ax = sns.distplot(ser_data, color=color)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    # https://html.developreference.com/article/13223515/Matplotlib+float+values+on+the+axis+instead+of+integers
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    if show_stats:
        print_stats(ser_data)
    plt.show()


# %%
plot_investmet_count_distribution(
    rounds2[company_permalink_clean].value_counts(),
    'blue',
    "Investmet count distribution",
    "Investment counts",
    ""
)
# %%


def plot_investmet_distribution(ser_data: pd.Series, color: str, title: str, xlabel: str, ylabel: str, show_stats=True) -> None:
    plt.figure(dpi=100)
    ax = sns.distplot(ser_data, color=color)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if show_stats:
        print_stats(ser_data)
    plt.show()
# %%


plot_investmet_distribution(
    np.log(rounds2[rounds2[raised_amount_usd] > 0][raised_amount_usd]),
    'red',
    "Raised amount distribution",
    "Raised amount",
    ""
)
# %% [markdown]
'''
- We can see that data is highly skewed
'''
plot_investmet_distribution(
    np.log(rounds2[rounds2[raised_amount_usd] > 0][raised_amount_usd]),
    'red',
    "Raised amount distribution on log scale",
    "Raised amount(log scale)",
    ""
)
# %% [markdown]
'''
- On log scale it looks like normal distribution, ie original distribution is power distribution
'''
# %% [markdown]
'''
### Lets do box and boxen plot 
'''
# %%


def plot_investmet_boxplot(ser_data: pd.Series, color: str, title: str, xlabel: str, ylabel: str, show_stats=True, is_boxen=False) -> None:
    plt.figure(dpi=100)
    if is_boxen:
        ax = sns.boxenplot(ser_data, color=color)
    else:
        ax = sns.boxplot(ser_data, color=color)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if show_stats:
        print_stats(ser_data)
    plt.show()


# %%
plot_investmet_boxplot(
    rounds2[rounds2[raised_amount_usd] > 0][raised_amount_usd],
    'red',
    "Raised amount distribution",
    "Raised amount",
    ""
)
# %%
plot_investmet_boxplot(
    np.log(rounds2[rounds2[raised_amount_usd] > 0][raised_amount_usd]),
    'red',
    "Raised amount distribution on log scale",
    "Raised amount(log scale)",
    ""
)
# %%
plot_investmet_boxplot(
    rounds2[rounds2[raised_amount_usd] > 0][raised_amount_usd],
    'red',
    "Raised amount distribution",
    "Raised amount",
    "",
    is_boxen=True
)
# %%
plot_investmet_boxplot(
    np.log(rounds2[rounds2[raised_amount_usd] > 0][raised_amount_usd]),
    'red',
    "Raised amount distribution on log scale",
    "Raised amount(log scale)",
    "",
    is_boxen=True
)
# %% [markdown]
'''
- We can see that investments count is externally skewed to right
'''
# %%


def print_and_display_funding_types_data() -> None:
    funding_type_value_counts = rounds2[funding_round_type].value_counts()
    display(funding_type_value_counts)
    plt.figure(figsize=(15, 7))
    plot_series(funding_type_value_counts, title="Funding Type Distribution",
                xlabel="Funding Type", ylabel="Investment Count", x_rotation=45)


# %%
print_and_display_funding_types_data()


# %% [markdown]
'''
## Merge the two data frames so that all variables (columns) in the companies frame are added to the rounds2 data frame. Name the merged frame master_frame. How many observations are present in master_frame?
'''
# %%
master_frame = companies.merge(
    rounds2,
    left_on=permalink_clean,
    right_on=company_permalink_clean,
    how="inner"
)
master_frame.shape[0]

# %% [markdown]
'''
Lets check for nulls now
'''
# %%
master_frame.isnull().sum()

# %% [markdown]
'''
As we can see that now we have clean data, we can proceed further
'''
# %%
# master_frame.to_csv('master_frame.csv', index=False, quoting=csv.QUOTE_ALL)

# %% [markdown]
'''
### Checkpoint 2: Funding Type Analysis
- 1. Calculate the most representative value of the investment amount for each of the four funding types (venture, angel, seed, and private equity) and report the answers in Table 2.1
- 2. Based on the most representative investment amount calculated above, which investment type do you think is the most suitable for Spark Funds?
**Considering that Spark Funds wants to invest between 5 to 15 million USD per  investment round, which investment type is the most suitable for them?**
'''
# %%


def get_average_invested_amounts_funding_round_type() -> pd.DataFrame:
    return master_frame[[funding_round_type, raised_amount_usd]].groupby(by=funding_round_type).mean(
    ).reset_index().sort_values(by=raised_amount_usd, ascending=False)


def print_mean_of_and_most_suited_investment_type() -> None:
    average_invested_amounts = get_average_invested_amounts_funding_round_type()
    # Ref : https://pbpython.com/styling-pandas.html
    format_dict = {raised_amount_usd: '${0:,.2f}'}
    display(average_invested_amounts.style.format(format_dict).hide_index())
    display(average_invested_amounts[((5000000 <= average_invested_amounts[raised_amount_usd]) & (average_invested_amounts[raised_amount_usd]
                                                                                                  <= 15000000))].style.format(format_dict).hide_index())


# %%
print_mean_of_and_most_suited_investment_type()
# %% [markdown]
'''
### The best investments type will be : venture
'''
# %%

unfiltered_master_frame = master_frame.copy()

# %% [markdown]
'''
Identify the investment type and, for further analysis, filter the data so it only contains the chosen investment type.
'''
# %%
master_frame.shape
# %%
master_frame = master_frame[master_frame[funding_round_type] == 'venture']

# %%
master_frame.shape
# %% [markdown]
'''
Ref
 - https://worldpopulationreview.com/country-rankings/english-speaking-countries
 - https://raw.githubusercontent.com/stefangabos/world_countries/master/data/en/countries.csv
 - https://github.com/stefangabos/world_countries
'''
# %%


def read_english_speaking_country_data() -> None:
    countries = pd.read_csv("countries.csv")
    countries['name_l'] = countries['name'].str.lower()
    eng_countries = pd.read_csv("EnglishSpeakingCountries.csv")
    eng_countries = eng_countries['country'].str.lower()
    return countries, eng_countries


def map_eng_rank_to_ccode(country_to_match, match_series):
    res = list(filter(lambda t: fuzz.token_set_ratio(
        country_to_match, t) > 90, match_series))
    return True if res else False


def add_english_speaking_flag_master_frame() -> None:
    countries, eng_countries = read_english_speaking_country_data()
    countries['IsEnglish'] = countries['name_l'].apply(
        lambda n: map_eng_rank_to_ccode(n, eng_countries))
    countries['alpha3_u'] = countries['alpha3'].str.upper()
    eng_countries_filtered = list(
        countries[countries['IsEnglish']]['alpha3_u'])
    master_frame['IsEnglish'] = master_frame['country_code'].apply(
        lambda cc: cc in eng_countries_filtered)


# %%
add_english_speaking_flag_master_frame()


# %%[markdown]
'''
## Checkpoint 3: Country Analysis
'''

# %%


def get_top_n_highest_invested_countries(top_n: int) -> pd.DataFrame:
    return master_frame[master_frame['IsEnglish'] == True].groupby(country_code_c)[raised_amount_usd].sum(
    ).reset_index().sort_values(by=raised_amount_usd, ascending=False)[:top_n]


def print_highest_invested_countries(top_n_data: pd.DataFrame) -> None:
    avg_invested_amt_name = 'Average Invested Amount'
    top_n_data.columns = ['Country Code', avg_invested_amt_name]
    display_style = {avg_invested_amt_name: '${0:,.2f}'}
    display(top_n_data.style.format(display_style).hide_index())


# %% [markdown]
'''
**Spark Funds wants to see the **top nine** countries which have received the highest total funding (across ALL sectors for the chosen investment type)**
'''
# %%
top9 = get_top_n_highest_invested_countries(9)
print_highest_invested_countries(top9)
# %% [markdown]
# Identify the top three English-speaking countries in the data frame top9.
# %%
top3 = top9[:3]
print_highest_invested_countries(top3)

# %% [markdown]
# ## Checkpoint 4: Sector Analysis 1
# %% [markdown]
'''
### Extracting the primary sector of each category list from the category_list column
'''
# %%


def extract_primary_sector() -> None:
    master_frame[primary_sector] = master_frame[category_list].str.split("|").apply(
        lambda c: (c[0].strip()))


# %%
extract_primary_sector()
# %%
mapping = read_nonutf8_encoded_file('mapping.csv')
mapping.head(25)


# %% [markdown]
'''
**We can see that there are Nulls and spelling mistakes in above data like Alter0tive should be Alternative, and A0lytics should be Analytics**
**It seems someone replace na with 0, so lets run the spelling check on category_list column**
'''
# %%


def get_incorrect_words() -> None:
    data = mapping[mapping[category_list].notna()][category_list]
    allwords = set(flat_2d_list(data.str.split()))
    d = enchant.Dict("en_US")
    invalid_words = set([w for w in allwords if not d.check(w)])
    return invalid_words


# %%
get_incorrect_words()
# %% [markdown]
'''
**Now its confirmed that other than na->0 we don't have any other spelling mistakes, sor lets replace 0->na**

'''

# %%


def replace_na(str_value: str, ch: str = "0") -> str:
    """replaces \"0\" with na, specifically designed for category list, may not work for others need

    Args:
        str_value (str): category list
        ch (str, optional): Replacemet char. Defaults to "0".

    Returns:
        str: clean cotegory name
    """
    if str_value is not None:
        len_str = len(str_value)
        if len_str > 0:
            if str_value == "0":
                return "na"
            all_indices = [i for i, ltr in enumerate(str_value) if ltr == ch]
            if all_indices:
                for i in all_indices:
                    if i == 0 and str_value[1].isalpha():
                        str_value = "na"+str_value[1:]
                    elif i == (len_str - 1) and (str_value[len_str-2].isalpha() or str_value[len_str-2] != "."):
                        str_value = str_value[:len_str] + "na"
                    elif str_value[len_str-2] != ".":
                        str_value = str_value[:i] + "na" + str_value[(i+1):]
    return str_value


# %%


def clean_mappings(mapping: pd.DataFrame) -> pd.DataFrame:
    mapping = mapping[mapping[category_list].notna()]
    mapping[category_list] = mapping[category_list].apply(replace_na)
    return mapping

# %%


# %%
mapping = clean_mappings(mapping)

# %%
get_incorrect_words()

# %%
# mapping.to_csv("mapping_frame.csv", index=False, quoting=csv.QUOTE_ALL)
# %% [markdown]
'''
We don't have incorrect words in category list
'''

# %%


def add_main_sector_to_master_frame(master_frame: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    melted_mapping = mapping.melt(id_vars=category_list)
    melted_mapping = melted_mapping[melted_mapping['value'] == 1]
    melted_mapping.drop('value', axis=1, inplace=True)
    melted_mapping.columns = [category_list, main_sector]

    master_frame = master_frame.merge(
        melted_mapping, left_on=primary_sector, right_on=category_list, how='inner')
    master_frame.drop('category_list_y', axis=1, inplace=True)
    return master_frame


# %%
master_frame = add_main_sector_to_master_frame(master_frame, mapping)

# %% [markdown]
# ## Checkpoint 5: Sector Analysis 2
# %%
# %%
master_frame.head()

# %% [markdown]
# #### TOP 3 COUNTRIES

# %%
USA = "USA"
GBR = "GBR"
IND = "IND"

# %%


def filter_data_for_c5(country):
    return master_frame[
        (
            # TODO: Could be done by dict based filtering for nice looking code
            (master_frame[country_code_c].notnull()) &
            (master_frame[country_code_c].notna()) &
            (master_frame[country_code_c] == country) &
            (master_frame[funding_round_type].notnull()) &
            (master_frame[funding_round_type].notna()) &
            (master_frame[funding_round_type] == "venture") &
            (master_frame[raised_amount_usd].notnull()) &
            (master_frame[raised_amount_usd].notna()) &
            (master_frame[raised_amount_usd] >= 5000000) &
            (master_frame[raised_amount_usd] <= 15000000)
        )]


# %% [markdown]
#  "venture"
D1 = filter_data_for_c5(USA)
D2 = filter_data_for_c5(GBR)
D3 = filter_data_for_c5(IND)
# %%
D1.head()
# %%
D2.head()

# %%
D3.head()
# %% [markdown]
# Total number of investments (count)
# %%
print(D1.shape[0], D2.shape[0], D3.shape[0])
# %% [markdown]
# Total amount of investment (USD)
# %%
print(D1[raised_amount_usd].sum(),
      D2[raised_amount_usd].sum(), D3[raised_amount_usd].sum())

# %%


def print_investment_status(curated_data):
    temp_frame = pd.pivot_table(curated_data, values=raised_amount_usd,
                                index=main_sector, aggfunc=['count', 'sum']).reset_index()
    temp_frame.columns = [main_sector,
                          'raised_amount_usd_count', 'raised_amount_usd_sum']
    return temp_frame


# %%
D11 = print_investment_status(D1)
D22 = print_investment_status(D2)
D33 = print_investment_status(D3)

# %%
format_dict = {'raised_amount_usd_sum': '${0:,.2f}'}

# %%


def print_main_sector_in_order() -> None:
    # Ref: https://stackoverflow.com/questions/61363712/how-to-print-a-pandas-io-formats-style-styler-object
    for rd in [(USA, D11), (GBR, D22), (IND, D33)]:
        print(f"Country: {rd[0]}")
        print("\t Sorted by Investment Count")
        display(rd[1].sort_values(by='raised_amount_usd_count',
                                  ascending=False).style.format(format_dict).hide_index())
        # print("\n\t Sorted by Investment Sum")
        # display(rd[1].sort_values(by='raised_amount_usd_sum',
        #                         ascending=False).style.format(format_dict).hide_index())


# %%
print_main_sector_in_order()
# %%
# Print complete frame
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)


# %%
def get_highest_invested_companies() -> List[Tuple[int, str, List[str], pd.DataFrame]]:
    highest_invested_companies = []
    for df, sectors, c in [(D1, ['Social, Finance, Analytics, Advertising', 'Others'], USA),
                           (D2, [
                            'Others', 'Social, Finance, Analytics, Advertising'], GBR),
                           (D3, ['Others', 'Social, Finance, Analytics, Advertising'], IND)]:
        for i, sector in enumerate(sectors):
            sector_df = df[df[main_sector] == sector]
            highest_invested_value = sector_df[raised_amount_usd].max()
            sector_df = sector_df[sector_df[raised_amount_usd]
                                  == highest_invested_value]
            sector_df.drop_duplicates(subset=name_c, keep=False, inplace=True)
            names = list(sector_df[name_c])
            highest_invested_companies.append((i, c, names, sector_df))
            sector_df = None
            names = None
    return highest_invested_companies


# %%
highest_invested_companies = get_highest_invested_companies()
# %%
for _, c, companies, _ in highest_invested_companies:
    print(f"{c} : {len(companies)}")
# %%
for _, c, _, df in highest_invested_companies:
    print(c)
    display(df)
# %%
for _, c, companies, _ in highest_invested_companies:
    print(f"{c} : {len(companies)}")
# %%
for i, c, companies, _ in highest_invested_companies:
    print(
        f"\nCompany(ies) for \"{c}\" {'top sector count-wise' if i==0 else 'second best sector count-wise'}\n")
    print(', '.join(companies))

# %%


def get_top_invested_companies_by_main_sector(country_sector_map: List[Tuple[str, List[str]]], n: int) -> pd.DataFrame:
    def get_country_data(country_code: str, sector: str) -> pd.DataFrame:
        return master_frame[
            ((master_frame[country_code_c] == country_code) &
             (master_frame[funding_round_type] == 'venture') &
                (master_frame[main_sector] == sector)
             )].nlargest(n, raised_amount_usd)

    cols = [name_c, main_sector, raised_amount_usd, country_code_c]
    res_df = pd.DataFrame(columns=cols)
    for country, sectors in country_sector_map:
        for sector in sectors:
            res_df = res_df.append(get_country_data(country, sector)[cols])
    return res_df


# %%
get_top_invested_companies_by_main_sector([(USA, ['Social, Finance, Analytics, Advertising', 'Others']),
                                           (GBR, [
                                            'Others', 'Social, Finance, Analytics, Advertising']),
                                           (IND, ['Others', 'Social, Finance, Analytics, Advertising'])], 1)

# %% [markdown]
'''
# Checkpoint 6: Plots
'''
# %%

# %% [markdown]
'''
A plot showing the fraction of total investments (globally) in angel, venture, seed, and private equity, and the average amount of investment in each funding type. This chart should make it clear that a certain funding type (FT) is best suited for Spark Funds.
'''
# %%


def get_global_grouped_data() -> pd.DataFrame:
    global_total_fund = unfiltered_master_frame[raised_amount_usd].sum()
    global_grouped_data = unfiltered_master_frame[[funding_round_type, raised_amount_usd]].groupby(funding_round_type).agg(
        {raised_amount_usd: ['sum', 'mean']}
    ).reset_index()

    global_grouped_data.columns = [funding_round_type,
                                   'raised_amount_usd_sum', 'raised_amount_usd_mean']
    global_grouped_data['raised_amount_ratio'] = global_grouped_data['raised_amount_usd_sum']/global_total_fund
    global_grouped_data.sort_values(
        'raised_amount_ratio', ascending=False, inplace=True)
    return global_grouped_data


# %%
global_grouped_data = get_global_grouped_data()

# %%

k = global_grouped_data.style.format({'raised_amount_usd_mean': '${0:,.2f}',
                                      'raised_amount_usd_sum': '${0:,.2f}',
                                      'raised_amount_ratio': '{0:.2f}'
                                      }).hide_index()
display(k)
# %%


def plot_average_fund_type_share() -> None:
    graph_data = global_grouped_data[global_grouped_data[funding_round_type].isin(
        ['angel', 'venture', 'private_equity'])]

    g = sns.barplot(x="funding_round_type",
                    y='raised_amount_ratio', data=graph_data)
    g.set(xlabel='Funding Round Type',
          ylabel='Investment Proportion', title='Fund Type Share with Average Global Investment')
    # Ref: https://github.com/mwaskom/seaborn/issues/1582
    for i, p in enumerate(g.patches):
        g.annotate('${0:,.2f}'.format(graph_data.iloc[i]['raised_amount_usd_mean']), (p.get_x(
        ) + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 4), textcoords='offset points')
    plt.show()


# %%
plot_average_fund_type_share()
# %% [markdown]
'''
A plot showing the top 9 countries against the total amount of investments of funding type FT. This should make the top 3 countries (Country 1, Country 2, and Country 3) very clear.
'''
# %%
avg_invested_amt = "Average Invested Amount"
country_code_col_name = "Country Code"
graph_data = top9.sort_values(by=avg_invested_amt, ascending=False)
# %%


def plot_top_9_inested_countries(graph_data: pd.DataFrame) -> None:
    plt.figure(figsize=(15, 7))
    g = sns.barplot(x=country_code_col_name,
                    y=avg_invested_amt, data=graph_data)
    g.set(xlabel='Country',
          ylabel='Total Investment', title='Top 9 Countries with highest investments')
    for i, p in enumerate(g.patches):
        g.annotate('${0:,.2f}'.format(graph_data.iloc[i][avg_invested_amt]), (p.get_x(
        ) + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 4), textcoords='offset points')
    plt.show()


# %%
plot_top_9_inested_countries(graph_data)
# %%


def plot_top_9_inested_countries_on_log_scale() -> None:

    graph_data['raised_amount_usd_log'] = np.log(graph_data[avg_invested_amt])
    g = sns.barplot(x=country_code_col_name,
                    y="raised_amount_usd_log", data=graph_data)
    g.set(xlabel='Country',
          ylabel='Total Investment (Log Scale)', title='Top 9 Countries with highest investments')
    plt.show()


# %%
plot_top_9_inested_countries_on_log_scale()
# %% [markdown]
'''
A plot showing the number of investments in the top 3 sectors of the top 3 countries on one chart (for the chosen investment type FT).
'''
# %%


def plot_sector_wise_investment_comparision() -> None:
    usa_data = D1[main_sector].value_counts()[:3]
    gbr_data = D2[main_sector].value_counts()[:3]
    ind_data = D3[main_sector].value_counts()[:3]

    graph_data = pd.DataFrame(
        {"Sector": usa_data.index, "Count": usa_data.values,
            "Country": [USA for _ in range(len(usa_data))]}
    ).append(pd.DataFrame({"Sector": gbr_data.index, "Count": gbr_data.values, "Country": [GBR for _ in range(len(gbr_data))]})) \
        .append(pd.DataFrame({"Sector": ind_data.index, "Count": ind_data.values, "Country": [IND for _ in range(len(ind_data))]}))

    plt.figure(figsize=(15, 12))
    g = sns.catplot(y="Count", x="Country", hue="Sector",
                    data=graph_data, kind="bar", palette="muted", orient='v', legend_out=True)
    # for i, p in enumerate(g.patches):
    #     g.annotate('${0:,.2f}'.format(graph_data.iloc[i]['Count']), (p.get_x(
    #     ) + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 4), textcoords='offset points')

    g.set(xlabel='Country',
          ylabel='Investment Count', title='Country wise Investment count')
    plt.show()


# %%
plot_sector_wise_investment_comparision()
# %%
master_frame.to_csv("master_frame.csv", index=False, quoting=csv.QUOTE_ALL)
mapping.to_csv("mapping_frame.csv", index=False, quoting=csv.QUOTE_ALL)
melted_mapping = mapping.melt(id_vars=category_list)
melted_mapping = melted_mapping[melted_mapping['value'] == 1]
melted_mapping.to_csv("melted_mapping_frame.csv",
                      index=False, quoting=csv.QUOTE_ALL)
D1.to_csv("D1_frame.csv", index=False, quoting=csv.QUOTE_ALL)
D2.to_csv("D2_frame.csv", index=False, quoting=csv.QUOTE_ALL)
D3.to_csv("D3_frame.csv", index=False, quoting=csv.QUOTE_ALL)


# %%
