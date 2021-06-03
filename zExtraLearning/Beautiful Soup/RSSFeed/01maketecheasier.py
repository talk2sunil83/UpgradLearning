# Copyright 2021 Sunil Yadav
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%
# import numpy as np
import pickle
import json
from time import sleep
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from bs4 import BeautifulSoup
import requests as rq
import random
from urllib.parse import urlparse
import feedparser
# %%
WebSite = 'WebSite'
Group = 'Group'
Name = 'Name'
BaseUrl = 'BaseUrl'
RelativeUrl = 'RelativeUrl'
FeedUrls = 'FeedUrls'
FeedUrlsHtml = 'FeedUrlsHtml'
cols = [WebSite, Group, Name, BaseUrl, RelativeUrl, FeedUrls]
RSSFeeds = "RSS Feeds"
RSSFeed = "RSS Feed"
data_file = "feed_maps.xlsx"
# %%


def GET_UA():
    chrome_91_agent_string = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36", "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.111 Safari/537.36"
    user_agent_strings = [chrome_91_agent_string,
                          "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/28.0.1500.72 Safari/537.36",
                          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10) AppleWebKit/600.1.25 (KHTML, like Gecko) Version/8.0 Safari/600.1.25",
                          "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:33.0) Gecko/20100101 Firefox/33.0",
                          "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.111 Safari/537.36",
                          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.111 Safari/537.36",
                          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit/600.1.17 (KHTML, like Gecko) Version/7.1 Safari/537.85.10",
                          "Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko",
                          "Mozilla/5.0 (Windows NT 6.3; WOW64; rv:33.0) Gecko/20100101 Firefox/33.0",
                          "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.104 Safari/537.36"
                          ]

    return str(random.choice(user_agent_strings))
    # return chrome_91_agent_string


# %%
df = pd.read_excel(data_file)
df.head()
# %%


def parse_url(url, features: str = "lxml"):

    headers = {'User-Agent': GET_UA()}
    content = None
    try:
        response = rq.get(url, headers=headers)
        ct = response.headers['Content-Type'].lower().strip()
        content = response.content

        if 'html' in ct or 'xml' in ct:
            soup = BeautifulSoup(content, features=features)
        else:
            soup = None

        return content, soup, ct
    except Exception as e:
        print("Error:", str(e))


# %%


def parse_feed_links(soup_object: BeautifulSoup):
    # return [a['href'].lower().strip() for a in soup.find_all('a', href=True) if urlparse(a['href']).netloc == urlparse(current_page).netloc]
    fsb = soup_object.find("div", {"id": "fsb"})
    headings = fsb.find_all('h3')
    paragraphs = fsb.find_all('p')
    print(len(headings), len(paragraphs))
    feed_urls = []
    for h, p in zip(headings, paragraphs):
        anchor = h.find("a")
        heading_text: str = anchor.getText() if anchor is not None else None

        if heading_text is not None and RSSFeed in heading_text:
            heading_text = heading_text.replace(RSSFeed, "")
        feed_link = p.find("a", href=True).attrs.get("href", None)
        feed_urls.append((heading_text, feed_link))
    return feed_urls


# %%
def get_feed_urls():
    feed_urls = []
    for i, row in df.iterrows():
        print(i)
        collector_url = row[BaseUrl]+row[RelativeUrl]
        _, soup, _ = parse_url(collector_url)
        if soup:
            feed_links = parse_feed_links(soup_object=soup)
            feed_urls.append(feed_links)
            sleep(random.randint(5, 10))
    if len(feed_urls) > 0:
        df[FeedUrls] = feed_urls


# get_feed_urls()

# %%
def append_extra_feeds():
    df.loc[len(df.index)] = ["MIT", "Machine learning", "Machine learning", None, None, [("ML News", " https://news.mit.edu/topic/mitmachine-learning-rss.xml")]]
    df.to_excel(data_file, sheet_name="Feed Urls", index=False)


# append_extra_feeds()

# %%
def create_html(lst):
    urls_html = ''
    for name, url in eval(lst):
        if len(url.strip()) > 0:
            urls_html += f'<span class="pointer btn btn-outline-secondary" onclick=getFeeds("{url}")>{name}</span>&nbsp;&nbsp;'
    return urls_html


df[FeedUrlsHtml] = df[FeedUrls].apply(create_html)
df.to_excel(data_file, sheet_name="Feed Urls", index=False)
# %%
df_dict = {}
for w in df[WebSite].unique():
    df_dict[w] = {}
    for g in df[df[WebSite] == w][Group].unique():
        df_dict[w][g] = {}
        for n in df[(df[WebSite] == w) & (df[Group] == g)][Name].unique():
            df_dict[w][g][n] = {}
            for bu in df[(df[WebSite] == w) & (df[Group] == g) & (df[Name] == n)][BaseUrl].unique():
                df_dict[w][g][n][bu] = {}
            for ru in df[(df[WebSite] == w) & (df[Group] == g) & (df[BaseUrl] == bu)][RelativeUrl].unique():
                df_dict[w][g][n][bu][ru] = str(list(df[(df[WebSite] == w) & (df[Group] == g) & (df[BaseUrl] == bu) & (df[RelativeUrl] == ru)][FeedUrlsHtml])[0])

# %%
df_dict['MIT']
# %%
with open('df_dict', 'wb') as f:
    pickle.dump(df_dict, f)
# %%
# Serializing json
# json_object = json.dumps(df_dict)
# print(json_object)

# %%
feeds = []
for i, row in df.iterrows():
    row_list = eval(row[FeedUrls])
    print(f"Before {len(row_list)}")
    row_list = row_list[:1]
    print(f"After {len(row_list)}")
    # print(f"i:{i}")
    for j, feed_url in enumerate(row_list):
        topic_name, url = feed_url
        # print(f"j:{j}")
        if len(url.strip()) > 0:
            # print(f"{topic_name}: {url}")
            try:
                feed = parse_url(url, features='xml')
                # feed = feedparser.parse(url)
                # print(f"i:{i}, j:{j}")
                # print(len(feed.entries))
            except Exception as ex:
                print(ex)

# %%
