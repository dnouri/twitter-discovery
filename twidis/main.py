from functools import reduce
import logging
import operator
import re
import sqlite3

import altair as alt
import click
import pandas
import streamlit as st

from .extract import extract_articles_in_chunks


logging.basicConfig(filename='twidis.log', level=logging.DEBUG)


@click.group()
def cli():
    pass


def query_user_ids_for_screen_names(conn, users):
    cursor = conn.cursor()
    cursor.execute(
        'SELECT id FROM users '
        'WHERE screen_name IN (' + ','.join(['?'] * len(users)) + ')',
        users,
        )
    return [user_id for (user_id,) in cursor.fetchall()]


def tweets_results_to_dict(results):
    return [{
        'id': res[0],
        'user': res[1],
        'created_at': res[2],
        'full_text': res[3],
        'retweeted_status': res[4],
        'retweet_count': res[5],
        'favorite_count': res[6],
        'user_name': res[7],
        'user_screen_name': res[8],
        'url': f'https://twitter.com/{res[8]}/status/{res[0]}',
        } for res in results]


TWEET_QUERY = (
    'SELECT ' + ','.join([
        'tweets.id',
        'tweets.user',
        'tweets.created_at',
        'tweets.full_text',
        'tweets.retweeted_status',
        'tweets.retweet_count',
        'tweets.favorite_count',
        'users.name',
        'users.screen_name',
        ]) +
    ' FROM tweets '
    'LEFT JOIN users ON tweets.user = users.id '
    )


def query_tweets_for_user_ids(conn, user_ids):
    cursor = conn.cursor()
    query = TWEET_QUERY
    if user_ids:
        query += (
            'WHERE tweets.user IN (' + ','.join(['?'] * len(user_ids)) + ')')
        results = cursor.execute(query, user_ids)
    else:
        results = cursor.execute(query)
    return tweets_results_to_dict(results)


def query_tweets_favorited_for_user_ids(conn, user_ids):
    cursor = conn.cursor()
    query = 'SELECT tweet FROM favorited_by '
    if user_ids:
        query += 'WHERE user IN (' + ','.join(['?'] * len(user_ids)) + ')'
        results = cursor.execute(query, user_ids)
    else:
        results = cursor.execute(query)
    tweets = [res[0] for res in results]

    query = TWEET_QUERY + (
        'WHERE tweets.id IN (' + ','.join(['?'] * len(tweets)) + ')'
        )
    return tweets_results_to_dict(cursor.execute(query, tweets))


def extract_links(texts):
    return [
        [word for word in text.split()
         if word.startswith('http://') or word.startswith('https://')]
        for text in texts
        ]


@cli.command('articles')
@click.argument('db')
@click.argument('users', nargs=-1)
@click.option('--threads', default=16)
@click.option('--chunk-size', default=100)
@click.option('--outfile', default=None)
def articles_cli(db, users, threads, chunk_size, outfile):
    if outfile is None:
        outfile = f'articles-{"_".join(users)}.csv.gz'
    user_ids = None
    with sqlite3.connect(db) as conn:
        if users:
            user_ids = query_user_ids_for_screen_names(conn, users)
        tweets = query_tweets_for_user_ids(conn, user_ids)
        tweets += query_tweets_favorited_for_user_ids(conn, user_ids)

    links_to_tweets = {}
    tweets_links = extract_links([tw['full_text'] for tw in tweets])
    for tweet, links in zip(tweets, tweets_links):
        tweet['links'] = links
        for link in links:
            links_to_tweets[link] = tweet

    links = tuple(sorted(links_to_tweets.keys()))
    articles = extract_articles_in_chunks(
        links,
        chunk_size=chunk_size,
        n_threads=threads,
        )

    for article in articles:
        article_tweet = links_to_tweets[article['request_url']]
        for key, value in article_tweet.items():
            article[f'tweet_{key}'] = value

    pandas.DataFrame(articles).to_csv(outfile, index=False)
    print(f"Wrote to {outfile}")


@st.cache
def _read_csv(fname):
    df = pandas.read_csv(fname, parse_dates=['tweet_created_at'])
    df = df[pandas.isnull(df['error'])]
    df = df[~pandas.isnull(df['title'])]
    return df


@cli.command('gui')
@click.argument('infile')
@click.option('--max-articles-display', default=500)
def gui_cli(infile, max_articles_display):
    st.title("Twitter Discovery")
    st.write("Search articles referenced from"
             "your tweets, retweets, and favorites.")
    af = _read_csv(infile)
    af.sort_values('tweet_created_at', ascending=False, inplace=True)

    query_text = st.text_input("Search")
    query_netloc = st.multiselect(
        "Filter by article domain",
        af['url_netloc'].value_counts().index,
        )
    query_sorting = st.selectbox(
        "Sort results",
        ["Newest first", "Oldest first", "By domain"],
        )

    if query_netloc:
        af = af[af['url_netloc'].isin(query_netloc)]

    if query_text:
        parts = []
        for match in re.finditer(r"([\"'])(?:(?=(\\?))\2.)*?\1", query_text):
            matchg = match.group()
            query_text = query_text.replace(matchg, '')
            parts.append(matchg.strip('\'"'))
        parts.extend(query_text.replace('\'', '').replace('"', '').split())
        parts = [pa.lower() for pa in parts]

        to_search = {
            col: af[col].fillna('').str.lower()
            for col in [
                'title',
                'text',
                'tweet_full_text',
                'tweet_user_screen_name',
                'tweet_user_name',
                ]
            }
        to_search['tweet_user_screen_name'] = '@' + to_search[
            'tweet_user_screen_name']

        mask = None
        for part in parts:
            mask_part = [
                data.apply(lambda text: part in text)
                for data in to_search.values()
                ]
            mask_part = reduce(operator.__or__, mask_part)
            if mask is None:
                mask = mask_part
            else:
                mask = mask & mask_part

        af = af[mask]

    if query_sorting == 'Oldest first':
        af.sort_values('tweet_created_at', ascending=True, inplace=True)
    elif query_sorting == 'By domain':
        af.sort_values('url_netloc', kind='mergesort', inplace=True)

    st.write(f"Found {len(af)} articles:")

    af_chart = af[[
        'title', 'meta_description',
        'url', 'url_netloc', 'tweet_url',
        'tweet_full_text', 'tweet_user_screen_name',
        'tweet_created_at', 'tweet_favorite_count', 'tweet_retweet_count',
        ]]
    af_chart['favorites'] = af_chart['tweet_favorite_count'].clip(1)
    af_chart['retweets'] = af_chart['tweet_retweet_count'].clip(1)
    chart = alt.Chart(
        af_chart,
        width=min(len(af) * 60, 600),
        height=min(len(af) * 40, 400),
    ).mark_circle(size=60).encode(
        x='tweet_created_at',
        y=alt.X('retweets', scale=alt.Scale(type='log')),
        color=alt.Color(
            'url_netloc',
            sort=alt.EncodingSortField(
                field='url_netloc', op='count', order='descending')),
        tooltip=[
            'title',
            'url',
            'tweet_user_screen_name',
            'tweet_created_at',
            'tweet_full_text',
            'meta_description',
            ],
        href='tweet_url',
        )
    st.write(chart)

    af = af[:max_articles_display]
    for index, article in af.iterrows():
        favicon_html = (
            f"<img "
            f"src='{article.url_scheme}://{article.url_netloc}/favicon.ico' "
            f"width='24px'>"
            )
        st.markdown(
            f"### {favicon_html} [{article.title}]({article.url}) "
            f"<small>({article.url_netloc})</small>",
            unsafe_allow_html=True,
            )
        st.markdown(
            f"[@{article.tweet_user_screen_name} on "
            f"{article.tweet_created_at.strftime('%Y-%m-%d')}]"
            f"({article.tweet_url}): "
            f"<q>{article.tweet_full_text}</q>",
            unsafe_allow_html=True,
            )


if __name__ == '__main__':
    cli(obj={}, standalone_mode=False)
