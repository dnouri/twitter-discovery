Twitter Discovery
=================

Twitter Discovery lets you search articles referenced from your
tweets, retweets, and favorites.

Installation
------------

Install Twitter Discovery from source using ``pip install -e .``.
(And do yourself a favor and `use venv
<https://docs.python.org/3/tutorial/venv.html>`_.)

Also install `twitter-to-sqlite
<https://pypi.org/project/twitter-to-sqlite/>`_ using ``pip install
twitter-to-sqlite``.  This is the tool that you'll use to download
data from Twitter.  Follow the instructions in the twitter-to-sqlite
docs and set up authentication with Twitter.

Usage
-----

Note that *dnouri* is my own Twitter handle, so whenever you encounter
that in the commands below, you should replace *dnouri* with your own
Twitter handle.

Here's how you first download data from Twitter:

- Retrieve your tweets and retweets using ``twitter-to-sqlite
  user-timeline dnouri.db``.

- Add favorites using ``twitter-to-sqlite favorites dnouri.db``.

Now you're ready to find in the tweets that you downloaded any links
to news articles and blogs.  The following command will find all such
URLs and it will attempt to download the linked documents and save
them for further use: ``twitter-discovery articles dnouri.db dnouri``.

Finally, use the web-based GUI to search all those articles.  To start
it up, run: ``streamlit run twidis/main.py gui
articles-dnouri.csv.gz``.

The GUI allows you to search articles by text, order them by date or
website, and so on.  The text search is case insensitive and supports
the use of quotation marks to group words, such that a search for
``"big finance"`` will return articles that contain the phrase *big
finance*, while searching for ``big finance`` without the quotes will
return articles that contain both the word *big* and the word
*finance*.  The text search will match text inside the article's body,
but also in the article's title and the text of the tweet that
referenced the article.
