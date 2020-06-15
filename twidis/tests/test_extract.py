import pytest


class TestExtractArticle:
    @pytest.fixture
    def extract_article(self):
        from twidis.extract import extract_article
        return extract_article

    def test_cnn(self, extract_article):
        url = ('https://edition.cnn.com/2020/06/15/politics/'
               'us-fighter-jet-crash-england/index.html')
        info = extract_article(url)
        assert info['url'] == url
        assert info['title'] == (
            'US Air Force fighter jet pilot dead after North Sea crash')
        assert info['authors'] == [
            'Sharon Braithwaite', 'Chandelis Duster', 'Stephanie Halasz',
            'Schams Elwazer', 'Caroline Kelly',
            ]
        assert info['summary'].startswith('(CNN) A US Air Force F-15 fighter')
        assert info['text'].endswith('with one pilot on board."\n\nRead More')
