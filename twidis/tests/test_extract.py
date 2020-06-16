from datetime import datetime

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
        assert info['publish_date'] == datetime(2020, 6, 15, 0, 0)
        assert info['summary'].startswith('(CNN) A US Air Force F-15 fighter')
        assert info['text'].endswith('with one pilot on board."\n\nRead More')

    @pytest.mark.parametrize('url', [
        'https://arxiv.org/abs/1703.06870',
        'https://arxiv.org/pdf/1703.06870.pdf',
        ])
    def test_arxiv(self, extract_article, url):
        info = extract_article(url)
        if url.endswith('.pdf'):
            url = url.replace('.pdf', '').replace('pdf', 'abs')
        assert info['url'] == url
        assert info['title'] == 'Mask R-CNN'
        assert info['authors'] == [
            'Kaiming He', 'Georgia Gkioxari', 'Piotr Dollár', 'Ross Girshick'
            ]
        assert info['publish_date'] == datetime(2017, 3, 20, 17, 53, 38)
        assert info['summary'].startswith('We present a conceptually simple')
        assert info['text'].endswith('github.com/facebookresearch/Detectron')
