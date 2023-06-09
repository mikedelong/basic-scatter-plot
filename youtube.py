from json import load
from logging import INFO
from logging import basicConfig
from logging import getLogger
from math import log10
from typing import Generator

from arrow import now
from bs4 import BeautifulSoup
from bs4.element import ResultSet
from pandas import DataFrame
from pandas import Series
from pandas import to_datetime
from pyppeteer.errors import TimeoutError
from requests_html import HTMLSession
from scrapetube import get_channel


def get_data_from_generator(videos: Generator) -> DataFrame:
    result = [get_meta_from_url(url='https://youtu.be/{}'.format(video['videoId'])) for video in videos]
    result_df = DataFrame(data=result, )
    DEBUG['result_df'] = result_df
    result_df['views'] = result_df['interactionCount'].astype(int)
    result_df['log10_views'] = result_df['views'].apply(lambda x: 0 if x == 0 else round(log10(x), 2))
    result_df['published'] = to_datetime(arg=result_df['datePublished'], )
    result_df = result_df.sort_values(by='published', )
    return result_df


def get_generator(channel: str, ) -> Generator:
    return get_channel(channel_url=channel)


def get_representation(settings: dict, ) -> str:
    pieces = settings['channel_url'].split('/')
    result = [piece for piece in pieces if piece.startswith('@')][0]
    return result


def get_meta_from_url(url: str) -> Series:
    logger = getLogger(name='get_meta_from_url')
    logger.info(msg=url, )
    try:
        with HTMLSession() as session:
            response = session.get(url=url, )
            response.html.render(sleep=1, )
            soup = BeautifulSoup(response.html.html, 'html.parser')
            DEBUG['soup'] = soup
            result = Series(data=tags_to_dict(soup.find_all(name='meta', ), ))
    except TimeoutError as timeout_error:
        logger.warning(timeout_error)
        result = Series()
    return result


def tags_to_dict(tags: ResultSet, ) -> dict:
    result = {}
    for tag in tags:
        if tag.get('property'):
            result[tag['property']] = tag['content']
        elif tag.get('name'):
            result[tag['name']] = tag['content']
        elif tag.get('itemprop'):
            result[tag['itemprop']] = tag['content']
    return result


def today_as_string() -> str:
    time_now = now()
    return '{}-{:02d}{:02d}'.format(time_now.date().year, time_now.date().month, time_now.date().day, )


def main():
    time_start = now()
    basicConfig(level=INFO, datefmt='%Y-%m-%d %H:%M:%S',
                format='%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s', )
    logger = getLogger(name='main', )
    with open(file='./youtube.json', mode='r', ) as input_fp:
        settings = load(fp=input_fp, )

    videos_generator = get_generator(channel=settings['channel_url'], )
    representation = get_representation(settings, )
    videos_df = get_data_from_generator(videos=videos_generator, )
    filename = DATA_FOLDER + '-'.join([today_as_string(), representation, DATA_FILENAME])
    videos_df.to_csv(index=False, path_or_buf=filename, )

    time_seconds = (now() - time_start).total_seconds()
    logger.info(msg='done: {:02d}:{:05.2f}'.format(int(time_seconds // 60), time_seconds % 60))


DATA_FILENAME = 'youtube-data.csv'
DATA_FOLDER = './data/'
DEBUG = {}
OUTPUT_FOLDER = './result/'
PLOT_FILENAME = 'youtube-scatterplot.png'

if __name__ == '__main__':
    main()
