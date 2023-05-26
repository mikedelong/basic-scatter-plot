import math
from json import load
from logging import INFO
from logging import basicConfig
from logging import getLogger
from typing import Generator

from arrow import now
from bs4 import BeautifulSoup as bs
from bs4.element import ResultSet
from matplotlib.pyplot import close
from matplotlib.pyplot import savefig
from pandas import DataFrame
from pandas import Series
from pandas import to_datetime
from pyppeteer.errors import TimeoutError
from requests_html import HTMLSession
from scrapetube import get_channel
from seaborn import scatterplot


def get_generator(channel: str, channel_kind: str, ) -> Generator:
    if channel_kind == 'id':
        return get_channel(channel_id=channel, )
    elif channel_kind == 'url':
        return get_channel(channel_url=channel)
    else:
        raise NotImplementedError(channel_kind)


# todo figure out how to encode the channel name

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


def get_data_from_generator(videos: Generator) -> DataFrame:
    logger = getLogger(name='get_data_from_generator')
    result = []
    for video in videos:
        video_id = video['videoId']
        url = 'https://youtu.be/{}'.format(video_id)
        logger.info(msg=url, )
        try:
            with HTMLSession() as session:
                response = session.get(url=url, )
                response.html.render(sleep=1)
                soup = bs(response.html.html, 'html.parser')
                DEBUG['soup'] = soup
                tags = soup.find_all(name='meta', )
                result.append(Series(data=tags_to_dict(tags, )))
        except TimeoutError as timeout_error:
            logger.warning(timeout_error)
    result_df = DataFrame(data=result, )
    DEBUG['result_df'] = result_df
    result_df['published'] = to_datetime(arg=result_df['datePublished'], )
    result_df['views'] = result_df['interactionCount'].astype(int)
    result_df['log10_views'] = result_df['views'].apply(lambda x: 0 if x == 0 else round(math.log10(x), 2))
    result_df = result_df.sort_values(by='published', )
    return result_df


def today_as_string() -> str:
    time_now = now()
    return '{}-{:02d}{:02d}'.format(time_now.date().year, time_now.date().month, time_now.date().day, )


def main():
    time_start = now()
    basicConfig(level=INFO, datefmt='%Y-%m-%d %H:%M:%S',
                format='%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s', )
    logger = getLogger(name='main', )
    with open(file='youtube.json', mode='r', ) as input_fp:
        settings = load(fp=input_fp, )

    videos_generator = get_generator(channel=settings['channel_url'], channel_kind='url', )
    videos_df = get_data_from_generator(videos=videos_generator, )
    filename = DATA_FOLDER + '-'.join([today_as_string(), DATA_FILENAME])
    videos_df.to_csv(index=False, path_or_buf=filename, )

    scatterplot(data=videos_df, x='published', y='views', )
    filename = OUTPUT_FOLDER + '-'.join([today_as_string(), PLOT_FILENAME])
    logger.info(msg='saving plot to {}'.format(filename))
    savefig(backend=None, bbox_inches=None, dpi='figure', edgecolor='auto', facecolor='auto', fname=filename,
            format='png', metadata=None, pad_inches=0.1, )
    close()

    time_seconds = (now() - time_start).total_seconds()
    logger.info(msg='done: {:02d}:{:05.2f}'.format(int(time_seconds // 60), time_seconds % 60))


DATA_FILENAME = 'youtube-data.csv'
DATA_FOLDER = './data/'
DEBUG = {}
PLOT_FILENAME = 'youtube-scatterplot.png'
OUTPUT_FOLDER = './result/'

if __name__ == '__main__':
    main()
