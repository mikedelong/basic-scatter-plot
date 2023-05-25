import math
from datetime import datetime
from json import load
from logging import INFO
from logging import basicConfig
from logging import getLogger
from typing import Generator

from arrow import now
from bs4 import BeautifulSoup as bs
from matplotlib.pyplot import close
from matplotlib.pyplot import savefig
from pandas import DataFrame
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


# todo get other columns of data including name, upload date, video ID, etc.
# todo figure out how to encode the channel name

def get_data_from_generator(videos: Generator) -> DataFrame:
    logger = getLogger(name='get_data_from_generator')
    video_ids = []
    views = []
    date_published = []
    for video in videos:
        video_id = video['videoId']
        url = 'https://youtu.be/{}'.format(video_id)
        logger.info(msg=url, )
        try:
            with HTMLSession() as session:
                response = session.get(url=url, )
                response.html.render(sleep=1)
                soup = bs(response.html.html, 'html.parser')
                video_ids.append(video_id)
                views.append(soup.find('meta', itemprop='interactionCount')['content'])
                date_published.append(soup.find('meta', itemprop='datePublished')['content'])
        except TimeoutError as timeout_error:
            logger.warning(timeout_error)

    date_published = [datetime.strptime(item, '%Y-%m-%d') for item in date_published]
    result_df = DataFrame(data={'id': video_ids, 'views': views, 'published': date_published})
    result_df['published'] = to_datetime(arg=result_df['published'], )
    result_df['views'] = result_df['views'].astype(int)
    result_df['log10_views'] = result_df['views'].apply(lambda x: 0 if x == 0 else math.log10(x))
    result_df =  result_df.sort_values(by='published', )
    return result_df


def main():
    time_start = now()
    basicConfig(level=INFO, datefmt='%Y-%m-%d %H:%M:%S',
                format='%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s', )
    logger = getLogger(name='main', )
    with open(file='youtube.json', mode='r', ) as input_fp:
        settings = load(fp=input_fp, )

    videos_generator = get_generator(channel=settings['channel_url'], channel_kind='url', )
    videos_df = get_data_from_generator(videos=videos_generator, )
    videos_df.to_csv(index=False, path_or_buf='./videos.csv', )

    scatterplot(data=videos_df, x='published', y='views', )
    filename = OUTPUT_FOLDER + FILENAME
    logger.info(msg='saving plot to {}'.format(filename))
    savefig(backend=None, bbox_inches=None, dpi='figure', edgecolor='auto', facecolor='auto', fname=filename,
            format='png', metadata=None, pad_inches=0.1, )
    close()

    time_seconds = (now() - time_start).total_seconds()
    logger.info(msg='done: {:02d}:{:05.2f}'.format(int(time_seconds // 60), time_seconds % 60))


DEBUG = {}
FILENAME = 'youtube.scatterplot.png'
OUTPUT_FOLDER = './result/'

if __name__ == '__main__':
    main()
