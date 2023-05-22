from datetime import timedelta
from json import load
from logging import INFO
from logging import basicConfig
from logging import getLogger
from re import findall
from typing import Generator

from arrow import now
from matplotlib.pyplot import close
from matplotlib.pyplot import savefig
from pandas import DataFrame
from scrapetube import get_channel
from seaborn import scatterplot

from requests_html import HTMLSession
from bs4 import BeautifulSoup as bs # importing BeautifulSoup


def get_generator(channel: str, channel_kind: str, ) -> Generator:
    if channel_kind == 'id':
        return get_channel(channel_id=channel, )
    elif channel_kind == 'url':
        return get_channel(channel_url=channel)
    else:
        raise NotImplementedError(channel_kind)


def alt(videos: Generator) -> DataFrame:
    return DataFrame()


def get_data_from_generator(videos: Generator) -> DataFrame:
    arrow_now = now()
    logger = getLogger(name='get_data_from_url')

    video_ids = []
    published_time = []
    view_count = []
    for video in videos:
        video_id = video['videoId']
        video_ids.append(video_id)
        published_time.append(video['publishedTimeText']['simpleText'])
        view_count.append(video['viewCountText']['simpleText'])
        logger.info(video_id)
        DEBUG['example_video'] = video
    result_df = DataFrame(data={'id': video_ids, 'published_time_text': published_time, 'view_count_text': view_count})
    DEBUG['video_data'] = result_df
    result_df['view_count'] = result_df['view_count_text'].astype(str).apply(lambda x: int(''.join(findall(r'\d+', x))))
    result_df['published_date'] = result_df['published_time_text'].apply(get_timedelta) + arrow_now.datetime
    return result_df


def get_timedelta(adjustment: str):
    pieces = adjustment.split()
    amount = -int(pieces[0])
    if pieces[1] == 'hours':
        return timedelta(hours=amount)
    elif pieces[1] == 'days':
        return timedelta(days=amount)
    elif pieces[1] == 'months':
        return timedelta(days=30 * amount)
    elif pieces[1] == 'years':
        return timedelta(days=365 * amount)
    else:
        return timedelta(days=0)


def main():
    time_start = now()
    basicConfig(level=INFO, datefmt='%Y-%m-%d %H:%M:%S',
                format='%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s', )
    logger = getLogger(name='main', )
    with open(file='youtube.json', mode='r', ) as input_fp:
        settings = load(fp=input_fp, )

    videos_generator = get_generator(channel=settings['channel_url'], channel_kind='url', )
    videos_df = get_data_from_generator(videos=videos_generator, )

    scatterplot(data=videos_df, x='published_date', y='view_count', )
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
