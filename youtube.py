from datetime import timedelta
from json import load
from logging import INFO
from logging import basicConfig
from logging import getLogger
from re import findall

from arrow import now
from pandas import DataFrame
from scrapetube import get_channel


def get_data(channel_id: str) -> DataFrame:
    arrow_now = now()
    logger = getLogger(name='get_data')
    videos = get_channel(channel_id=channel_id)

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

    videos_df = get_data(channel_id=settings['channel_id'])

    time_seconds = (now() - time_start).total_seconds()
    logger.info(msg='done: {:02d}:{:05.2f}'.format(int(time_seconds // 60), time_seconds % 60))


DEBUG = {}
if __name__ == '__main__':
    main()
