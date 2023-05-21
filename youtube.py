from logging import INFO
from logging import basicConfig
from logging import getLogger

from arrow import now
from scrapetube import get_channel


def main():
    time_start = now()
    basicConfig(level=INFO, datefmt='%Y-%m-%d %H:%M:%S',
                format='%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s', )
    logger = getLogger(name='main', )

    videos = get_channel('UCCezIgC97PvUuR4_gbFUs5g')

    for video in list(videos)[:2]:
        # logger.info(video['videoId'])
        logger.info(video)
        DEBUG['video'] = video

    time_seconds = (now() - time_start).total_seconds()
    logger.info(msg='done: {:02d}:{:05.2f}'.format(int(time_seconds // 60), time_seconds % 60))


DEBUG = {}
if __name__ == '__main__':
    main()
