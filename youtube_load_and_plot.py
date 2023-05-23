from logging import INFO
from logging import basicConfig
from logging import getLogger

from arrow import now
from matplotlib.pyplot import close
from matplotlib.pyplot import savefig
from matplotlib.pyplot import scatter
from pandas import read_csv
from pandas import to_datetime


def main():
    time_start = now()
    basicConfig(level=INFO, datefmt='%Y-%m-%d %H:%M:%S',
                format='%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s', )
    logger = getLogger(name='main', )

    videos_df = read_csv(filepath_or_buffer='./videos.csv', usecols=['published', 'views'])
    videos_df['published'] = to_datetime(arg=videos_df['published'], )
    videos_df = videos_df.sort_values(by='published', )
    scatter(data=videos_df, x='published', y='views', )
    filename = OUTPUT_FOLDER + FILENAME
    logger.info(msg='saving plot to {}'.format(filename), )
    savefig(backend=None, bbox_inches=None, dpi='figure', edgecolor='auto', facecolor='auto', fname=filename,
            format='png', metadata=None, pad_inches=0.1, )
    close()

    time_seconds = (now() - time_start).total_seconds()
    logger.info(msg='done: {:02d}:{:05.2f}'.format(int(time_seconds // 60), time_seconds % 60, ))


DEBUG = {}
FILENAME = 'youtube.matplotlib.scatter.png'
OUTPUT_FOLDER = './result/'

if __name__ == '__main__':
    main()
