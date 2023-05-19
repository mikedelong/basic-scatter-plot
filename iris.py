from logging import INFO
from logging import basicConfig
from logging import getLogger

from arrow import now
from matplotlib.pyplot import close
from matplotlib.pyplot import savefig
from pandas import DataFrame
from seaborn import pairplot
from sklearn.datasets import load_iris


def get_data() -> DataFrame:
    result_df, target = load_iris(as_frame=True, return_X_y=True, )
    result_df['class'] = target.replace({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    return result_df


def main():
    time_start = now()
    basicConfig(level=INFO, datefmt='%Y-%m-%d %H:%M:%S',
                format='%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s', )
    logger = getLogger(name='main', )
    data_df = get_data()
    logger.info(msg='columns: {}'.format(data_df.columns.tolist()))
    pairplot(data=data_df, hue='class')
    filename = OUTPUT_FOLDER + OUTPUT_FILE
    logger.info(msg='saving plot to {}'.format(filename))
    savefig(backend=None, bbox_inches=None, dpi='figure', edgecolor='auto', facecolor='auto', fname=filename,
            format='png', metadata=None, pad_inches=0.1, )
    close()
    time_seconds = (now() - time_start).total_seconds()
    logger.info(msg='done: {:02d}:{:05.2f}'.format(int(time_seconds // 60), time_seconds % 60))


OUTPUT_FOLDER = './result/'
OUTPUT_FILE = 'iris.pairplot.png'

if __name__ == '__main__':
    main()
