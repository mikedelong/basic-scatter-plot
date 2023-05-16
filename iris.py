from logging import INFO
from logging import basicConfig
from logging import getLogger

from matplotlib.pyplot import savefig
from pandas import DataFrame
from seaborn import pairplot
from sklearn.datasets import load_iris


def get_data() -> DataFrame:
    iris = load_iris()
    result_df = DataFrame(iris.data, columns=iris.feature_names)
    result_df['target'] = iris.target
    return result_df


def main():
    basicConfig(level=INFO, datefmt='%Y-%m-%d %H:%M:%S',
                format='%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s', )
    logger = getLogger(name='main', )
    data_df = get_data()
    logger.info(msg='columns: %s'.format(data_df.columns.tolist()))
    pairplot(data=data_df, hue='target')
    filename = OUTPUT_FOLDER + OUTPUT_FILE
    logger.info(msg='saving plot to {}'.format(filename))
    savefig(format='png', fname=filename)


OUTPUT_FOLDER = './result/'
OUTPUT_FILE = 'iris.pairplot.png'

if __name__ == '__main__':
    main()
