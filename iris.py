from logging import INFO
from logging import basicConfig
from logging import getLogger

from pandas import DataFrame
from sklearn.datasets import load_iris


def get_data() -> DataFrame:
    iris = load_iris()
    result_df = DataFrame(iris.data, columns=iris.feature_names)
    result_df['target'] = iris.target
    return result_df


def main():
    basicConfig(level=INFO, )
    logger = getLogger(name='main', )
    data_df = get_data()
    logger.info(data_df.columns.tolist())


if __name__ == '__main__':
    main()
