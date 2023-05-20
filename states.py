from logging import INFO
from logging import basicConfig
from logging import getLogger

from arrow import now
from matplotlib.pyplot import close
from matplotlib.pyplot import savefig
from pandas import DataFrame
from seaborn import pairplot


def get_data() -> DataFrame:
    result_df = DataFrame()
    return result_df


def make_scatter(df: DataFrame):
    logger = getLogger(name='make_pca_scatter', )
    logger.info(msg='columns: {}'.format(df.columns.tolist()))
    for (model, filename) in [
    ]:
        pixel_columns = [column for column in df.columns if column.startswith('pixel')]
        projection = model.fit_transform(X=df[pixel_columns])
        projection_df = DataFrame(data=projection, columns=['x', 'y'])
        projection_df['class'] = df['class']
        pairplot(data=projection_df, palette='colorblind', )
        logger.info(msg='saving plot to {}'.format(filename))
        savefig(backend=None, bbox_inches=None, dpi='figure', edgecolor='auto', facecolor='auto', fname=filename,
                format='png', metadata=None, pad_inches=0.1, )
        close()


def main():
    time_start = now()
    basicConfig(level=INFO, datefmt='%Y-%m-%d %H:%M:%S',
                format='%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s', )
    logger = getLogger(name='main', )
    data_df = get_data()
    make_scatter(df=data_df, )
    time_seconds = (now() - time_start).total_seconds()
    logger.info(msg='done: {:02d}:{:05.2f}'.format(int(time_seconds // 60), time_seconds % 60))


OUTPUT_FOLDER = './result/'

if __name__ == '__main__':
    main()
