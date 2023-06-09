from logging import INFO
from logging import basicConfig
from logging import getLogger

from arrow import now
from matplotlib.pyplot import close
from matplotlib.pyplot import savefig
from pandas import DataFrame
from seaborn import scatterplot
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def get_data() -> DataFrame:
    result_df, target = load_digits(as_frame=True, return_X_y=True, )
    result_df['class'] = target
    return result_df


def make_scatter(df: DataFrame):
    logger = getLogger(name='make_pca_scatter', )
    logger.info(msg='columns: {}'.format(df.columns.tolist()))
    for (model, filename) in [
        (PCA(n_components=2, ), OUTPUT_FOLDER + SCATTER_PCA_FILE,),
        (TSNE(n_components=2, random_state=RANDOM_STATE, ), OUTPUT_FOLDER + SCATTER_TSNE_FILE,)
    ]:
        pixel_columns = [column for column in df.columns if column.startswith('pixel')]
        projection = model.fit_transform(X=df[pixel_columns])
        projection_df = DataFrame(data=projection, columns=['x', 'y'])
        projection_df['class'] = df['class']
        scatterplot(data=projection_df, hue='class', legend=False, palette='colorblind', x='x', y='y', )
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
RANDOM_STATE = 0
SCATTER_PCA_FILE = 'digits.pca.scatterplot.png'
SCATTER_TSNE_FILE = 'digits.tsne.scatterplot.png'

if __name__ == '__main__':
    main()
