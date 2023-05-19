from logging import INFO
from logging import basicConfig
from logging import getLogger

from matplotlib.pyplot import savefig
from pandas import DataFrame

from sklearn.datasets import load_digits
from seaborn import scatterplot
from sklearn.decomposition import PCA
from matplotlib.pyplot import close

def get_data() -> DataFrame:
    result_df, target = load_digits(as_frame=True, return_X_y=True, )
    result_df['class'] = target
    return result_df



def make_pca_scatter(df: DataFrame):
    logger = getLogger(name='make_pca_scatter', )
    logger.info(msg='columns: {}'.format(df.columns.tolist()))
    model = PCA(n_components=2,)
    pixel_columns = [column for column in df.columns if column.startswith('pixel')]
    projection = model.fit_transform(X=df[pixel_columns])
    projection_df = DataFrame(data=projection, columns=['x', 'y'])
    projection_df['class'] = df['class']
    DEBUG['projection'] = projection_df
    scatterplot(data=projection_df, x='x', y='y', hue='class',)
    filename = OUTPUT_FOLDER + OUTPUT_FILE
    logger.info(msg='saving plot to {}'.format(filename))
    savefig(backend=None, bbox_inches=None, dpi='figure', edgecolor='auto', facecolor='auto', fname=filename,
            format='png', metadata=None, pad_inches=0.1, )
    close()

def main():
    basicConfig(level=INFO, datefmt='%Y-%m-%d %H:%M:%S',
                format='%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s', )
    logger = getLogger(name='main', )
    data_df = get_data()
    DEBUG['data'] = data_df
    make_pca_scatter(df=data_df,)

DEBUG = {}
OUTPUT_FOLDER = './result/'
OUTPUT_FILE = 'digits.scatterplot.png'

if __name__ == '__main__':
    main()
