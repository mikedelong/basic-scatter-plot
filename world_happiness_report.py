from logging import INFO
from logging import basicConfig
from logging import getLogger

from matplotlib.pyplot import close
from matplotlib.pyplot import savefig
from matplotlib.pyplot import text
from matplotlib.pyplot import title
from matplotlib.pyplot import xlabel
from matplotlib.pyplot import ylabel
from pandas import DataFrame
from pandas import read_excel
from seaborn import barplot
from seaborn import pairplot
from seaborn import scatterplot


def get_data(url: str) -> DataFrame:
    result_df = read_excel(io=url, sheet_name='2022')
    result_df = result_df[result_df['Country'] != 'xx']
    return result_df


def make_barplot(df: DataFrame):
    logger = getLogger(name='make_barplot', )
    value_vars = [item for item in df.columns if item.startswith('Explained')]
    melted_df = df.melt(id_vars=['Country'], value_vars=value_vars, var_name='Score', )
    barplot(data=melted_df, x='Country', y='value', hue='Score', palette='colorblind', )
    filename = OUTPUT_FOLDER + 'world_happiness_report_barplot.png'
    logger.info(msg='saving plot to {}'.format(filename))
    savefig(backend=None, bbox_inches=None, dpi='figure', edgecolor='auto', facecolor='auto', fname=filename,
            format='png', metadata=None, pad_inches=0.1, )
    close()


def make_pairplot(df: DataFrame):
    logger = getLogger(name='make_pairplot', )
    plot_df = df.drop(columns=['RANK', 'Happiness score', 'Whisker-high', 'Whisker-low',
                               'Dystopia (1.83) + residual',
                               ]).rename(columns={
        'Explained by: GDP per capita': 'GDP',
        'Explained by: Social support': 'Social',
        'Explained by: Healthy life expectancy': 'Health',
        'Explained by: Freedom to make life choices': 'Freedom',
        'Explained by: Generosity': 'Generosity',
        'Explained by: Perceptions of corruption': 'Corruption',
    })
    pairplot(data=plot_df, hue='Country')
    filename = OUTPUT_FOLDER + 'world_happiness_report_pairplot.png'
    logger.info(msg='saving plot to {}'.format(filename))
    savefig(backend=None, bbox_inches=None, dpi='figure', edgecolor='auto', facecolor='auto', fname=filename,
            format='png', metadata=None, pad_inches=0.1, )
    close()


def make_scatterplot(df: DataFrame):
    logger = getLogger(name='make_scatterplot', )
    scatterplot(data=df, x=FIGURE_X, y=FIGURE_Y, marker='')
    title('Happiness Score (2022)')
    xlabel(FIGURE_X.lower().capitalize())
    ylabel(FIGURE_Y.split()[1].lower().capitalize())
    for index, (rank, score) in enumerate(zip(df[FIGURE_X].values, df[FIGURE_Y].values)):
        text(fontsize=5, ha='center', s=df['Country'].values[index], x=rank, y=score, )
    filename = OUTPUT_FOLDER + OUTPUT_FILE
    logger.info(msg='saving plot to {}'.format(filename))
    savefig(backend=None, bbox_inches=None, dpi='figure', edgecolor='auto', facecolor='auto', fname=filename,
            format='png', metadata=None, pad_inches=0.1, )
    close()


def main():
    basicConfig(level=INFO, datefmt='%Y-%m-%d %H:%M:%S',
                format='%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s', )
    logger = getLogger(name='main', )
    data_df = get_data(url=FIGURE_URL)
    DEBUG['data'] = data_df
    make_barplot(df=data_df, )
    make_pairplot(df=data_df, )
    make_scatterplot(df=data_df, )

    logger.info(msg='columns: {}'.format(data_df.columns.tolist()))


DEBUG = {}
FIGURE_URL = 'https://happiness-report.s3.amazonaws.com/2022/Appendix_2_Data_for_Figure_2.1.xls'
FIGURE_X = 'RANK'
FIGURE_Y = 'Happiness score'
OUTPUT_FOLDER = './result/'
OUTPUT_FILE = 'world_happiness_report_figure.png'
TABLE_URL = 'https://happiness-report.s3.amazonaws.com/2022/DataForTable2.1.xls'

if __name__ == '__main__':
    main()
