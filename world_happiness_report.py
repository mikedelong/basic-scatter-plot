from logging import INFO
from logging import basicConfig
from logging import getLogger

from matplotlib.pyplot import savefig
from pandas import DataFrame
from pandas import read_excel
from seaborn import scatterplot


def get_data(url: str) -> DataFrame:
    result_df = read_excel(io=url, sheet_name='2022')
    return result_df


def main():
    basicConfig(level=INFO, datefmt='%Y-%m-%d %H:%M:%S',
                format='%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s', )
    logger = getLogger(name='main', )
    data_df = get_data(url=FIGURE_URL)
    scatterplot(data=data_df, x='RANK', y='Happiness score', )
    filename = OUTPUT_FOLDER + OUTPUT_FILE
    logger.info(msg='saving plot to {}'.format(filename))
    savefig(backend=None, bbox_inches=None, dpi='figure', edgecolor='auto', facecolor='auto', fname=filename,
            format='png', metadata=None, pad_inches=0.1, )

    logger.info(msg='columns: {}'.format(data_df.columns.tolist()))


FIGURE_URL = 'https://happiness-report.s3.amazonaws.com/2022/Appendix_2_Data_for_Figure_2.1.xls'
OUTPUT_FOLDER = './result/'
OUTPUT_FILE = 'world_happiness_report_figure.png'
TABLE_URL = 'https://happiness-report.s3.amazonaws.com/2022/DataForTable2.1.xls'

if __name__ == '__main__':
    main()
