from logging import INFO
from logging import basicConfig
from logging import getLogger

from arrow import now
from matplotlib.pyplot import close
from matplotlib.pyplot import savefig
from pandas import DataFrame
from pandas import read_html
from seaborn import pairplot


def get_data() -> DataFrame:
    result_df = read_html(io=URL, header=0)[0]
    columns = ['Rank', 'Rank.1', 'Census population[8][a].1', 'Census population[8][a].2',
               'Change, 2010–2020[8][a]', 'Change, 2010–2020[8][a].1', 'House of Reps. from the 2022 elections.1',
               'Pop. per elec. vote, 2020[b] from the 2022 elections', 'Census pop. per seat[a].1',
               '% of the total U.S. pop.[c]', '% of the total U.S. pop.[c].1', '% of Elec. Coll.']
    index = [0, 31, 50, 53, 54, 55, 56, 57, 58, 59, 60]
    result_df = result_df.drop(columns=columns).drop(index=index, ).reset_index().drop(columns=['index'])
    result_df.columns = ['State', 'Population', 'Votes', 'Ratio']
    for column in [item for item in result_df.columns if item != 'State']:
        result_df[column] = result_df[column].astype(int)
    result_df['Votes'] = result_df['Votes'] + 2

    DEBUG['data'] = result_df
    return result_df


def make_pairplot(df: DataFrame):
    logger = getLogger(name='make_pairplot', )
    logger.info(msg='columns: {}'.format(df.columns.tolist()))
    for filename in [OUTPUT_FOLDER + PAIRPLOT_FILE]:
        pairplot(data=df, corner=True, )
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
    make_pairplot(df=data_df, )
    time_seconds = (now() - time_start).total_seconds()
    logger.info(msg='done: {:02d}:{:05.2f}'.format(int(time_seconds // 60), time_seconds % 60))


DEBUG = {}
OUTPUT_FOLDER = './result/'
PAIRPLOT_FILE = 'states.pairplot.png'
URL = 'https://en.wikipedia.org/wiki/List_of_U.S._states_and_territories_by_population#State_and_territory_rankings'
if __name__ == '__main__':
    main()
