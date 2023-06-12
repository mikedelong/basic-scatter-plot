from json import load
from logging import INFO
from logging import basicConfig
from logging import getLogger

from arrow import now
from holoviews import Chord
from holoviews import Dataset
from holoviews import dim
from holoviews import extension
from holoviews import opts
from holoviews import output
from holoviews import save
from pandas import DataFrame
from pandas import read_csv

COLUMNS = ['No.', 'Time', 'Source', 'Destination', 'Protocol', 'Length', 'Info']
DATA_FOLDER = './data/'
DEBUG = {}
OUTPUT_FOLDER = './result/'


def read_dataframe(filename: str, ) -> DataFrame:
    result_df = read_csv(filepath_or_buffer=filename, )
    return result_df


def main():
    time_start = now()
    basicConfig(level=INFO, datefmt='%Y-%m-%d %H:%M:%S',
                format='%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s', )
    logger = getLogger(name='main', )

    with open(file='./basic_wireshark.json', encoding='utf-8', mode='r') as input_fp:
        settings = load(fp=input_fp, )

    df = read_dataframe(filename=DATA_FOLDER + settings['data_file'])
    DEBUG['data'] = df
    logger.info(msg=df.shape)
    logger.info(msg=df.columns.tolist())
    logger.info(msg=df[['Source', 'Destination', ]].head())
    logger.info(msg=df['Protocol'].value_counts().to_dict())
    tcp_df = df.drop(columns=['No.'])[df['Protocol'] == 'TCP']
    tcp_df = tcp_df[~tcp_df['Source'].str.contains(':')]
    DEBUG['tcp'] = tcp_df
    logger.info(msg=tcp_df.shape)
    columns = ['Source', 'Destination']
    count_df = tcp_df[columns].groupby(by=columns, as_index=False).size()
    DEBUG['counts'] = count_df
    extension('bokeh', )
    nodes = Dataset(data=count_df, kdims='Source', vdims='Destination')
    DEBUG['nodes'] = nodes
    chord = Chord(data=count_df, )
    chord.opts(
        opts.Chord(cmap='Viridis', edge_cmap='Viridis', labels='Destination',
                   node_color=dim('index').str())
    )
    output(size=300, )

    filename = OUTPUT_FOLDER + settings['data_file'].replace('.csv', '.html')
    save(obj=chord, filename=filename)

    time_seconds = (now() - time_start).total_seconds()
    logger.info(msg='done: {:02d}:{:05.2f}'.format(int(time_seconds // 60), time_seconds % 60))


if __name__ == '__main__':
    main()
