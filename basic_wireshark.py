from logging import INFO
from logging import basicConfig
from logging import getLogger

from arrow import now
from pandas import read_csv
from pandas import DataFrame
from json import load

DATA_FOLDER = './data/'
DEBUG = {}
OUTPUT_FOLDER = './result/'

def read_dataframe(filename:str, )-> DataFrame:
    result_df = read_csv(filepath_or_buffer=filename, )
    return result_df

def main():
    time_start = now()
    basicConfig(level=INFO, datefmt='%Y-%m-%d %H:%M:%S',
                format='%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s', )
    logger = getLogger(name='main', )

    with open(file='./basic_wireshark.json', encoding='utf-8', mode='r') as input_fp:
        settings = load(fp=input_fp,)

    df = read_dataframe(filename= DATA_FOLDER + settings['data_file'])
    logger.info(msg=df.shape)
    logger.info(msg=df.columns.tolist())
    logger.info(msg=df['Source'].value_counts().to_dict())
    logger.info(msg=df[['Source', 'Destination']].head())

    time_seconds = (now() - time_start).total_seconds()
    logger.info(msg='done: {:02d}:{:05.2f}'.format(int(time_seconds // 60), time_seconds % 60))


if __name__ == '__main__':
    main()
