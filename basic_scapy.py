from logging import INFO
from logging import basicConfig
from logging import getLogger

from arrow import now
from scapy.all import sniff

DATA_FOLDER = './data/'
DEBUG = {}
OUTPUT_FOLDER = './result/'


def callback(packet):
    logger = getLogger(name='callback')
    logger.info(msg=packet.show())


def main():
    time_start = now()
    basicConfig(level=INFO, datefmt='%Y-%m-%d %H:%M:%S',
                format='%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s', )
    logger = getLogger(name='main', )

    sniff(prn=callback, count=1)
    time_seconds = (now() - time_start).total_seconds()
    logger.info(msg='done: {:02d}:{:05.2f}'.format(int(time_seconds // 60), time_seconds % 60))


if __name__ == '__main__':
    main()
