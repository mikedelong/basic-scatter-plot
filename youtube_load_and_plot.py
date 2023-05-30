from json import load
from logging import INFO
from logging import basicConfig
from logging import getLogger

from arrow import now
from matplotlib.pyplot import close
from matplotlib.pyplot import savefig
from matplotlib.pyplot import scatter
from matplotlib.pyplot import ylabel
from pandas import read_csv
from pandas import to_datetime
from seaborn import scatterplot


def main():
    time_start = now()
    basicConfig(level=INFO, datefmt='%Y-%m-%d %H:%M:%S',
                format='%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s', )
    logger = getLogger(name='main', )
    with open(encoding='utf-8', file='youtube_load_and_plot.json', mode='r') as input_fp:
        settings = load(fp=input_fp, )

    videos_df = read_csv(filepath_or_buffer=settings['input_data_file'], usecols=['published', 'log10_views'])
    videos_df['published'] = to_datetime(arg=videos_df['published'], )
    videos_df = videos_df.sort_values(by='published', )
    plotting_package = settings['plotting_package']
    if plotting_package not in {'matplotlib.pyplot', 'plotly', 'seaborn'}:
        raise NotImplementedError(plotting_package)
    elif plotting_package == 'matplotlib.pyplot':
        filename = OUTPUT_FOLDER + SCATTER_FILENAME
        scatter(data=videos_df, x='published', y='log10_views', )
        ylabel(ylabel='log10 views')
        logger.info(msg='saving plot to {}'.format(filename), )
        savefig(backend=None, bbox_inches=None, dpi='figure', edgecolor='auto', facecolor='auto', fname=filename,
                format='png', metadata=None, pad_inches=0.1, )
        close()
    elif plotting_package == 'plotly':
        filename = OUTPUT_FOLDER + SCATTER_PLOTLY_FILENAME
        raise NotImplementedError(plotting_package)
    elif plotting_package == 'seaborn':
        filename = OUTPUT_FOLDER + SCATTERPLOT_FILENAME
        scatterplot(data=videos_df, x='published', y='log10_views', )
        ylabel(ylabel='log10 views')
        logger.info(msg='saving plot to {}'.format(filename), )
        savefig(backend=None, bbox_inches=None, dpi='figure', edgecolor='auto', facecolor='auto', fname=filename,
                format='png', metadata=None, pad_inches=0.1, )
        close()
    else:
        raise NotImplementedError(plotting_package)

    time_seconds = (now() - time_start).total_seconds()
    logger.info(msg='done: {:02d}:{:05.2f}'.format(int(time_seconds // 60), time_seconds % 60, ))


OUTPUT_FOLDER = './result/'
SCATTER_FILENAME = 'youtube.matplotlib.scatter.png'
SCATTER_PLOTLY_FILENAME = 'youtube.plotly.scatter.html'
SCATTERPLOT_FILENAME = 'youtube.seaborn.scatterplot.png'

if __name__ == '__main__':
    main()
