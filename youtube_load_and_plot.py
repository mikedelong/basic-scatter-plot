from json import dumps
from json import load
from logging import INFO
from logging import basicConfig
from logging import getLogger

from arrow import now
from matplotlib.pyplot import close
from matplotlib.pyplot import savefig
from matplotlib.pyplot import scatter
from matplotlib.pyplot import title
from matplotlib.pyplot import ylabel
from pandas import DataFrame
from pandas import read_csv
from plotly.express import scatter as plotly_scatter
from seaborn import scatterplot


def load_settings(filename: str, ) -> dict:
    with open(encoding='utf-8', file=filename, mode='r') as input_fp:
        result = dict(load(fp=input_fp, ))
    return {key: value for key, value in result.items() if key in result['keys']}


def make_plot(plotting_package: str, df: DataFrame, page_title: str, ):
    logger = getLogger(name='make_plot', )
    logger.info(msg='plotting package: {}'.format(plotting_package, ), )
    if plotting_package not in {'matplotlib.pyplot', 'plotly', 'seaborn'}:
        raise NotImplementedError(plotting_package)
    elif plotting_package == 'matplotlib.pyplot':
        filename = OUTPUT_FOLDER + SCATTER_FILENAME
        scatter(data=df, x='published', y='log10_views', )
        ylabel(ylabel='log10 views')
        title(label=page_title, )
        logger.info(msg='saving plot to {}'.format(filename), )
        savefig(backend=None, bbox_inches=None, dpi='figure', edgecolor='auto', facecolor='auto', fname=filename,
                format='png', metadata=None, pad_inches=0.1, )
        close()
    elif plotting_package == 'plotly':
        custom_data = ['name', 'published', 'views']
        labels = {'published': 'Date Published', 'log10_views': 'log10 of views'}
        figure = plotly_scatter(custom_data=custom_data, data_frame=df, labels=labels, title=page_title, x='published',
                                y='log10_views', )
        hover_template = '<br>'.join(
            ['video: %{customdata[0]}', 'date: %{customdata[1]}', 'views: %{customdata[2]}'])
        figure.update_traces(hovertemplate=hover_template, )
        filename = OUTPUT_FOLDER + SCATTER_PLOTLY_FILENAME
        logger.info(msg='saving plot to {}'.format(filename), )
        figure.write_html(filename)
    elif plotting_package == 'seaborn':
        filename = OUTPUT_FOLDER + SCATTERPLOT_FILENAME
        scatterplot(data=df, x='published', y='log10_views', )
        ylabel(ylabel='log10 views')
        title(label=page_title, )
        logger.info(msg='saving plot to {}'.format(filename), )
        savefig(backend=None, bbox_inches=None, dpi='figure', edgecolor='auto', facecolor='auto', fname=filename,
                format='png', metadata=None, pad_inches=0.1, )
        close()
    else:
        raise NotImplementedError(plotting_package)


def main():
    time_start = now()
    basicConfig(level=INFO, datefmt='%Y-%m-%d %H:%M:%S',
                format='%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s', )
    logger = getLogger(name='main', )
    settings = load_settings(filename='youtube_load_and_plot.json')
    logger.info(msg='settings: {}'.format(dumps(obj=settings, indent=4, sort_keys=True, ), ), )

    input_filename = settings['input_data_file']
    videos_df = read_csv(filepath_or_buffer=input_filename, usecols=USECOLS, )
    logger.info(msg='data from {} has shape: {}'.format(input_filename, videos_df.shape))
    make_plot(df=videos_df.sort_values(by='published', ), plotting_package=settings['plotting_package'],
              page_title=settings['page_title'], )

    time_seconds = (now() - time_start).total_seconds()
    logger.info(msg='done: {:02d}:{:05.2f}'.format(int(time_seconds // 60), time_seconds % 60, ))


OUTPUT_FOLDER = './result/'
SCATTER_FILENAME = 'youtube.matplotlib.scatter.png'
SCATTER_PLOTLY_FILENAME = 'youtube.plotly.scatter.html'
SCATTERPLOT_FILENAME = 'youtube.seaborn.scatterplot.png'
USECOLS = ['log10_views', 'name', 'published', 'views', ]

if __name__ == '__main__':
    main()
