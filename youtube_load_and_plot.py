from json import dumps
from json import load
from logging import INFO
from logging import basicConfig
from logging import getLogger
from math import log10

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


def duration_seconds(duration: str, ) -> int:
    pieces = duration.replace('PT', '').replace('S', '').split('M')
    return 60 * int(pieces[0]) + int(pieces[1])


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
        custom_data = ['name', 'published', 'views_with_commas']
        for item in [
            {
                'color': 'log10_duration_seconds',
                'filename': OUTPUT_FOLDER + SCATTER_PLOTLY_DATE_VIEWS_FILENAME,
                'labels': {'published': 'Date Published', 'log10_views': 'log10 of views'},
                'showlegend': True,
                'x': 'published',
                'y': 'log10_views',
            },
            {
                'color': 'year_published',
                'filename': OUTPUT_FOLDER + SCATTER_PLOTLY_DURATION_VIEWS_FILENAME,
                'labels': {'log10_duration_seconds': 'log10 of duration (sec)', 'log10_views': 'log10 of views'},
                'showlegend': True,
                'x': 'log10_duration_seconds',
                'y': 'log10_views',
            },
        ]:
            figure = plotly_scatter(color=item['color'], custom_data=custom_data, data_frame=df, labels=item['labels'],
                                    title=page_title, x=item['x'], y=item['y'], )
            figure.layout.showlegend = item['showlegend']
            hover_template = '<br>'.join(
                ['video: %{customdata[0]}', 'date: %{customdata[1]}', 'views: %{customdata[2]}'])
            figure.update_traces(hovertemplate=hover_template, )
            filename = item['filename']
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

    videos_df = read_csv(filepath_or_buffer=settings['input_data_file'], usecols=USECOLS, )
    videos_df['duration_seconds'] = videos_df['duration'].apply(duration_seconds)
    videos_df['log10_duration_seconds'] = videos_df['duration_seconds'].apply(func=lambda x: log10(1 + x))
    videos_df['views_with_commas'] = videos_df['views'].apply(func=lambda x: '{:,}'.format(x), )
    videos_df['year_published'] = videos_df['published'].apply(func=lambda x: x.split('-')[0])
    logger.info(msg='data has shape: {}'.format(videos_df.shape, ))
    make_plot(df=videos_df.sort_values(by='published', ), plotting_package=settings['plotting_package'],
              page_title=settings['page_title'], )

    time_seconds = (now() - time_start).total_seconds()
    logger.info(msg='done: {:02d}:{:05.2f}'.format(int(time_seconds // 60), time_seconds % 60, ))


OUTPUT_FOLDER = './result/'
SCATTER_FILENAME = 'youtube.matplotlib.scatter.png'
SCATTER_PLOTLY_DATE_VIEWS_FILENAME = 'youtube.plotly.date-views.scatter.html'
SCATTER_PLOTLY_DURATION_VIEWS_FILENAME = 'youtube.plotly.duration-views.scatter.html'
SCATTERPLOT_FILENAME = 'youtube.seaborn.scatterplot.png'
USECOLS = ['log10_views', 'name', 'published', 'views', 'duration', ]

if __name__ == '__main__':
    main()
