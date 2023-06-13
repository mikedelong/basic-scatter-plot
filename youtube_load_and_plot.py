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
from numpy import unique
from pandas import DataFrame
from pandas import read_csv
from pandas import to_datetime
from plotly.express import scatter as plotly_scatter
from plotly.express.colors import qualitative
from plotly.express.colors import sequential
from seaborn import scatterplot
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


def add_dbscan_cluster(df: DataFrame) -> DataFrame:
    logger = getLogger(name='add_dbscan_cluster', )
    vectorizer = TfidfVectorizer(analyzer='word', binary=False, decode_error='strict', encoding='utf-8',
                                 input='content', lowercase=True, max_df=1.0, max_features=None, min_df=1,
                                 ngram_range=(1, 3), preprocessor=None, stop_words=None, strip_accents=None,
                                 tokenizer=None, token_pattern=r'(?u)\b\w\w+\b', vocabulary=None, )
    x = vectorizer.fit_transform(raw_documents=df['keywords'], )
    logger.info(msg='built/fitted vectorizer')
    scores = {}
    clusters = {}
    eps = 0.5  # was 0.5
    min_samples = 10  # max(5, int(math.sqrt(len(df)) / 2))
    logger.info(msg='min_samples: {}'.format(min_samples))
    for n_clusters in range(2, 12):  # was (2, 12)
        logger.info(msg='running DBSCAN for {} clusters'.format(n_clusters))
        model = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', metric_params=None, algorithm='auto',
                       leaf_size=30, p=None, n_jobs=None, )
        model.fit(X=x, )
        scores[n_clusters] = silhouette_score(X=x, labels=model.labels_, metric='euclidean', sample_size=None,
                                              random_state=RANDOM_STATE, )
        clusters[n_clusters] = model.labels_
        logger.info(msg='cluster count: {} score: {:06.4f}'.format(len(unique(model.labels_)), scores[n_clusters]))
    best_key = max(scores, key=lambda key: scores[key])
    column = 'DBSCAN cluster'
    df[column] = clusters[best_key]
    logger.info(msg=df[column].value_counts().to_dict())
    logger.info(msg='DBSCAN score: {:06.4f}'.format(scores[best_key]))
    return df


def add_kmeans_cluster(df: DataFrame) -> DataFrame:
    logger = getLogger(name='add_kmeans_cluster', )
    vectorizer = TfidfVectorizer(analyzer='word', binary=False, decode_error='strict', encoding='utf-8',
                                 input='content', lowercase=True, max_df=1.0, max_features=None, min_df=1,
                                 ngram_range=(1, 3), preprocessor=None, stop_words=None, strip_accents=None,
                                 tokenizer=None, token_pattern=r'(?u)\b\w\w+\b', vocabulary=None, )
    x = vectorizer.fit_transform(raw_documents=df['keywords'], )
    scores = {}
    clusters = {}
    init = ['k-means++', 'random'][1]
    max_iter = 15
    for n_clusters in range(2, 12):
        model = KMeans(algorithm='lloyd', copy_x=True, init=init, max_iter=max_iter, n_clusters=n_clusters,
                       n_init='auto', random_state=RANDOM_STATE, tol=0.0001, verbose=1, )
        model.fit(X=x, )
        scores[n_clusters] = silhouette_score(X=x, labels=model.labels_, metric='euclidean', sample_size=None,
                                              random_state=RANDOM_STATE, )
        clusters[n_clusters] = model.labels_
    best_key = max(scores, key=lambda key: scores[key])
    column = 'kmeans cluster'
    df[column] = clusters[best_key]
    logger.info(msg=df[column].value_counts().to_dict())
    logger.info(msg='k-means score: {:06.4f}'.format(scores[best_key]))
    return df


def add_tsne_components(df: DataFrame, columns: list, ) -> DataFrame:
    model = TSNE(angle=0.5, early_exaggeration=12.0, init='pca', learning_rate='auto', method='barnes_hut',
                 metric='euclidean', metric_params=None, min_grad_norm=1e-7, n_components=2, n_iter=500,
                 n_iter_without_progress=100, n_jobs=None, perplexity=30.0, random_state=RANDOM_STATE, verbose=2, )
    tsne_result = model.fit_transform(X=df[columns], )
    df['t-SNE x'] = tsne_result[:, 0]
    df['t-SNE y'] = tsne_result[:, 1]
    return df


def duration_seconds(duration: str, ) -> int:
    pieces = duration.replace('PT', '').replace('S', '').split('M')
    return 60 * int(pieces[0]) + int(pieces[1])


def load_settings(filename: str, ) -> dict:
    with open(encoding='utf-8', file=filename, mode='r') as input_fp:
        result = dict(load(fp=input_fp, ))
    return {key: value for key, value in result.items() if key in result['keys']}


def make_plot(plotting_package: str, df: DataFrame, short_name: str, ):
    logger = getLogger(name='make_plot', )
    logger.info(msg='plotting package: {}'.format(plotting_package, ), )
    if plotting_package not in {'matplotlib.pyplot', 'plotly', 'seaborn'}:
        raise NotImplementedError(plotting_package)
    elif plotting_package == 'matplotlib.pyplot':
        filename = OUTPUT_FOLDER + SCATTER_FILENAME
        scatter(data=df, x='published', y='log10_views', )
        title(label='YouTube user {} video date/count scatter'.format(short_name), )
        ylabel(ylabel='log10 views')
        logger.info(msg='saving plot to {}'.format(filename), )
        savefig(backend=None, bbox_inches=None, dpi='figure', edgecolor='auto', facecolor='auto', fname=filename,
                format='png', metadata=None, pad_inches=0.1, )
        close()
    elif plotting_package == 'plotly':
        custom_data = ['name', 'published', 'views_with_commas']
        for item in [
            {
                'color': 'log10_duration_seconds',
                'color_discrete': False,
                'filename': OUTPUT_FOLDER + SCATTER_PLOTLY_DATE_VIEWS_FILENAME.format(short_name),
                'labels': {'log10_views': 'log10 of views', 'published': 'Date Published', },
                'page_title': 'YouTube user {} date/count scatter'.format(short_name),
                'x': 'published',
                'y': 'log10_views',
            },
            {
                'color': 'age (days)',
                'color_discrete': False,
                'filename': OUTPUT_FOLDER + SCATTER_PLOTLY_DURATION_VIEWS_FILENAME.format(short_name),
                'labels': {'log10_duration_seconds': 'log10 of duration (sec)', 'log10_views': 'log10 of views'},
                'page_title': 'YouTube user {} duration/count scatter'.format(short_name),
                'x': 'log10_duration_seconds',
                'y': 'log10_views',
            },
            {
                'color': 'age (days)',
                'color_discrete': False,
                'filename': OUTPUT_FOLDER + SCATTER_PLOTLY_TSNE_FILENAME.format(short_name),
                'labels': {'log10_duration_seconds': 'log10 of duration (sec)', 'log10_views': 'log10 of views'},
                'page_title': 'YouTube user {} duration/count TSNE scatter'.format(short_name),
                'x': 't-SNE x',
                'y': 't-SNE y',
            },
            {
                'color': 'kmeans cluster',
                'color_discrete': True,
                'filename': OUTPUT_FOLDER + SCATTER_PLOTLY_KMEANS_FILENAME.format(short_name),
                'labels': {'log10_duration_seconds': 'log10 of duration (sec)', 'log10_views': 'log10 of views'},
                'page_title': 'YouTube user {} duration/count K-means scatter'.format(short_name),
                'x': 'log10_duration_seconds',
                'y': 'log10_views',
            },
            {
                'color': 'DBSCAN cluster',
                'color_discrete': True,
                'filename': OUTPUT_FOLDER + SCATTER_PLOTLY_DBSCAN_FILENAME.format(short_name),
                'labels': {'log10_duration_seconds': 'log10 of duration (sec)', 'log10_views': 'log10 of views'},
                'page_title': 'YouTube user {} duration/count DBSCAN scatter'.format(short_name),
                'x': 'log10_duration_seconds',
                'y': 'log10_views',
            },
        ]:
            if item['color_discrete']:
                df[item['color']] = df[item['color']].astype(str)
            figure = plotly_scatter(color=item['color'], color_continuous_scale=sequential.Viridis,
                                    color_discrete_sequence=qualitative.Alphabet, custom_data=custom_data,
                                    data_frame=df, labels=item['labels'], title=item['page_title'], x=item['x'],
                                    y=item['y'], )
            hover_template = '<br>'.join(
                ['video: %{customdata[0]}', 'date: %{customdata[1]}', 'views: %{customdata[2]}'])
            figure.update_traces(hovertemplate=hover_template, )
            filename = item['filename']
            logger.info(msg='saving plot to {}'.format(filename), )
            figure.write_html(filename)
    elif plotting_package == 'seaborn':
        filename = OUTPUT_FOLDER + SCATTERPLOT_FILENAME
        scatterplot(data=df, x='published', y='log10_views', )
        title(label='YouTube user {} video date/count scatter'.format(short_name), )
        ylabel(ylabel='log10 views')
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
    settings = load_settings(filename='youtube_load_and_plot.json', )
    logger.info(msg='settings: {}'.format(dumps(indent=4, obj=settings, sort_keys=True, ), ), )

    videos_df = read_csv(filepath_or_buffer=settings['input_data_file'], usecols=USECOLS, )
    videos_df['age (days)'] = videos_df['published'].apply(
        func=lambda x: (time_start.date() - to_datetime(x).date()).days)
    videos_df['duration_seconds'] = videos_df['duration'].apply(func=duration_seconds, )
    videos_df['log10_duration_seconds'] = videos_df['duration_seconds'].apply(func=lambda x: log10(1 + x), )
    videos_df['views_with_commas'] = videos_df['views'].apply(func=lambda x: '{:,}'.format(x), )
    videos_df['year_published'] = videos_df['published'].apply(func=lambda x: x.split('-')[0])
    logger.info(msg='built initial DataFrame')
    videos_df = add_dbscan_cluster(df=videos_df, )
    logger.info(msg='added DBSCAN clusters')
    videos_df = add_kmeans_cluster(df=videos_df, )
    logger.info(msg='added k-means clusters')
    tsne_columns = ['age (days)', 'duration_seconds', 'views', ]
    videos_df = add_tsne_components(columns=tsne_columns, df=videos_df, )
    logger.info(msg='data has shape: {}'.format(videos_df.shape, ))

    make_plot(df=videos_df.sort_values(by='published', ),
              short_name=[piece for piece in settings['input_data_file'].split('-') if piece.startswith('@')][0],
              plotting_package=settings['plotting_package'], )
    time_seconds = (now() - time_start).total_seconds()
    logger.info(msg='done: {:02d}:{:05.2f}'.format(int(time_seconds // 60), time_seconds % 60, ))


OUTPUT_FOLDER = './result/'
RANDOM_STATE = 1
SCATTER_FILENAME = 'youtube.matplotlib.scatter.png'
SCATTER_PLOTLY_DBSCAN_FILENAME = 'youtube.plotly.{}.DBSCAN.scatter.html'
SCATTER_PLOTLY_DATE_VIEWS_FILENAME = 'youtube.plotly.{}.date-views.scatter.html'
SCATTER_PLOTLY_DURATION_VIEWS_FILENAME = 'youtube.plotly.{}.duration-views.scatter.html'
SCATTER_PLOTLY_KMEANS_FILENAME = 'youtube.plotly.{}.kmeans.scatter.html'
SCATTER_PLOTLY_TSNE_FILENAME = 'youtube.plotly.{}.tsne.scatter.html'
SCATTERPLOT_FILENAME = 'youtube.seaborn.scatterplot.png'
USECOLS = ['log10_views', 'name', 'published', 'views', 'duration', 'keywords', ]

if __name__ == '__main__':
    main()
