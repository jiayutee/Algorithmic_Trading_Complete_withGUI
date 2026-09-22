from datetime import datetime, timezone
from unittest.mock import Mock
import pandas as pd
from core.news_sources import NewsItem
from core.news_context import build_snapshot, fetch_snapshot, observed_move, candles_from_figure
from dash_app.news_context import context_figure, safe_url, filtered_events, event_card


def item(headline='Bitcoin exchange suffers hack', time='2026-09-01T12:00:00Z'):
    return NewsItem(datetime_utc=pd.Timestamp(time).to_pydatetime(), source='Test publisher', headline=headline)


def test_snapshot_removes_duplicates_invalid_future_and_orders_publications():
    items=[item(),item(),item('Future', '2030-01-01T00:00:00Z'),item('Older','2026-08-31T00:00:00Z')]
    report=build_snapshot(items,'BTCUSDT',now=datetime(2026,9,2,tzinfo=timezone.utc))
    assert len(report['events'])==2
    assert report['events'][0]['headline']==items[0].headline
    assert report['events'][0]['interpretation']['evidence']['source']=='Test publisher'


def test_fast_snapshot_does_not_score_or_persist():
    pipe=Mock()
    pipe._fetch_all_sources.return_value=[item()]
    pipe._prefilter.return_value=[item()]
    pipe.source_status.return_value=[{'name':'rss','status':'ok'}]
    report=fetch_snapshot('BTCUSDT',pipe)
    pipe._fetch_all_sources.assert_called_once_with('Bitcoin',40,ticker_query='BTCUSDT')
    pipe.fetch_news_items.assert_not_called()
    pipe._open_store.assert_not_called()
    pipe.sentiment_analyzer.analyze_many.assert_not_called()
    assert report['sources'][0]['status']=='ok'


def chart():
    return {'data':[{'type':'candlestick','x':['2026-09-01T00:00Z','2026-09-02T00:00Z','2026-09-03T00:00Z'],
                    'open':[99,101,109],'high':[102,112,122],'low':[98,100,108],'close':[100,110,120]}]}


def test_observed_move_uses_prior_bar_and_requires_both_sides():
    df=candles_from_figure(chart())
    move=observed_move(df,'2026-09-02T00:00Z')
    assert round(move['percent'])==20 # prior close 100, not publication-bar close 110
    assert observed_move(df,'2026-08-20') is None
    assert observed_move(df,'2026-09-04') is None


def test_chart_does_not_mutate_main_or_invent_future_prices():
    original=chart()
    events=build_snapshot([item()], 'BTCUSDT',now=datetime(2026,9,4,tzinfo=timezone.utc))['events']
    fig,df=context_figure(original,events,events[0]['id'])
    assert len(original['data'])==1
    assert len(fig.data[0].close)==3
    assert len(fig.layout.shapes)==2
    assert list(fig.data[1].customdata)==[events[0]['id']]
    assert len(context_figure(None,[],None)[0].layout.annotations)==1


def test_unsafe_source_links_are_not_rendered():
    assert safe_url('javascript:alert(1)') is None
    assert safe_url('https://publisher.example/story')
    event=build_snapshot([item()], 'BTCUSDT',now=datetime(2026,9,4,tzinfo=timezone.utc))['events'][0]
    event['url']='javascript:alert(1)'
    assert 'javascript:' not in str(event_card(event,True).to_plotly_json())


def test_real_plotly_serialized_chart_arrays_are_decoded():
    import json
    from core.chart_builder import build_candlestick_figure
    df=pd.DataFrame({'Open':[100.,110.], 'High':[102.,112.], 'Low':[98.,108.], 'Close':[101.,111.]},
                    index=pd.date_range('2026-09-01', periods=2, tz='UTC'))
    serialized=json.loads(build_candlestick_figure(df,symbol='ETHUSDT').to_json())
    decoded=candles_from_figure(serialized)
    assert decoded['close'].tolist()==[101.,111.]
    assert not context_figure(serialized,[],None)[1].empty


def test_intra_bar_publication_does_not_use_that_bars_close_as_baseline():
    move=observed_move(candles_from_figure(chart()),'2026-09-02T12:00Z')
    assert round(move['percent'])==20
