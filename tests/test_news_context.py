from datetime import datetime, timezone
from unittest.mock import Mock
import pandas as pd
from core.news_sources import NewsItem
from core.news_context import build_snapshot, fetch_snapshot, observed_move, candles_from_figure
from dash_app.news_context import context_figure, safe_url, filtered_events, event_card


def item(headline='Bitcoin exchange suffers hack', time='2026-09-01T12:00:00Z'):
    return NewsItem(datetime_utc=pd.Timestamp(time).to_pydatetime(), source='Test publisher', headline=headline)


def test_snapshot_removes_duplicates_invalid_future_and_orders_publications():
    items=[item(),item(),item('Future', '2030-01-01T00:00:00Z'),item('Older Bitcoin story','2026-08-31T00:00:00Z')]
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


def test_ai_research_for_event_returns_none_without_event():
    from core.news_context import ai_research_for_event
    assert ai_research_for_event(None, 'BTCUSDT') is None


def test_ai_research_for_event_passes_interpretation_fields_through():
    from unittest.mock import patch
    from core.news_context import ai_research_for_event
    event = build_snapshot([item('AAPL beats earnings estimates')], 'AAPL',
                            now=datetime(2026, 9, 2, tzinfo=timezone.utc))['events'][0]
    with patch('core.news_context.research_event', return_value={'method': 'ai-research-groq-test'}) as mock_call:
        result = ai_research_for_event(event, 'AAPL')
    assert result == {'method': 'ai-research-groq-test'}
    _, kwargs = mock_call.call_args
    assert kwargs['symbol'] == 'AAPL'
    assert kwargs['headline'] == event['headline']
    assert kwargs['event_category'] == event['interpretation']['event_category']
    assert kwargs['sentiment_label'] == event['interpretation']['headline_tone']['label']


def test_event_card_shows_ai_research_placeholder_when_none_for_selected():
    event = build_snapshot([item()], 'BTCUSDT', now=datetime(2026, 9, 2, tzinfo=timezone.utc))['events'][0]
    rendered = str(event_card(event, True, None).to_plotly_json())
    assert 'No AI research fetched' in rendered


def test_event_card_renders_ai_research_when_present():
    event = build_snapshot([item()], 'BTCUSDT', now=datetime(2026, 9, 2, tzinfo=timezone.utc))['events'][0]
    ai_research = {'method': 'ai-research-groq-llama-3.3-70b-versatile', 'reasoning': 'Because of the supplied text.',
                    'conditional_bias': 'bearish', 'confidence': 0.4, 'contrary_view': 'Could be contained.',
                    'corroboration_needed': ['Check official statement'], 'limitations': 'Not verified.'}
    rendered = str(event_card(event, True, ai_research).to_plotly_json())
    assert 'Because of the supplied text.' in rendered
    assert 'Check official statement' in rendered


def _src(headline, source='CoinDesk'):
    return NewsItem(datetime_utc=pd.Timestamp('2026-09-01T12:00:00Z').to_pydatetime(), source=source, headline=headline)


def test_explainer_reference_and_undated_search_pages_are_kept_off_the_timeline():
    items = [_src('Bitcoin ETFs have erased a $5.8 billion hole'),
             _src('What Bitcoin Is And How It Works - Forbes', 'Forbes'),
             _src('How does Bitcoin work? - Bitcoin', 'duckduckgo'),
             _src('Bitcoin & Crypto Basics: Beginner Guides to Get Started', 'duckduckgo'),
             _src('BTC USD — Bitcoin Price and Chart — TradingView', 'duckduckgo'),
             _src('Bitcoin - Wikipedia', 'duckduckgo'),
             _src('Top 9 Daytrading Plattformen - Top 9 Trading Broker 2026', 'duckduckgo')]
    report = build_snapshot(items, 'BTCUSDT', now=datetime(2026, 9, 2, tzinfo=timezone.utc))
    assert [e['headline'] for e in report['events']] == ['Bitcoin ETFs have erased a $5.8 billion hole']
    assert report['hidden']['undated'] == 5 and report['hidden']['evergreen'] == 1


def test_unrelated_and_other_asset_news_is_counted_but_not_shown():
    items = [_src('Bitcoin Surges Past $87,000 After CLARITY Act Failure: 3 Reasons Why'),
             _src('U.S. retail sales rise to 6%'),
             _src('Zcash Recent Surge: Is $5,000 the Next Target?'),
             _src("What to know about Anthropic's new $12B computing deal")]
    report = build_snapshot(items, 'BTCUSDT', now=datetime(2026, 9, 2, tzinfo=timezone.utc))
    shown = [e['headline'] for e in report['events']]
    assert 'Bitcoin Surges Past $87,000 After CLARITY Act Failure: 3 Reasons Why' in shown
    assert 'U.S. retail sales rise to 6%' in shown          # market-wide data stays
    assert not any('Zcash' in h or 'Anthropic' in h for h in shown)
    assert report['hidden']['off_topic'] == 2 and report['hidden']['evergreen'] == 0


def _candles(n=60, vol=0.02, seed=1):
    import numpy as np
    idx = pd.date_range('2026-01-01', periods=n, freq='D', tz='UTC')
    close = 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, vol, n)))
    return pd.DataFrame({'open': close, 'high': close * 1.01, 'low': close * 0.99, 'close': close}, index=idx)


def test_scenario_fan_starts_at_last_close_tilts_the_median_and_is_not_a_straight_line():
    import math
    from core.news_context import scenario_fan
    c = _candles()
    f = scenario_fan(c, 14)
    last = float(c['close'].iloc[-1])
    assert f['last'] == last and len(f['times']) == 15 and f['times'][-1] == c.index[-1] + pd.Timedelta(days=14)
    sc = f['scenarios']
    for name in ('bullish', 'bearish', 'range'):
        assert sc[name]['median'][0] == sc[name]['sample'][0] == last
        assert all(a <= b <= c_ for a, b, c_ in zip(sc[name]['q25'], sc[name]['median'], sc[name]['q75']))
        assert all(lo <= a and b <= hi for lo, a, b, hi in zip(sc[name]['q10'], sc[name]['q25'], sc[name]['q75'], sc[name]['q90']))
    assert sc['bullish']['median'][-1] > sc['range']['median'][-1] > sc['bearish']['median'][-1]
    assert sc['bullish']['median'][-1] / last - 1 > 0.5 * (math.exp(f['sigma'] * math.sqrt(14)) - 1)
    sample = sc['bullish']['sample']
    steps = [b - a for a, b in zip(sample, sample[1:])]
    assert len({round(x, 6) for x in steps}) > 5                       # jagged: the increments are not all equal


def test_scenario_fan_is_deterministic_and_the_shocks_are_shared():
    from core.news_context import scenario_fan
    c = _candles()
    a, b = scenario_fan(c, 14), scenario_fan(c, 14)
    assert a['scenarios']['bullish']['median'] == b['scenarios']['bullish']['median']
    assert scenario_fan(c, 14, seed=5)['scenarios']['range']['sample'] != a['scenarios']['range']['sample']


def test_scenarios_need_history_and_a_moving_price():
    from core.news_context import scenario_fan
    assert scenario_fan(_candles(20), 14) is None
    assert scenario_fan(_candles(60, vol=0.0), 14) is None


def test_event_mix_counts_cases_without_turning_them_into_probabilities():
    from core.news_context import event_mix
    events = [{'interpretation': {'conditional_bias': b}} for b in ('bullish', 'bullish', 'bearish', 'unknown', 'mixed')]
    assert event_mix(events) == {'bullish': 2, 'bearish': 1, 'unclear': 2}


def test_context_figure_draws_three_labelled_scenarios_only_when_asked():
    import json
    from core.chart_builder import build_candlestick_figure
    df = pd.DataFrame({'Open': _candles()['open'], 'High': _candles()['high'], 'Low': _candles()['low'], 'Close': _candles()['close']})
    serialized = json.loads(build_candlestick_figure(df, symbol='BTCUSDT').to_json())
    on = context_figure(serialized, [], None, True, 14, {'bullish': 3, 'bearish': 1, 'unclear': 2})[0]
    names = [t.name for t in on.data if t.name]
    assert any(n.startswith('Bullish scenario (3 bullish') for n in names)
    assert any(n.startswith('Bearish scenario (1 bearish') for n in names)
    assert any(n.startswith('Unclear / range scenario (2 unclear/mixed') for n in names)
    assert len(on.data) == 1 + 3 * 3                                    # candles + (band, example path, median) x 3 scenarios
    off = context_figure(serialized, [], None, False)[0]
    assert not any('scenario' in (t.name or '') for t in off.data)
