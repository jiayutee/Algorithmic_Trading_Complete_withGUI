"""Chart-linked news context, isolated from trading and live-tick callbacks."""
from __future__ import annotations
from urllib.parse import urlsplit
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State, ctx, no_update
from core.chart_builder import THEME
from core.news_context import (fetch_snapshot, candles_from_figure, observed_move, candle_step, ai_research_for_event,
                               scenario_paths, event_mix, SCENARIO_HORIZONS)

COLORS = {'bullish': '#42d9a1', 'bearish': '#ff6b86', 'unknown': '#a9b6cd', 'mixed': '#f2c66d'}


def context_panel():
    return html.Div(className='news-context', children=[
        dcc.Store(id='context-snapshot'), dcc.Store(id='context-candles'), dcc.Store(id='context-ai-research'),
        html.Div(className='context-heading', children=[
            html.Div([html.Small('MARKET CONTEXT'), html.H3('The story behind the candles'),
                      html.P('Explore reported events, possible implications and observed price changes.')]),
            html.Button('Refresh context', id='context-refresh', n_clicks=0)]),
        html.Div(id='context-status', role='status'),
        html.Div(className='context-controls', children=[
            dcc.RadioItems(id='context-filter', options=[{'label': l, 'value': v} for l,v in
                [('All events','all'),('Bullish case','bullish'),('Bearish case','bearish'),('Unclear / mixed','unclear')]],
                value='all', inline=True),
            dcc.Dropdown(id='context-selected', options=[], placeholder='Select an event to inspect', clearable=True),
            html.Button('Get AI research on selected event', id='context-ai-button', n_clicks=0)]),
        html.Div(id='context-ai-status', className='context-note', role='status'),
        html.Div(className='context-scenario-controls', children=[
            dcc.Checklist(id='context-scenarios', options=[{'label': ' Show scenario projections (illustrative, not a forecast)', 'value': 'on'}],
                          value=['on']),
            html.Span('Horizon (bars):'),
            dcc.RadioItems(id='context-horizon', options=[{'label': str(h), 'value': h} for h in SCENARIO_HORIZONS],
                           value=14, inline=True)]),
        dcc.Loading(children=html.Div(className='context-grid', children=[
            html.Div([dcc.Graph(id='context-chart', config={'displaylogo':False}),
                      html.Div(id='context-move', className='context-note')]),
            html.Div(id='context-timeline', className='context-timeline')])),
        html.Details([html.Summary('Feed coverage'), html.Div(id='context-health')]),
        html.P('Interpretations are conditional rule-based context. Optional AI research (Groq, opt-in via '
               'GROQ_API_KEY) is a hosted-LLM hypothesis from the same text, fetched on demand, never a forecast. '
               'Dotted scenario lines are volatility-scaled illustrations, not forecasts. Price changes do not establish causation. Scheduled earnings are shown in News & Earnings; '
               'no macro calendar is connected.', className='context-note')])


def filtered_events(snapshot, mode):
    events = (snapshot or {}).get('events', [])
    return [e for e in events if mode == 'all' or
            (mode == 'unclear' and e['interpretation']['conditional_bias'] not in ('bullish','bearish')) or
            e['interpretation']['conditional_bias'] == mode]


def safe_url(value):
    try:
        p = urlsplit(value or '')
        return value if p.scheme in ('http','https') and p.netloc else None
    except ValueError:
        return None


def ai_research_block(ai_research):
    if ai_research is None:
        return html.P('No AI research fetched for this event yet. Use "Get AI research on selected '
                       'event" above (needs GROQ_API_KEY; unavailable falls back silently).', className='context-note')
    return html.Div(className='context-ai-research', children=[
        html.Strong('AI research (' + ai_research['method'] + ')'),
        html.P(ai_research['reasoning']),
        html.P('Reading: ' + ai_research['conditional_bias'].capitalize() +
               ' · confidence {:.0%}'.format(ai_research['confidence'])),
        html.Strong('How this could be wrong'), html.P(ai_research['contrary_view']),
        html.Strong('Verify before acting'), html.Ul([html.Li(w) for w in ai_research['corroboration_needed']]) if
            ai_research['corroboration_needed'] else html.P('Not supplied by the model.'),
        html.P(ai_research['limitations'], className='context-note')])


def event_card(event, selected, ai_research=None):
    i = event['interpretation']; impact = i['conditional_bias']; color = COLORS.get(impact, COLORS['unknown'])
    url = safe_url(event.get('url'))
    title = html.A(event['headline'], href=url, target='_blank', rel='noopener noreferrer') if url else html.Span(event['headline'])
    return html.Article(className='context-event selected' if selected else 'context-event',
        style={'borderLeftColor':color}, children=[
            html.Div([html.Time(pd.Timestamp(event['time']).strftime('%d %b · %H:%M UTC')),
                      html.Span(impact.capitalize() + (' case' if impact in ('bullish','bearish') else ''),
                                className='context-badge', style={'color':color})]),
            html.H4(title), html.Small(event['source']),
            html.Details(open=selected, children=[html.Summary('Interpretation'),
                html.Strong('Why it may matter'), html.P(i['mechanism']),
                html.Strong('What could change the picture'), html.P(i['counterargument']),
                html.Strong('Watch next'), html.Ul([html.Li(w) for w in i['what_to_watch']]),
                html.Strong('Evidence available'), html.Blockquote(i['evidence']['excerpt']),
                html.P(' · '.join([i['event_category'], i['relevance'], i['method']]), className='context-note'),
                html.P('Headline tone: '+i['headline_tone']['label']+' · '+i['headline_tone']['model'], className='context-note'),
                html.P(i['conditional_bias_basis'], className='context-note'),
                html.P(' '.join(i['limitations']) if isinstance(i['limitations'],list) else i['limitations'], className='context-note'),
                ai_research_block(ai_research) if selected else None])])


SCENARIO_COLORS = {'bullish': '#42d9a1', 'bearish': '#ff6b86', 'range': '#f2c66d'}


def add_scenarios(fig, paths, mix):
    """Dotted scenario lines plus the range cone; every legend entry says these are scenarios, not forecasts."""
    t = paths['times']
    fig.add_trace(go.Scatter(x=t + t[::-1], y=paths['range_high'] + paths['range_low'][::-1], fill='toself',
                  fillcolor='rgba(242,198,109,0.10)', line={'width': 0}, hoverinfo='skip', showlegend=False))
    for key, label, ys in (('bullish', 'Bullish scenario', paths['bullish']), ('bearish', 'Bearish scenario', paths['bearish']),
                           ('range', 'Unclear / range scenario', paths['range_mid'])):
        n = mix['bullish'] if key == 'bullish' else mix['bearish'] if key == 'bearish' else mix['unclear']
        fig.add_trace(go.Scatter(x=t, y=ys, mode='lines', line={'color': SCENARIO_COLORS[key], 'width': 2, 'dash': 'dot'},
                      name='{} ({} {} events)'.format(label, n, key if key != 'range' else 'unclear/mixed'),
                      hovertemplate=label + ': %{y:,.2f}<br>%{x}<extra></extra>'))


def context_figure(figure, events, selected, scenarios=None, horizon=14, mix=None):
    candles = candles_from_figure(figure)
    fig = go.Figure()
    if not candles.empty:
        fig.add_trace(go.Candlestick(x=candles.index, open=candles.open, high=candles.high,
                      low=candles.low, close=candles.close, name='Loaded price',
                      increasing_line_color=COLORS['bullish'], decreasing_line_color=COLORS['bearish']))
        for impact, color in COLORS.items():
            points = []
            for event in events:
                t = pd.Timestamp(event['time'])
                if event['interpretation']['conditional_bias'] != impact or t < candles.index[0] or t >= candles.index[-1] + candle_step(candles):
                    continue
                idx = candles.index.searchsorted(t, side='right')-1
                points.append((t, candles.high.iloc[idx]*1.006, event))
            if points:
                fig.add_trace(go.Scatter(x=[p[0] for p in points], y=[p[1] for p in points], mode='markers',
                    marker={'color':color,'size':11,'symbol':'diamond'}, name=impact.capitalize()+' context',
                    customdata=[p[2]['id'] for p in points], text=[p[2]['headline'] for p in points],
                    hovertemplate='%{text}<br>%{x}<extra></extra>'))
        event = next((e for e in events if e['id']==selected), None)
        if event and candles.index[0] <= pd.Timestamp(event['time']) < candles.index[-1] + candle_step(candles):
            fig.add_shape(type='line', x0=event['time'], x1=event['time'], y0=0, y1=1, yref='paper',
                          line={'color':'#8eabff','width':2,'dash':'dot'})
            fig.add_shape(type='rect', x0=event['time'], x1=max(pd.Timestamp(event['time']),candles.index[-1]), y0=0, y1=1, yref='paper',
                          fillcolor='#8eabff', opacity=.07, line_width=0)
        paths = scenario_paths(candles, horizon) if scenarios else None
        if paths:
            add_scenarios(fig, paths, mix or {'bullish': 0, 'bearish': 0, 'unclear': 0})
            # Open zoomed on recent bars + the projection so the lines are legible (the full history stays one drag away).
            fig.update_xaxes(range=[candles.index[max(0, len(candles) - max(45, 4 * horizon))], paths['times'][-1] + candle_step(candles)])
    else:
        fig.add_annotation(text='Load a chart, then refresh context.', x=.5,y=.5,xref='paper',yref='paper',showarrow=False)
    fig.update_layout(template='plotly_dark',paper_bgcolor='#111722',plot_bgcolor='#111722',height=430,
                      margin={'l':45,'r':20,'t':20,'b':40},xaxis_rangeslider_visible=False,
                      legend={'orientation':'h','y':1.1},uirevision='context',font={'color':'#cbd4e5'})
    return fig, candles


def register_context_callbacks(app):
    @app.callback(Output('context-snapshot','data'), Output('context-candles','data'),
                  Input('context-refresh','n_clicks'), State('active-symbol-store','data'),
                  State('main-chart','figure'), prevent_initial_call=True,
                  running=[(Output('context-refresh','disabled'), True, False)])
    def refresh(n, symbol, figure):
        if not symbol:
            return {'events':[], 'error':'Load a chart before refreshing context.'}, None
        try:
            snapshot = fetch_snapshot(symbol)
        except Exception:
            snapshot = {'symbol':symbol,'events':[], 'sources':[], 'error':'News is unavailable. Refresh to retry; no cached results are shown.'}
        return snapshot, figure

    @app.callback(Output('context-selected','options'), Output('context-selected','value'),
                  Input('context-snapshot','data'), Input('context-filter','value'), Input('context-chart','clickData'))
    def select(snapshot, mode, click):
        events = filtered_events(snapshot, mode)
        ids = {e['id'] for e in events}
        selected = events[0]['id'] if events else None
        if ctx.triggered_id == 'context-chart' and click:
            value = (click.get('points') or [{}])[0].get('customdata')
            if value in ids: selected = value
        return [{'label':pd.Timestamp(e['time']).strftime('%d %b')+' · '+e['headline'], 'value':e['id']} for e in events], selected

    @app.callback(Output('context-ai-research','data'), Output('context-ai-status','children'),
                  Input('context-ai-button','n_clicks'),
                  State('context-selected','value'), State('context-snapshot','data'), State('active-symbol-store','data'),
                  prevent_initial_call=True, running=[(Output('context-ai-button','disabled'), True, False)])
    def get_ai_research(n, selected, snapshot, active_symbol):
        if not selected or not snapshot:
            return no_update, 'Select an event first.'
        event = next((e for e in snapshot.get('events', []) if e['id'] == selected), None)
        if event is None:
            return no_update, 'Selected event not found in the current snapshot.'
        result = ai_research_for_event(event, active_symbol)
        if result is None:
            return {'event_id': selected, 'symbol': active_symbol, 'result': None}, (
                'AI research unavailable for this event (no GROQ_API_KEY configured, or the call failed/timed out). '
                'Deterministic interpretation above is unaffected.')
        return {'event_id': selected, 'symbol': active_symbol, 'result': result}, 'AI research updated below.'

    @app.callback(Output('context-chart','figure'), Output('context-timeline','children'),
                  Output('context-move','children'), Output('context-status','children'), Output('context-health','children'),
                  Input('context-snapshot','data'), Input('context-candles','data'),
                  Input('context-filter','value'), Input('context-selected','value'), Input('active-symbol-store','data'),
                  Input('context-ai-research','data'), Input('context-scenarios','value'), Input('context-horizon','value'))
    def render(snapshot, figure, mode, selected, active_symbol, ai_research, scenarios, horizon):
        if not snapshot or snapshot.get('symbol') != active_symbol:
            return context_figure(None,[],None)[0], [], '', 'Refresh context for the loaded symbol.', []
        events = filtered_events(snapshot, mode)
        fig, candles = context_figure(figure, events, selected, bool(scenarios), horizon or 14, event_mix(snapshot.get('events')))
        event = next((e for e in events if e['id']==selected),None)
        move = observed_move(candles, event['time']) if event else None
        movement = ('Observed close-to-close move: {:+.2f}% · candle open times {} → {}. Last loaded candle may still be forming.'
                    .format(move['percent'], move['start'], move['end'])) if move else 'Price comparison unavailable: select an event within the loaded candle range.'
        ai_result = (ai_research or {}).get('result') if (ai_research or {}).get('event_id') == selected and \
            (ai_research or {}).get('symbol') == active_symbol else None
        rows = [event_card(e, e['id']==selected, ai_result if e['id']==selected else None) for e in events] or \
            [html.P('No events match this view. Try All events or refresh.')]
        health = [html.Div(str(s.get('name'))+' · '+str(s.get('status','unknown'))) for s in snapshot.get('sources',[])]
        status = snapshot.get('error') or '{} · {} events · Retrieved {} · candle snapshot from last context refresh'.format(
            snapshot.get('symbol',''),len(events),snapshot.get('as_of',''))
        hidden = snapshot.get('hidden') or {}
        if not snapshot.get('error') and sum(hidden.values()):
            status += ' · hidden as not market context: {} off-topic, {} explainer/reference pages, {} undated web results'.format(
                hidden.get('off_topic', 0), hidden.get('evergreen', 0), hidden.get('undated', 0))
        paths = scenario_paths(candles, horizon or 14) if scenarios else None
        if paths:
            mix = event_mix(snapshot.get('events'))
            movement += (' Scenario lines (dotted) are volatility-scaled illustrations, not forecasts: one standard deviation '
                         '(daily-return sigma {:.2%}) over {} bars is about {:+.1%} / {:+.1%}. Reported cases in view: {} bullish, {} bearish, {} unclear/mixed; '
                         'the counts describe the news, they do not tilt the lines or make either more likely.').format(
                paths['sigma'], paths['horizon'], paths['bullish'][-1] / paths['last'] - 1, paths['bearish'][-1] / paths['last'] - 1,
                mix['bullish'], mix['bearish'], mix['unclear'])
        elif scenarios:
            movement += ' Scenario lines need at least 30 loaded candles.'
        return fig, rows, movement, status, health
