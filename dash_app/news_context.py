"""Chart-linked news context, isolated from trading and live-tick callbacks."""
from __future__ import annotations
from urllib.parse import urlsplit
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State, ctx, no_update
from core.chart_builder import THEME
from core.news_context import (fetch_snapshot, candles_from_figure, observed_move, candle_step, ai_research_for_event,
                               scenario_fan, event_mix, effective_bias, SCENARIO_HORIZONS)

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
            dcc.Checklist(id='context-scenarios', options=[{'label': ' Show scenario fans (simulated, not a forecast)', 'value': 'on'}],
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
               'Scenario fans are volatility-scaled simulations, not forecasts. Price changes do not establish causation. Scheduled earnings are shown in News & Earnings; '
               'no macro calendar is connected.', className='context-note')])


def filtered_events(snapshot, mode):
    events = (snapshot or {}).get('events', [])
    return [e for e in events if mode == 'all' or
            (mode == 'unclear' and effective_bias(e)[0] not in ('bullish','bearish')) or
            effective_bias(e)[0] == mode]


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


def model_reading_block(event):
    r = event.get('model_reading')
    if not r:
        return None
    shown = effective_bias(event)[1] == 'model'
    return html.Div(className='context-ai-research', children=[
        html.Strong('Model reading (' + r['method'].replace('model-reading-groq-', 'Groq ') + ', untested label)'),
        html.P('{} · {:.0%} confidence{}'.format(r['bias'].capitalize(), r['confidence'],
               '' if shown or r['bias'] == 'unclear' else ' · below the 60% threshold, so not used to colour this event')),
        html.P(r['reason']),
        html.P('A labelling aid from the headline text only; it has not been tested against price moves and does not tilt the scenario fans.', className='context-note')])


def event_card(event, selected, ai_research=None):
    i = event['interpretation']; impact, source = effective_bias(event); color = COLORS.get(impact, COLORS['unknown'])
    url = safe_url(event.get('url'))
    title = html.A(event['headline'], href=url, target='_blank', rel='noopener noreferrer') if url else html.Span(event['headline'])
    return html.Article(className='context-event selected' if selected else 'context-event',
        style={'borderLeftColor':color}, children=[
            html.Div([html.Time(pd.Timestamp(event['time']).strftime('%d %b · %H:%M UTC')),
                      html.Span(impact.capitalize() + (' case' if impact in ('bullish','bearish') else '') + (' · model' if source == 'model' else ''),
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
                model_reading_block(event),
                ai_research_block(ai_research) if selected else None])])


SCENARIO_COLORS = {'bullish': '#42d9a1', 'bearish': '#ff6b86', 'range': '#f2c66d'}


def _rgba(hex_color, alpha):
    h = hex_color.lstrip('#')
    return 'rgba({},{},{},{})'.format(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)


def add_scenarios(fig, fan, mix):
    """Per scenario: shaded middle-50% band, dotted median, and one jagged example path (all toggled together in the legend).
    Every legend entry says these are scenarios, not forecasts."""
    t = fan['times']
    for key, label in (('bullish', 'Bullish scenario'), ('bearish', 'Bearish scenario'), ('range', 'Unclear / range scenario')):
        sc, color = fan['scenarios'][key], SCENARIO_COLORS[key]
        n = mix['bullish'] if key == 'bullish' else mix['bearish'] if key == 'bearish' else mix['unclear']
        fig.add_trace(go.Scatter(x=t + t[::-1], y=sc['q75'] + sc['q25'][::-1], fill='toself', fillcolor=_rgba(color, 0.13),
                      line={'width': 0}, hoverinfo='skip', showlegend=False, legendgroup=key))
        fig.add_trace(go.Scatter(x=t, y=sc['sample'], mode='lines', line={'color': _rgba(color, 0.75), 'width': 1.2},
                      showlegend=False, legendgroup=key, hovertemplate=label + ' (one example path): %{y:,.2f}<br>%{x}<extra></extra>'))
        fig.add_trace(go.Scatter(x=t, y=sc['median'], mode='lines', line={'color': color, 'width': 2, 'dash': 'dot'},
                      legendgroup=key, name='{} ({} {} events)'.format(label, n, key if key != 'range' else 'unclear/mixed'),
                      hovertemplate=label + ' median: %{y:,.2f}<br>%{x}<extra></extra>'))


def context_figure(figure, events, selected, scenarios=None, horizon=14, mix=None):
    candles = candles_from_figure(figure)
    fig = go.Figure()
    if not candles.empty:
        fig.add_trace(go.Candlestick(x=candles.index, open=candles.open, high=candles.high,
                      low=candles.low, close=candles.close, name='Loaded price',
                      increasing_line_color=COLORS['bullish'], decreasing_line_color=COLORS['bearish']))
        for impact, color in COLORS.items():
            for source, symbol_, tag in (('rules', 'diamond', 'context (rules)'), ('model', 'circle-open', 'reading (model, untested)')):
                points = []
                for event in events:
                    t = pd.Timestamp(event['time'])
                    bias, src = effective_bias(event)
                    if bias != impact or (src or 'rules') != source or t < candles.index[0] or t >= candles.index[-1] + candle_step(candles):
                        continue
                    idx = candles.index.searchsorted(t, side='right')-1
                    points.append((t, candles.high.iloc[idx]*1.006, event))
                if points:
                    fig.add_trace(go.Scatter(x=[p[0] for p in points], y=[p[1] for p in points], mode='markers',
                        marker={'color':color,'size':11 if source == 'rules' else 12,'symbol':symbol_, 'line': {'width': 2, 'color': color}},
                        name=impact.capitalize()+' '+tag,
                        customdata=[p[2]['id'] for p in points], text=[p[2]['headline'] for p in points],
                        hovertemplate='%{text}<br>%{x}<extra></extra>'))
        event = next((e for e in events if e['id']==selected), None)
        if event and candles.index[0] <= pd.Timestamp(event['time']) < candles.index[-1] + candle_step(candles):
            fig.add_shape(type='line', x0=event['time'], x1=event['time'], y0=0, y1=1, yref='paper',
                          line={'color':'#8eabff','width':2,'dash':'dot'})
            fig.add_shape(type='rect', x0=event['time'], x1=max(pd.Timestamp(event['time']),candles.index[-1]), y0=0, y1=1, yref='paper',
                          fillcolor='#8eabff', opacity=.07, line_width=0)
        fan = scenario_fan(candles, horizon) if scenarios else None
        if fan:
            add_scenarios(fig, fan, mix or {'bullish': 0, 'bearish': 0, 'unclear': 0})
            # Open zoomed on recent bars + the projection so the fans are legible (the full history stays one drag away).
            fig.update_xaxes(range=[candles.index[max(0, len(candles) - max(45, 4 * horizon))], fan['times'][-1] + candle_step(candles)])
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
            snapshot = fetch_snapshot(symbol, model_readings=True)
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
        mr = snapshot.get('model_readings') or {}
        if not snapshot.get('error') and mr.get('status') not in (None, 'off'):
            status += ' · model readings {}/{}{}'.format(mr.get('received', 0), mr.get('requested', 0),
                                                         '' if mr.get('status') in ('ok', 'partial') else ' (unavailable: rate limit or error; rule labels only)')
        elif not snapshot.get('error') and mr.get('status') == 'off':
            status += ' · model readings off (set GROQ_API_KEY)'
        hidden = snapshot.get('hidden') or {}
        if not snapshot.get('error') and sum(hidden.values()):
            status += ' · hidden as not market context: {} off-topic, {} explainer/reference pages, {} undated web results'.format(
                hidden.get('off_topic', 0), hidden.get('evergreen', 0), hidden.get('undated', 0))
        fan = scenario_fan(candles, horizon or 14) if scenarios else None
        if fan:
            mix = event_mix(snapshot.get('events'))
            sc = fan['scenarios']
            movement += (' Scenario fans are volatility-scaled simulations, not forecasts: {} paths per scenario are resampled from this '
                         'chart\'s own daily moves (sigma {:.2%}); the shaded band is the middle 50%, the dotted line the median, and the '
                         'jagged line one example path. Median after {} bars: {:+.1%} bullish / {:+.1%} bearish / {:+.1%} range. Reported '
                         'cases in view: {} bullish, {} bearish, {} unclear/mixed (rules plus confident, untested model readings); the counts describe the news, they do not tilt the '
                         'fans or make either more likely (no reading has been shown to predict price: Phase 13.1).').format(
                fan['n_paths'], fan['sigma'], fan['horizon'], sc['bullish']['median'][-1] / fan['last'] - 1,
                sc['bearish']['median'][-1] / fan['last'] - 1, sc['range']['median'][-1] / fan['last'] - 1,
                mix['bullish'], mix['bearish'], mix['unclear'])
        elif scenarios:
            movement += ' Scenario lines need at least 30 loaded candles.'
        return fig, rows, movement, status, health
