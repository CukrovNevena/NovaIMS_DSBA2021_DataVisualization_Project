import dash
from dash import dcc
from dash import html
import dash_daq as daq
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import statsmodels.api as sm
import plotly.figure_factory as ff

######################################################Data##############################################################

df = pd.read_csv('2015-2021.csv')

Happiness_Indicators = ['Happiness Rank','Happiness Score']

Influencing_factors = ['Economy (GDP per Capita)','Social support','Health (Life Expectancy)','Freedom','Corruption Perceptions','Generosity']

######################################################Interactive Components############################################

country_options = [dict(label=country, value=country) for country in df['Country'].unique()]

Happiness_options = [dict(label=happiness, value=happiness) for happiness in Happiness_Indicators]

factors_options = [dict(label=factors, value=factors) for factors in Influencing_factors]

dropdown_country = dcc.Dropdown(
        id='country_drop',
        options=country_options,
        value=['Portugal','Germany','China','Finland','Slovenia'],
        multi=True
    )


dropdown_scope = dcc.Dropdown(
        id='scopes_option',
        clearable=False,
        searchable=False,
        options=[{'label': 'World', 'value': 'world'},
                {'label': 'Europe', 'value': 'europe'},
                {'label': 'Asia', 'value': 'asia'},
                {'label': 'Africa', 'value': 'africa'},
                {'label': 'North america', 'value': 'north america'},
                {'label': 'South america', 'value': 'south america'}],
        value='world',
    )

dropdown_country1 = dcc.Dropdown(
        id='country1',
        options=country_options,
        value='Germany'
    )

dropdown_country2 = dcc.Dropdown(
        id='country2',
        options=country_options,
        value='Portugal'
    )

dropdown_factors = dcc.Dropdown(
        id='dropdown_factor',
        options=factors_options,
        value='Economy (GDP per Capita)'
    )

slider_year = dcc.Slider(
        id='year_slider',
        min=df['Year'].min(),
        max=df['Year'].max(),
        marks={str(i): '{}'.format(str(i)) for i in
               [2015,2016,2017,2018,2019,2020,2021]},
        value=df['Year'].min(),
        step=1
    )

radio_lin_log = dcc.RadioItems(
        id='lin_log',
        options=[dict(label='Linear', value=0), dict(label='log', value=1)],
        value=0
    )


##################################################APP###################################################################

app = dash.Dash(__name__)

server = app.server

app.layout = html.Div([

    html.Div([
        html.Img(src=app.get_asset_url('whr.png'),
                 style={'width': '15%', 'position':'relative','margin': '3%'}),
    ], id='head'),

    html.Div([
        html.H1(id='choropleth_title',style={'color':'#000000'}),
        html.Div([
            html.Div([
                dropdown_scope,
            ], id='Map_Scope', style={'width': '40%'}, className='pretty_box'),
            html.Div([
                slider_year,
            ], style={'width': '60%'}, className='pretty_box'),
        ], style={'display': 'flex','width': '60%','padding-right':'20%','padding-left':'20%'}),
        html.Div([
            dcc.Graph(id='choropleth'),
        ], id='Map', className='map_box'),
    ], id='1st row', className='row1back'),

    html.Div([
        html.H2(id='Year_selected_1'),
        html.Div([
            html.Div([
                html.Div([
                    html.Label('Choose one factor to compare',style={'color':'#ffffff'}),
                    dropdown_factors,
                ],className='pretty_box'),
            ], style={'width': '25%','padding-right':'35.5%','padding-left':'35.5%','padding-bottom':'5%'}),
            html.Div([
                dcc.Graph(id='box_graph')
            ],style={'width': '50%','padding-right':'25%','padding-left':'25%'}),
        ]),
        html.Div([
            html.Div([
                dcc.Graph(id='top10rank')
            ],style={'width': '30%','padding-left':'25%'}),
            html.Div([
                dcc.Graph(id='cor_graph')
            ],style={'width': '30%','padding-right':'25%'})
        ],style={'display': 'flex'},)
    ], id='2nd row',  className='row2back'),

    html.Div([
        html.H3(id='Year_selected_2'),
        html.Div([
            html.Div([
                html.Div([
                    html.Label('Country Choice'),
                    dropdown_country,
                ], id='Iteraction1', style={'width': '50%','padding-right':'25%','padding-left':'25%','padding-bottom':'2%'}),
                html.Div([
                    html.Label('Linear Log'),
                    radio_lin_log,
                ], id='Iteraction2', style={'width': '50%','padding-right':'25%','padding-left':'25%'}),
            ], id='Iteraction'),
        ]),
        html.Div([
            html.Div([
                dcc.Graph(id='line_graph'),
            ], id='Graph1', style={'width': '50%','padding-right':'25%','padding-left':'25%',}),
            html.Div([
                dcc.Graph(id='bar_graph'),
            ], id='Graph2', style={'width': '50%','padding-right':'25%','padding-left':'25%',})
        ]),
    ], id='3rd row', className='row3back'),

    html.Div([
        html.H4(id='Year_selected_3'),
        html.Div([
            html.Div([
                html.Div([
                    html.Label('Select the first country and compare the impact factors',style={'color': '#ffffff'}),
                    html.Br(),
                    dropdown_country1,
                    html.Br(),
                ]),
                html.Div([
                    html.Label(id='Happiness_1'),
                    html.Br(),
                ], style={'color': '#ffffff'},className='ranksbox'),
                html.Div([
                    html.Label(id='Happiness_2'),
                    html.Br(),
                ], style={'color': '#ffffff'},className='ranksbox'),
            ], id='c1', style={'width': '25%','padding-right':'1%','padding-left':'25%'}),
            html.Div([
                html.Div([
                    html.Label('Select the second country and compare the impact factors',style={'color': '#ffffff'},),
                    html.Br(),
                    dropdown_country2,
                    html.Br(),
                ]),
                html.Div([
                    html.Label(id='Happiness_3'),
                    html.Br(),
                ], style={'color': '#ffffff'},className='ranksbox'),
                html.Div([
                    html.Label(id='Happiness_4'),
                    html.Br(),
                ],style={'color': '#ffffff'},className='ranksbox')
            ], id='c2', style={'width': '25%','padding-right':'25%','padding-left':'1%'}),
        ],style={'display': 'flex','padding-bottom':'5%'}),
        html.Div([
            dcc.Graph(id='polar-graph')
        ], id='polar', style={'width': '50%','padding-right':'25%','padding-left':'30%',}),
    ], id='4th row', className='row4back'),
    html.Div([
        html.Div([
            html.H5("Authors",style={'color':'#000000'}),
            dcc.Markdown("""\
              Tongjiuzhou Liu (m20211012@novaims.unl.pt)  
              Nina Urbancic (m20211314@novaims.unl.pt)  
              Amelie Florentine Langenstein (m20210637@novaims.unl.pt)  
              Laura Isabella Cuna (m20211312@novaims.unl.pt) 
            """,style={"text-align": "center", "font-size": "15pt"}),
        ]),
        html.Div([
            html.H6("Sources",style={'color':'#000000'}),
            dcc.Markdown("""\
            - Dash Enterprise App Gallery: https://dash.gallery/Portal/
            - The World Happiness Report: https://worldhappiness.report/
            - World Happiness Report up to 2022: https://www.kaggle.com/datasets/mathurinache/world-happiness-report
            """, style={"text-align": "center","font-size": "15pt"}),

        ]),
    ], className='lastrow')
])



######################################################Callbacks#########################################################


@app.callback(
    [
        Output("line_graph", "figure"),
        Output("bar_graph", "figure"),
        Output("choropleth", "figure"),
        Output('choropleth_title',"children")
    ],
    [
        Input("year_slider", "value"),
        Input("country_drop", "value"),
        Input("lin_log", "value"),
        Input('scopes_option', 'value')
    ]
)
def plots(year, countries, scale, continent):
    ############################################First Bar Plot##########################################################
    data_line = []
    data_bar = []
    for country in countries:
        df_line= df.loc[(df['Country'] == country)]

        x_line = df_line['Year']
        y_line = df_line['Happiness Rank']

        data_line.append(dict(type='scatter', x=x_line, y=y_line, name=country))

        df_bar = df.loc[(df['Country'] == country)]

        x_bar = df_bar['Year']
        y_bar = df_bar['Happiness Score']

        data_bar.append(dict(type='bar', x=x_bar, y=y_bar, name=country))

    layout_line= dict(title=dict(text='Happiness Rank from 2015 until 2021'),
                      yaxis=dict(title='Indicators', type=['linear', 'log'][scale], autorange='reversed'),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      legend=dict(font=dict(color='black',size=20))
                      )

    layout_bar = dict(title=dict(text='Happiness Score from 2015 until 2021'),
                      yaxis=dict(title='Indicators', type=['linear', 'log'][scale]),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', titlefont=dict(color='white'),
                      legend=dict(font=dict(color='black',size=20))
                      )

    #############################################Second Choropleth######################################################

    df_happiness_0 = df.loc[df['Year'] == year]

    data_choropleth = dict(type='choropleth',
                           locations=df_happiness_0['Country'],
                           # There are three ways to 'merge' your data with the data pre embedded in the map
                           locationmode='country names',
                           z=df_happiness_0['Happiness Score'],
                           text=df_happiness_0['Country'],
                           colorscale='pinkyl',
                           colorbar=dict(title='Happiness Score Range',len=0.75,orientation='h',\
                                         y=-0.1,tickfont=dict(color='black'),titlefont=dict(size=20,color='black')),
                           hovertemplate='Country: %{text} <br>' + 'Happiness Score' + ': %{z}',
                           name=''
                           )

    layout_choropleth = dict(geo=dict(scope=continent,  # default
                                      projection=dict(type='equirectangular'
                                                      ),
                                      # showland=True,   # default = True
                                      landcolor='white',
                                      lakecolor='white',
                                      showocean=True,  # default = False
                                      oceancolor='white',
                                      bgcolor='#f9f9f9',
                                      ),
                             width=1080,
                             height=800,
                             dragmode=False,
                             margin=dict(l=0,
                                         r=0,
                                         b=100,
                                         t=0,
                                         pad=0),
                             paper_bgcolor='rgba(0,0,0,0)',
                             plot_bgcolor='rgba(0,0,0,0)'
                             )
#######################################################################################################################

    return go.Figure(data=data_line, layout=layout_line), \
           go.Figure(data=data_bar, layout=layout_bar), \
           go.Figure(data=data_choropleth, layout=layout_choropleth), \
           str(continent).capitalize() + ' Happiness Score' + ' Choropleth Map on the year ' + str(year),

  #############################################polar plot############################################################
@app.callback(
    Output('polar-graph', 'figure'),
    [Input("country1", "value"),
     Input("country2", "value")]
)

def polar_function(country1, country2):

    # scatterpolar
    df1_for_plot = pd.DataFrame(df[df['Country'] == country1][Influencing_factors].iloc[0])
    df1_for_plot.columns = ['score']
    df2_for_plot = pd.DataFrame(df[df['Country'] == country2][Influencing_factors].iloc[0])
    df2_for_plot.columns = ['score']
    list_scores = [df1_for_plot.index[i].capitalize() +' = ' \
                   + str(df1_for_plot['score'][i]) for i in range(len(df1_for_plot))]
    text_scores_1 = country1
    for i in list_scores:
        text_scores_1 += '<br>' + i

    list_scores = [df2_for_plot.index[i].capitalize() +' = ' \
                   + str(df2_for_plot['score'][i]) for i in range(len(df2_for_plot))]
    text_scores_2 = country2
    for i in list_scores:
        text_scores_2 += '<br>' + i

    fig = go.Figure(data=go.Scatterpolar(
        r=df1_for_plot['score'],
        theta=df1_for_plot.index,
        fill='toself',
        marker_color = '#ec647d',
        opacity =1,
        hoverinfo = "text" ,
        name = text_scores_1,
        text  = [df1_for_plot.index[i] +' = ' + str(df1_for_plot['score'][i]) for i in range(len(df1_for_plot))]
    ))
    fig.add_trace(go.Scatterpolar(
        r=df2_for_plot['score'],
        theta=df2_for_plot.index,
        fill='toself',
        marker_color = '#fbd35f',
        hoverinfo = "text" ,
        name= text_scores_2,
        text  = [df2_for_plot.index[i] +' = ' + str(df2_for_plot['score'][i]) for i in range(len(df2_for_plot))]
        ))

    fig.update_layout(
        polar=dict(
            hole=0.05,
            bgcolor='#ffffff',
            radialaxis=dict(
                visible=True,
                type='linear',
                autotypenumbers='strict',
                autorange=False,
                range=[0,1.5],
                angle=90,
                showline=False,
                showticklabels=False, ticks='',
                gridcolor='white'),
                ),
        width = 800,
        height = 500,
        margin=dict(l=0, r=0, t=20, b=20),
        showlegend=False,
        template="plotly_dark",
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
        paper_bgcolor = 'rgba(0, 0, 0, 0)',
        font_color="white",
        font_size= 15
    )

    return fig
############################################top 10 rank bar plot#################################################################

@app.callback(
    [
        Output("top10rank", "figure"),
    ],
    [
        Input("year_slider", "value")
    ]
)

def top10rankplot(year):
    df_rank = df[df['Year'] == year].sort_values('Happiness Rank').head(10)[['Country', 'Happiness Score']]
    fig_rank = go.Figure(
        data=[
            go.Bar(y=df_rank['Country'], x=df_rank['Happiness Score'], orientation='h', text=df_rank['Happiness Score'],
                   textposition='auto',marker_color='lightcoral')],
        layout_title_text="TOP 10 Countries Happniess Rank"
    )
    fig_rank.update_layout(hovermode='closest',titlefont=dict(color='white'),
                           xaxis=dict(color='white'),
                           yaxis=dict(color='white',autorange="reversed"),
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
    return [fig_rank]

#########################################Box scatter plot################################################################

@app.callback(

    Output('box_graph', 'figure'),
    [
    Input('year_slider', 'value'),
    Input('dropdown_factor', 'value'),

    ]
)

def box_graph_function(year,factor):

    df_box=df[df['Year']==year]

    x_box = df_box['Happiness Score']
    y_box = df_box[factor]

    fig_box = px.scatter(df_box, x=x_box, y=y_box,hover_name="Country",
                         log_x=False, marginal_x='box', marginal_y='box', template="simple_white",
                         color_discrete_sequence=["#ec647d", "#9c179e"])
    regline = sm.OLS(y_box, sm.add_constant(x_box)).fit().fittedvalues
    fig_box.add_traces(go.Scatter(x=x_box, y=regline,
                                  mode='lines',
                                  marker_color='#ffffff',
                                  name='OLS Trendline')
                       )
    fig_box.update_layout(legend=dict(orientation="h", xanchor='center', x=0.5, yanchor='top', y=-0.2))

    fig_box.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest',paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)',titlefont=dict(color='white'),
                          xaxis=dict(color='white'),yaxis=dict(color='white'),)
    return fig_box
#####################################cor heatmap##############################################################
@app.callback(

    Output('cor_graph', 'figure'),
    [
    Input('year_slider', 'value'),
    ]
)

def cor_graph_function(year):

    corr_list = ['Happiness Score','Economy (GDP per Capita)','Social support',\
                 'Health (Life Expectancy)','Freedom','Corruption Perceptions','Generosity']

    df_corr_r = df[df['Year'] == year][corr_list]
    df_corr_round = df_corr_r.corr()[['Happiness Score']].T[Influencing_factors].T.round(2)

    fig_cor = ff.create_annotated_heatmap(
        z=df_corr_round.to_numpy(),
        x=df_corr_round.columns.tolist(),
        y=df_corr_round.index.tolist(),
        zmax=1, zmin=-1,
        showscale=True,
        hoverongaps=True,
        ygap=3,
        colorscale='pinkyl'
    )

    fig_cor.update_layout(yaxis=dict(showgrid=False), xaxis=dict(showgrid=False),

                          legend=dict(
                              orientation="h",
                              yanchor="bottom",
                              y=1.02,
                              xanchor="right",
                              x=1
                          ))
    fig_cor.update_layout(xaxis_tickangle=0)
    fig_cor.update_layout(title_text="Correlation Heatmap between Happiness Score and factors",
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)', titlefont=dict(color='white'),
                          xaxis=dict(color='white'),yaxis=dict(color='white'))

    return fig_cor




###########################################happiness rank and score tag################################################
@app.callback(
    [
        Output("Happiness_1", "children"),
        Output("Happiness_2", "children")
    ],
    [
        Input("country1", "value"),
        Input("year_slider", "value"),
    ]
)
def indicator1(country, year):
    df_loc = df.loc[df['Country']==country].groupby('Year').sum().reset_index()

    value_1 = round(df_loc.loc[df_loc['Year'] == year][Happiness_Indicators[0]].values[0], 2)
    value_2 = round(df_loc.loc[df_loc['Year'] == year][Happiness_Indicators[1]].values[0], 2)


    return str(year)+' '+str(Happiness_Indicators[0])+' of '+str(country)+': ' + str(value_1), \
           str(year)+' '+str(Happiness_Indicators[1])+' of '+str(country)+': ' + str(value_2),

@app.callback(
    [
        Output("Happiness_3", "children"),
        Output("Happiness_4", "children"),
        Output("Year_selected_1", "children"),
        Output("Year_selected_2", "children"),
        Output("Year_selected_3", "children"),
    ],
    [
        Input("country2", "value"),
        Input("year_slider", "value"),
    ]
)
def indicator2(country, year):
    df_loc = df.loc[df['Country']==country].groupby('Year').sum().reset_index()

    value_1 = round(df_loc.loc[df_loc['Year'] == year][Happiness_Indicators[0]].values[0], 2)
    value_2 = round(df_loc.loc[df_loc['Year'] == year][Happiness_Indicators[1]].values[0], 2)

    return str(year) + ' ' + str(Happiness_Indicators[0]) + ' of ' + str(country) + ': ' + str(value_1), \
           str(year) + ' ' + str(Happiness_Indicators[1]) + ' of ' + str(country) + ': ' + str(value_2), \
           str(year) + ' Happiness Score and the related factors: Box plots with OLS, Top 10 ranking, Correlation Heatmap ',\
           ' Happiness Score and Rank Comparison for Multiple countries ', \
           'Two countries All-round competition in year ' + str(year), \



if __name__ == '__main__':
    app.run_server(debug=True)