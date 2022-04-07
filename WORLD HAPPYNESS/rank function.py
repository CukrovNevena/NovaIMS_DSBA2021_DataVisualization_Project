import dash
from dash import dcc
from dash import html
import dash_daq as daq
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import plotly.graph_objs as go

######################################################Data##############################################################

df = pd.read_csv('2015-2021.csv')

Happiness_Indicators = ['Happiness Rank','Happiness Score']

Influencing_factors = ['Economy (GDP per Capita)','Social support','Health (Life Expectancy)','Freedom','Corruption Perceptions','Generosity']

######################################################Interactive Components############################################

country_options = [dict(label=country, value=country) for country in df['Country'].unique()]

Happiness_options = [dict(label=happiness, value=happiness) for happiness in Happiness_Indicators]

factors_options = [dict(label=factors, value=factors) for factors in Influencing_factors]

#######################################################################################################################


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

slider_year = daq.Slider(
        id='year_slider',
        handleLabel={"showCurrentValue": True,"label": "Year"},
        min=df['Year'].min(),
        max=df['Year'].max(),
        marks={str(i): '{}'.format(str(i)) for i in
               [2015,2016,2017,2018,2019,2020,2021]},
        value=df['Year'].max(),
        color='#4B9072',
        size=1400,
        step=1
    )


radio_lin_log = dcc.RadioItems(
        id='lin_log',
        options=[dict(label='Linear', value=0), dict(label='log', value=1)],
        value=0
    )

######################################################################################################################


app = dash.Dash(__name__)

server = app.server

app.layout = html.Div([

    html.Div([
        html.Img(src=app.get_asset_url('WHR_logo.png'),style={'width': '20%', 'position':'relative','margin': '20px'}),
    ], id='head'),

    html.Div([
        html.Div([
            html.Div([
                html.Div([
                    dropdown_scope,
                ], id='Map_Scope', style={'width': '20%'}),
                html.Div([
                    slider_year,
                ],style={'width': '80%'}, className='pretty_box'),
            ],style={'display': 'flex'}),
            html.Div([
            dcc.Graph(id='choropleth'),
            ]),
        ], id='Map', className='pretty_box')
    ], id='1st row', style={'high':'100%','width': '100%'}, className='row1back'),
])

#####################################################################################################################

@app.callback(
    [
        Output("choropleth", "figure"),
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

    #############################################Second Choropleth######################################################

    df_happiness_0 = df.loc[df['Year'] == year]

    data_choropleth = dict(type='choropleth',
                           locations=df_happiness_0['Country'],
                           # There are three ways to 'merge' your data with the data pre embedded in the map
                           locationmode='country names',
                           z=df_happiness_0['Happiness Score'],
                           text=df_happiness_0['Country'],
                           colorscale='Viridis',
                           colorbar=dict(title='Happiness Score Range'),

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
                                      oceancolor='azure',
                                      bgcolor='#f9f9f9',
                                      ),
                             width=1800,
                             height=1000,
                             dragmode=False,
                             margin=dict(l=0,
                                         r=0,
                                         b=0,
                                         t=200,
                                         pad=0),

                             title=dict(
                                 text=str(continent).capitalize()  + ' Happiness Score' + ' Choropleth Map on the year ' + str(
                                     year),
                                 x=.5 , # Title relative position according to the xaxis, range (0,1)
                                 font = dict(size=50)

                             ),
                             paper_bgcolor='rgba(0,0,0,0)',
                             plot_bgcolor='rgba(0,0,0,0)'
                             )


    return go.Figure(data=data_choropleth, layout=layout_choropleth)

  ##################################################################################

if __name__ == '__main__':
    app.run_server(debug=True)