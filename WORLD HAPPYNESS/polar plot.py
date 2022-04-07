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
rangeslider_year = dcc.RangeSlider(
        id='rangeslider',
        marks={str(i): '{}'.format(str(i)) for i in
                [2015, 2016, 2017, 2018, 2019, 2020, 2021]},
        min=2015,
        max=2021,
        value=[2021, 2021],
        step=1
    )

# The App itself

app = dash.Dash(__name__)

server = app.server

app.layout = html.Div([
    html.H1('Exercise 3 Data Visualization Example'),

    html.Br(),

    html.Label('Select the range of years'),
    rangeslider_year,
    html.Br(),

    html.Label('Select the first country and compare the impact factors :'),
    html.Br(),
    dropdown_country1,

    html.Label('Select the second country and compare the impact factors :'),
    html.Br(),
    dropdown_country2,

    dcc.Graph(
        id='polar-graph'
    )

])


@app.callback(
    Output('polar-graph', 'figure'),
    [Input("country1", "value"),
     Input("country2", "value"),
     Input("rangeslider","value")]
)




def polar_function(country1, country2,year):

    # scatterpolar
    df_year_selected=df[df['Year']==year].groupby(by=['Country'])
    df1_for_plot = pd.DataFrame(df[df['Country'] == country1][Influencing_factors].iloc[0])
    df1_for_plot.columns = ['score']
    df2_for_plot = pd.DataFrame(df[df['Country'] == country2][Influencing_factors].iloc[0])
    df2_for_plot.columns = ['score']
    list_scores = [df1_for_plot.index[i].capitalize() +' = ' + str(df1_for_plot['score'][i]) for i in range(len(df1_for_plot))]
    text_scores_1 = country1
    for i in list_scores:
        text_scores_1 += '<br>' + i

    list_scores = [df2_for_plot.index[i].capitalize() +' = ' + str(df2_for_plot['score'][i]) for i in range(len(df2_for_plot))]
    text_scores_2 = country2
    for i in list_scores:
        text_scores_2 += '<br>' + i

    fig = go.Figure(data=go.Scatterpolar(
        r=df1_for_plot['score'],
        theta=df1_for_plot.index,
        fill='toself',
        marker_color = 'rgb(45,0,198)',
        opacity =1,
        hoverinfo = "text" ,
        name = text_scores_1,
        text  = [df1_for_plot.index[i] +' = ' + str(df1_for_plot['score'][i]) for i in range(len(df1_for_plot))]
    ))
    fig.add_trace(go.Scatterpolar(
        r=df2_for_plot['score'],
        theta=df2_for_plot.index,
        fill='toself',
        marker_color = 'rgb(255,171,0)',
        hoverinfo = "text" ,
        name= text_scores_2,
        text  = [df2_for_plot.index[i] +' = ' + str(df2_for_plot['score'][i]) for i in range(len(df2_for_plot))]
        ))

    fig.update_layout(
        polar=dict(
            hole=0.1,
            bgcolor="white",
            radialaxis=dict(
                visible=True,
                type='linear',
                autotypenumbers='strict',
                autorange=False,
                range=[0,1.5],
                angle=90,
                showline=False,
                showticklabels=False, ticks='',
                gridcolor='black'),
                ),
        width = 550,
        height = 550,
        margin=dict(l=80, r=80, t=20, b=20),
        showlegend=False,
        template="plotly_dark",
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
        paper_bgcolor = 'rgba(0, 0, 0, 0)',
        font_color="black",
        font_size= 15
    )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
