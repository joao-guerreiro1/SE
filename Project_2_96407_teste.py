
#Main libraries
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
from dash import Dash
import pandas as pd
import plotly.express as px
import pickle
from sklearn import  metrics
import numpy as np
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor


#import os
from dash.exceptions import PreventUpdate

#Define CSS style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


white_text_style = {'color': 'white'}

# Read the data
merged = pd.read_csv('merged_data.csv', index_col= 0)
merged.index = pd.to_datetime(merged.index)


# Create lag features for previous hour consumption
merged = merged.assign(Power_1=merged['Power (kW)'].shift(1),
                         Power_2=merged['Power (kW)'].shift(2))

merged=merged.dropna()

# Extract month, hour, week day, and weekend features
merged = merged.assign(Month=merged.index.month,
                         Hour=merged.index.hour,
                         Week_Day=merged.index.weekday + 1,
                         Weekend=merged.index.weekday.isin([5, 6]).astype(int))
columns = merged.columns.tolist()
begin = merged.index.min()
end = merged.index.max()


#Split the data into 2017-2018 and 2019
test_cutoff_date = '2019-01-01'
data = merged.loc[merged.index < test_cutoff_date] #2017 and 2018
data_2019 = merged.loc[merged.index >= test_cutoff_date] #2019

data = data.dropna()

data_fs = data.drop("Power (kW)", axis=1).copy()

#2019
real = pd.read_csv('data_real.csv')
real['Date'] = pd.to_datetime(real['Date'])
real = real.assign(Power_1=real['Power (kW)'].shift(1),
                   Power_2=real['Power (kW)'].shift(2))
real=real.dropna()

y=real['Power (kW)'].values

meteo_2019 = data_2019.drop('Power (kW)', axis=1)
dates_to_drop = meteo_2019[meteo_2019.index.isin(real.index)].index
df_meteo_2019 = meteo_2019.drop(dates_to_drop)



print(real.head())

X = None
Y = None

X_train = None
X_test = None
y_train = None
y_test = None

X_2019 = None


fig2 = px.line(real, x='Date', y='Power (kW)')


#Aux functions
def generate_table(dataframe, max_rows=10):
    # Apply some CSS styles to the table
    table_style = {
        'borderCollapse': 'collapse',
        'borderSpacing': '0',
        'width': '100%',
        'border': '1px solid #ddd',
        'fontFamily': 'Arial, sans-serif',
        'fontSize': '14px'
    }
    
    th_style = {
        'border': '1px solid #ddd',
        'padding': '8px',
        'textAlign': 'left',
        'backgroundColor': '#f2f2f2',
        'fontWeight': 'bold',
        'color': '#333'
    }
    
    td_style = {
        'border': '1px solid #ddd',
        'padding': '8px',
        'textAlign': 'left'
    }
    
    # Generate table header
    header_row = html.Tr([
        html.Th('Index', style=th_style),
        *[html.Th(col, style=th_style) for col in dataframe.columns]
    ])
    
    # Generate table body
    body_rows = [
        html.Tr([
            html.Td(dataframe.index[i], style=td_style),
            *[html.Td(dataframe.iloc[i][col], style=td_style) for col in dataframe.columns]
        ]) for i in range(min(len(dataframe), max_rows))
    ]
    
    # Assemble table
    table = html.Table(
        # Apply the table style
        style=table_style,
        children=[
            html.Thead(header_row),  # Add the table header
            html.Tbody(body_rows)   # Add the table body
        ]
    )
    
    return table



def generate_graph(df, columns, start_date, end_date):
    # Filter DataFrame
    filtered_df = df.loc[start_date:end_date, columns]
    
    # Create traces for each column
    traces = []
    for column in filtered_df.columns:
        traces.append(go.Scatter(x=filtered_df.index, y=filtered_df[column], name=column))
    
    # Define layout
    layout = go.Layout(title=', '.join(columns), xaxis_title='Date', yaxis=dict(title='Value'))
    
    # Create figure
    fig = go.Figure(data=traces, layout=layout)
    
    return fig




app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server


app.layout = html.Div(style={'backgroundColor': 'white'}, children=[
    html.H1('Civil Building Energy Forecast tool (kWh)', style={'margin': '20px'}),
    html.Div(id='merged', children=merged.to_json(orient='split'), style={'display': 'none'}),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw Data', children=[
            html.Div([
                html.H2("Raw Data"),
                html.P('Check the original data. It is possible to adjust time interval:', style={'margin-bottom': '10px'}),
                dcc.Dropdown(
                    id='column-dropdown',
                    options=[{'label': i, 'value': i} for i in merged.columns],
                    value=[merged.columns[0]],
                    multi=True
                ),
                dcc.DatePickerRange(
                    id='date-picker',
                    min_date_allowed=merged.index.min(),
                    max_date_allowed=merged.index.max(),
                    start_date=merged.index.min(),
                    end_date=merged.index.max()
                ),
                dcc.Graph(id='graph'),
            ], style={'padding': '20px'})
        ]),
        dcc.Tab(label='Exploratory Data Analysis', value='tab-2', children=[
            html.Div([
                html.H2("Exploratory Data Analysis", style={}),
                html.P('Check the variables correlation:', style={'margin-bottom': '10px'}),
                dcc.Dropdown(
                    id='feature1',
                    options=[{'label': col, 'value': col} for col in merged.columns],
                    value=data.columns[0]
                ),
                dcc.Dropdown(
                    id='feature2',
                    options=[{'label': col, 'value': col} for col in merged.columns],
                    value=data.columns[1]
                ),
                dcc.Graph(id='scatter-plot'),
                dcc.Dropdown(
                    id='feature-boxplot',
                    options=[{'label': col, 'value': col} for col in merged.columns],
                    value=merged.columns[1]
                ),
                dcc.Graph(id='box-plot')
            ], style={'padding': '20px'})
        ]),
        dcc.Tab(label='Feature Selection', value='tab-3', children=[
            html.Div([
                html.H2("Feature Selection", style={}),
                html.P('Choose the variables and do not forget locking them!', style={'margin-bottom': '10px'}),
                html.P('Hint: The best features are: Power-1, Power-2, Solar Radiation and Hour!', style={'margin-bottom': '10px'}),
                dcc.Dropdown(
                    id='feature-dropdown',
                    options=[{'label': col, 'value': col} for col in data_fs.columns],
                    value=[data_fs.columns[0]],
                    multi=True
                ),
                html.Div(id='feature-table-div'),
                html.Button('Lock Variables', id='split-button'),
                html.Div(id='split-values'),
                html.Div([
                    html.H6(""),
                    html.Pre(id="x-values", style=white_text_style)
                ]),
                html.Div([
                    html.H6(""),
                    html.Pre(id="y-values", style=white_text_style)
                ]),
                html.Div([
                    html.H6(""),
                    html.Pre(id="x-2019-values", style=white_text_style)
                ]),
            ], style={'padding': '20px'})
        ]),
        dcc.Tab(label='Regression Models', value='tab-4', children=[
            html.Div([
                html.H2("Regression Models", style={}),
                html.P('Choose from the different models and test it:', style={'margin-bottom': '10px'}),
                html.P('Hint: Usually the Random Forests method has better results!', style={'margin-bottom': '10px'}),
                dcc.Dropdown(
                    id='model-dropdown',
                    options=[
                        {'label': 'Linear Regression', 'value': 'linear'},
                        {'label': 'Random Forests', 'value': 'random_forests'},
                        {'label': 'Bootstrapping', 'value': 'bootstrapping'},
                        {'label': 'Decision Tree Regressor', 'value': 'decision_trees'},
                        {'label': 'Gradient Boosting', 'value': 'gradient_boosting'},
                    ],
                    value='linear'
                ),
                html.Button('Train Model', id='train-model-button'),
            ], style={'padding': '20px'}), 
            html.Div([
                html.H2(""),
                dcc.Loading(
                    id="loading-1",
                    children=[html.Div([dcc.Graph(id="lr-graph")])]
                )
            ]),
        ]),
        dcc.Tab(label='Model Deployment and Visualization', value='tab-6', children=[
            html.Div([
                html.H2('Model Deployment', style={}),
                html.P('Run the model:', style={'margin-bottom': '10px'}),
                dcc.Graph(id='time-series-plot', figure=fig2),
                html.Button('Run Model', id='button_model'),
                html.Div(id='model-performance-table')
            ], style={'padding': '20px'})
        ]),
        dcc.Tab(label='Prediction Errors', value='tab-5', children=[
            html.Div([
                html.H2("Model Performance", style={}),
                html.Div(id='model-performance-content')
            ], style={'padding': '20px'})
        ]),
    ]),
    html.Div(id='tabs-content')
])


# Define the callbacks
@app.callback(Output('graph', 'figure'),
              Input('column-dropdown', 'value'),
              Input('date-picker', 'start_date'),
              Input('date-picker', 'end_date')
)
              
              
def update_figure(columns, start_date, end_date):
    """
    Update the figure based on selected columns and date range.

    Parameters:
        columns (list): List of column names selected by the user.
        start_date (str): Start date selected by the user.
        end_date (str): End date selected by the user.

    Returns:
        dict: A dictionary representing the updated figure.
    """
    # Filter the DataFrame based on the selected date range and columns
    filtered_df = merged.loc[start_date:end_date, columns]
    
    # Define the y-axis configurations for each column
    y_axis_config = [{'overlaying': 'y', 'side': 'right', 'position': 1 - i * 0.1} for i, _ in enumerate(columns)]
    
    # Define the data and layout of the figure
    data = [{'x': filtered_df.index, 'y': filtered_df[column], 'type': 'line', 'name': column} for column in filtered_df.columns]
    layout = {'title': {'text': ', '.join(columns)}, 'xaxis': {'title': 'Date'}}
    
    # Update the layout to include the y-axis configurations
    for i, config in enumerate(y_axis_config):
        layout[f'yaxis{i+1}'] = config
    
    # Create the figure with the data and layout
    fig = {'data': data, 'layout': layout}
    
    return fig


import plotly.express as px

@app.callback(Output('scatter-plot', 'figure'),
              Input('feature1', 'value'),
              Input('feature2', 'value'))
def update_scatter_plot(feature1, feature2):
    fig = px.scatter(merged, x=feature1, y=feature2, title=f'{feature1} vs {feature2}')
    fig.update_xaxes(title_text=feature1)
    fig.update_yaxes(title_text=feature2)
    return fig



@app.callback(
    Output('box-plot', 'figure'),
    Input('feature-boxplot', 'value')
)
def update_box_plot(feature_boxplot):
    fig = px.box(merged, y=feature_boxplot, title=f"Box Plot for {feature_boxplot}")
    return fig


@app.callback(
    Output('feature-table-div', 'children'),
    Input('feature-dropdown', 'value')
)
def update_feature_table(selected_features):
    if selected_features:
        global model
        model = data_fs[selected_features]
        table = generate_table(model)
        return table
    else:
        return html.Div()
    
    
    
@app.callback(
    [
        Output('x-values', 'children'),
        Output('y-values', 'children'),
        Output('x-2019-values', 'children')
    ],
    [Input('feature-dropdown', 'value')]
)

def update_x_y(selected_features):
    global X, Y, X_2019
    if selected_features:
        X = model[selected_features].values
        Y = data['Power (kW)'].values
        X_2019 = df_meteo_2019[selected_features].values
        return str(X), str(Y), str(X_2019)
    else:
        return "", "", ""


@app.callback(
    Output('split-values', 'children'),
    Input('split-button', 'n_clicks')
)

def generate_train_test_split(n_clicks):
    global X_train, X_test, y_train, y_test
    if n_clicks:
        X_train, X_test, y_train, y_test = train_test_split(X, Y)
        return 'Done!'
    else:
        return ""


prediction = []
prediction_2019 = []



@app.callback(
    Output('lr-graph', 'figure'),
    Input('train-model-button', 'n_clicks'),
    State('model-dropdown', 'value')
)


def train_and_predict(n_clicks, model_type):
    from sklearn.impute import SimpleImputer
    from sklearn import linear_model


    global prediction, prediction_2019
    
    if n_clicks is None:
        return dash.no_update
    
    # Create an imputer object
    imputer = SimpleImputer(strategy='mean')
    
    # Impute missing values in the training, test, and 2019 data
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    X_2019_imputed = imputer.transform(X_2019)
    
    # Define a dictionary containing model types and their corresponding models
    models = {
        'linear': linear_model.LinearRegression(),
        'random_forests': RandomForestRegressor(bootstrap=True, min_samples_leaf=3,
                                                 n_estimators=200, min_samples_split=15,
                                                 max_features='sqrt', max_depth=20,
                                                 max_leaf_nodes=None),
        'bootstrapping': BaggingRegressor(),
        'decision_trees': DecisionTreeRegressor(),
        'gradient_boosting': GradientBoostingRegressor()
    }
    
    # Train the selected model using the imputed data
    model = models[model_type]
    model.fit(X_train_imputed, y_train)
    
    # Save the trained model
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)
    
    # Make predictions
    pred = model.predict(X_test_imputed)
    prediction.append(pred)
    
    prediction_2019 = model.predict(X_2019_imputed)
    
    # Generate scatter plot of predicted vs actual values
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=pred, mode='markers'))
    fig.update_layout(title=f'{model_type.capitalize()} Predictions')
    
    return fig

@app.callback(
    Output('time-series-plot', 'figure'),
    #Output('model-performance-table', 'children'),
    Input('button_model', 'n_clicks')
)
def run_model(n_clicks):
 

    if n_clicks is None:
        raise PreventUpdate
    else:
        if 'Date' in real.columns:
            real['Date'] = pd.to_datetime(real['Date'])
            real.set_index('Date', inplace=True)

        
        #df_real.set_index('Date', inplace=True)
        fig = go.Figure(layout=go.Layout(title='Real vs Predicted Power Consumption'))
        fig.add_scatter(x=real.index, y=real['Power (kW)'], name='Real Power (kW)')
        fig.add_scatter(x=real.index, y=prediction_2019, name='Predicted Power (kW)')
        
       
        
        
        # Calculate model performance metrics
        MAE = metrics.mean_absolute_error(real['Power (kW)'], prediction_2019)
        MBE = np.mean(real['Power (kW)'] - prediction_2019)
        MSE = metrics.mean_squared_error(real['Power (kW)'], prediction_2019)
        RMSE = np.sqrt(MSE)
        cvrmse = RMSE / np.mean(real['Power (kW)'])
        nmbe = MBE / np.mean(real['Power (kW)'])

        # Format the metrics as percentages with two decimal places
        cvRMSE_perc = "{:.2f}%".format(cvrmse * 100)
        NMBE_perc = "{:.2f}%".format(nmbe * 100)
        
        # Create the table with the metrics
        d = {'Model':['Model Selected'],'MAE': [MAE],'MBE': [MBE], 'MSE': [MSE], 'RMSE': [RMSE],'cvMSE': [cvRMSE_perc],'NMBE': [NMBE_perc]}
        df_metrics = pd.DataFrame(data=d)
        table = generate_table(df_metrics)
        
    return fig


@app.callback(
    Output('model-performance-content', 'children'),
    Input('button_model', 'n_clicks')
)
def update_model_performance(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    else:
        # Calculate model performance metrics
        MAE = metrics.mean_absolute_error(real['Power (kW)'], prediction_2019)
        MBE = np.mean(real['Power (kW)'] - prediction_2019)
        MSE = metrics.mean_squared_error(real['Power (kW)'], prediction_2019)
        RMSE = np.sqrt(MSE)
        cvrmse = RMSE / np.mean(real['Power (kW)'])
        nmbe = MBE / np.mean(real['Power (kW)'])

        # Format the metrics as percentages with two decimal places
        cvRMSE_perc = "{:.2f}%".format(cvrmse * 100)
        NMBE_perc = "{:.2f}%".format(nmbe * 100)
        
        # Create the table with the metrics
        d = {'Model':['Model Selected'],'MAE': [MAE],'MBE': [MBE], 'MSE': [MSE], 'RMSE': [RMSE],'cvMSE': [cvRMSE_perc],'NMBE': [NMBE_perc]}
        df_metrics = pd.DataFrame(data=d)
        table = generate_table(df_metrics)
        
    return table





if __name__ == '__main__':
    app.run_server()
