from os import listdir
from os.path import join
from pandas import concat, read_csv
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# Load forecast data
def load_forecasts():
    folder_path = 'C:\\Users\\samim\\OneDrive\\Documents\\Projects\\FinancialModelingTool\\forecast_data'
    all_forecasts = []
    for filename in listdir(folder_path):
        if filename.endswith(".csv"):
            df = read_csv(join(folder_path, filename))
            all_forecasts.append(df)
    return concat(all_forecasts, ignore_index=True)

# Initialize the Dash app
app = Dash(__name__)

# Load the forecasts
forecasts = load_forecasts()
companies = forecasts['Company'].unique()
models = forecasts['Model'].unique()
intervals = forecasts['Interval'].unique()

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Company Stock Forecasts"),
    dcc.Dropdown(
        id='company-dropdown',
        options=[{'label': company, 'value': company} for company in companies],
        value=companies[0]
    ),
    dcc.Dropdown(
        id='model-dropdown',
        options=[{'label': model, 'value': model} for model in models],
        value=models[0]
    ),
    dcc.Dropdown(
        id='interval-dropdown',
        options=[{'label': f'{interval} days', 'value': interval} for interval in intervals],
        value=intervals[0]
    ),
    dcc.Graph(id='forecast-graph')
])

# Update the graph based on dropdown selections
@app.callback(
    Output('forecast-graph', 'figure'),
    [Input('company-dropdown', 'value'),
     Input('model-dropdown', 'value'),
     Input('interval-dropdown', 'value')]
)
def update_graph(selected_company, selected_model, selected_interval):
    filtered_data = forecasts[(forecasts['Company'] == selected_company) &
                              (forecasts['Model'] == selected_model) &
                              (forecasts['Interval'] == selected_interval)]

    trace = go.Scatter(
        x=filtered_data['Day'],
        y=filtered_data['Prediction'],
        mode='lines',
        name=f'{selected_model} {selected_interval} days'
    )

    layout = go.Layout(
        title=f'{selected_company} Stock Price Prediction',
        xaxis={'title': 'Days'},
        yaxis={'title': 'Predicted Price'},
        showlegend=True
    )

    return {'data': [trace], 'layout': layout}

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
