from os import listdir
from os.path import join
from pandas import concat, read_csv
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from json import dump, load

# Load forecast data
def load_forecasts():
    folder_path = 'C:\\Users\\samim\\OneDrive\\Documents\\Projects\\FinancialModelingTool\\forecast_data'
    all_forecasts = []
    for filename in listdir(folder_path):
        if filename.endswith(".csv"):
            df = read_csv(join(folder_path, filename))
            all_forecasts.append(df)
    return concat(all_forecasts, ignore_index=True)

# Load average accuracy metrics
def load_average_accuracy_metrics():
    with open('average_accuracy_metrics.json', 'r') as f:
        return load(f)

# Initialize the Dash app
app = Dash(__name__)

# Load the forecasts
forecasts = load_forecasts()
average_accuracy_metrics = load_average_accuracy_metrics()
companies = forecasts['Company'].unique()
models = forecasts['Model'].unique()
intervals = forecasts['Interval'].unique()

# Map intervals to human-readable labels
interval_labels = {
    '1_days': '1 day',
    '3_days': '3 days',
    '5_days': '5 days',
    '7_days': '1 week',
    '14_days': '2 weeks',
    '30_days': '1 month',
    '91_days': '3 months',
    '183_days': '6 months',
    '365_days': '1 year'
}

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
        options=[{'label': f'{interval_labels[interval]}', 'value': interval} for interval in intervals],
        value=intervals[0]
    ),
    dcc.Graph(id='forecast-graph'),
    html.Div(id='accuracy-metrics')
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

    trace_pred = go.Scatter(
        x=filtered_data['Day'],
        y=filtered_data['Prediction'],
        mode='lines',
        name=f'{selected_model} {interval_labels[selected_interval]} Prediction'
    )

    trace_actual = go.Scatter(
        x=filtered_data['Day'],
        y=filtered_data['Actual'],
        mode='lines',
        name='Actual Price'
    )

    layout = go.Layout(
        title=f'{selected_company} Stock Price Prediction',
        xaxis={'title': 'Days'},
        yaxis={'title': 'Price'},
        showlegend=True
    )

    return {'data': [trace_pred, trace_actual], 'layout': layout}

# Update the accuracy metrics based on dropdown selections
@app.callback(
    Output('accuracy-metrics', 'children'),
    [Input('company-dropdown', 'value'),
     Input('model-dropdown', 'value'),
     Input('interval-dropdown', 'value')]
)
def update_accuracy_metrics(selected_company, selected_model, selected_interval):
    filtered_data = forecasts[(forecasts['Company'] == selected_company) &
                              (forecasts['Model'] == selected_model) &
                              (forecasts['Interval'] == selected_interval)]
    
    accuracy = filtered_data['Accuracy'].iloc[0]
    average_accuracy = average_accuracy_metrics[selected_model][selected_interval]
    average_accuracy_report = f"Accuracy: {average_accuracy:.2f}%"
    accuracy_metrics = html.Div([
        html.P(f"Prediction Accuracy: {accuracy:.2f}%"),
        html.P(f"Total Average {average_accuracy_report}")
    ])

    return accuracy_metrics

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
