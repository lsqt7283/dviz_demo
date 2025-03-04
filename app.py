import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Generate synthetic stock index data
def generate_stock_data():
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Date range from 2015 to 2024
    start_date = '2015-01-01'
    end_date = '2024-05-01'
    
    # Generate dates
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    # Create DataFrame with dates
    df = pd.DataFrame(index=dates)
    
    # Stock indices to simulate
    indices = ['S&P 500', 'NASDAQ', 'Dow Jones', 'FTSE 100', 'Nikkei 225', 'DAX']
    
    # Generate random walk data for each index
    for idx in indices:
        # Start value
        start_value = np.random.uniform(1000, 3000)
        
        # Daily returns with some correlation between indices
        daily_returns = np.random.normal(0.0003, 0.01, size=len(dates))
        
        # Add some trends and volatility changes
        time_factor = np.linspace(0, 1, len(dates))
        trend = 0.0005 * np.sin(time_factor * 10) # Add cyclical trend
        
        # Add COVID crash around March 2020
        covid_crash = np.zeros(len(dates))
        covid_period = (dates >= '2020-02-20') & (dates <= '2020-03-23')
        covid_crash[covid_period] = -0.03 * np.random.random(sum(covid_period))
        
        # Add 2022 downturn
        downturn_2022 = np.zeros(len(dates))
        downturn_period = (dates >= '2022-01-01') & (dates <= '2022-10-15')
        downturn_2022[downturn_period] = -0.01 * np.random.random(sum(downturn_period))
        
        # Combine effects
        adjusted_returns = daily_returns + trend + covid_crash + downturn_2022
        
        # Calculate cumulative returns
        cumulative_returns = (1 + adjusted_returns).cumprod()
        
        # Generate price series
        df[idx] = start_value * cumulative_returns
    
    # Save to CSV
    csv_path = 'stock_indices_data.csv'
    df.to_csv(csv_path)
    print(f"Data saved to {csv_path}")
    
    return df

# Check if data file exists, if not generate it
def get_data():
    csv_path = 'stock_indices_data.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        df = generate_stock_data()
    return df

# Initialize the Dash app
app = dash.Dash(__name__, title="Stock Indices Performance Visualization")
server = app.server  # Needed for deployment

# Define colors for the indices (visually distinguishable but not too bright/dark)
colors = {
    'S&P 500': '#1f77b4',  # Blue
    'NASDAQ': '#ff7f0e',    # Orange
    'Dow Jones': '#2ca02c', # Green
    'FTSE 100': '#d62728',  # Red
    'Nikkei 225': '#9467bd',# Purple
    'DAX': '#8c564b'        # Brown
}

# Get the data
df = get_data()

# Define the app layout
app.layout = html.Div([
    html.H1("Stock Indices Performance Comparison", 
           style={'textAlign': 'center', 'color': '#506784', 'fontFamily': 'Arial, sans-serif'}),
    
    html.Div([
        html.Div([
            html.Label("Select Date Range:", 
                     style={'fontWeight': 'bold', 'marginBottom': '10px'}),
            dcc.DatePickerRange(
                id='date-range',
                min_date_allowed=df.index.min(),
                max_date_allowed=df.index.max(),
                start_date=df.index.min(),
                end_date=df.index.max(),
                display_format='YYYY-MM-DD'
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            html.Label("Select Indices:", 
                     style={'fontWeight': 'bold', 'marginBottom': '10px'}),
            dcc.Checklist(
                id='indices-checklist',
                options=[{'label': idx, 'value': idx} for idx in df.columns],
                value=list(df.columns),  # Default: all selected
                inline=True
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
    ]),
    
    dcc.Graph(id='stock-performance-graph'),
    
    html.Div([
        html.H4("About This Visualization"),
        html.P("This dashboard shows the relative performance of major stock indices, rebased to 100 at the start date."),
        html.P("Use the date range selector to zoom in/out on specific time periods and see how different indices performed relative to each other."),
        html.P("The data shown is synthetic and generated for demonstration purposes only.")
    ], style={'margin': '20px', 'padding': '15px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px'})
], style={'margin': '20px', 'fontFamily': 'Arial, sans-serif'})

# Define callback to update the graph based on user inputs
@app.callback(
    Output('stock-performance-graph', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('indices-checklist', 'value')]
)
def update_graph(start_date, end_date, selected_indices):
    # Filter data by date range
    filtered_df = df.loc[start_date:end_date]
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each selected index
    for idx in selected_indices:
        if idx in filtered_df.columns:
            # Rebase to 100 at the start date
            rebased_values = 100 * filtered_df[idx] / filtered_df[idx].iloc[0]
            
            fig.add_trace(go.Scatter(
                x=filtered_df.index,
                y=rebased_values,
                mode='lines',
                name=idx,
                line=dict(color=colors[idx], width=2),
                hovertemplate='%{x}<br>%{y:.2f}'
            ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"Relative Performance (Rebased to 100 on {filtered_df.index[0].strftime('%Y-%m-%d')})",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Date",
        yaxis_title="Rebased Value",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified",
        template="plotly_white",
        height=600,
    )
    
    # Add a horizontal line at y=100 for reference
    fig.add_shape(
        type="line",
        x0=filtered_df.index[0],
        y0=100,
        x1=filtered_df.index[-1],
        y1=100,
        line=dict(color="gray", width=1, dash="dash"),
    )
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)