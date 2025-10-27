from flask import Flask, render_template, request, redirect, url_for
from dash import Dash, html, dcc, Input, Output, callback
import pandas as pd
import plotly.express as px
import pickle

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/prediksi')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        lifeExp = float(request.form['lifeExp'])
        gdpPercap = float(request.form['gdpPercap'])
        pop = float(request.form['pop'])
        year = float(request.form['year'])

        data = pd.DataFrame({
            'lifeExp': [lifeExp],
            'gdpPercap': [gdpPercap],
            'pop': [pop],
            'year': [year]
        })

        scaled = scaler.transform(data)
        prediction = model.predict(scaled)[0]

        return render_template('index.html', prediction=prediction,
                               lifeExp=lifeExp, gdpPercap=gdpPercap,
                               pop=pop, year=year)
    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/')
def dashboard():
    return redirect('/dash/')

dash_app = Dash(__name__, server=app, url_base_pathname='/dash/')

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')

dash_app.layout = html.Div([
    html.H2('📊 Visualisasi Data Gapminder', style={'textAlign': 'center'}),

    html.Label('Pilih Negara:'),
    dcc.Dropdown(
        df.country.unique(),
        'Indonesia',
        id='dropdown-country',
        style={'width': '50%'}
    ),
    dcc.Graph(id='line-chart'),

    html.Hr(),
    html.P('Dashboard ini menampilkan perkembangan GDP per Kapita tiap negara dari dataset Gapminder.'),
])

@dash_app.callback(
    Output('line-chart', 'figure'),
    Input('dropdown-country', 'value')
)
def update_graph(selected_country):
    dff = df[df.country == selected_country]
    fig = px.line(dff, x='year', y='gdpPercap', title=f'Perkembangan GDP per Kapita - {selected_country}')
    return fig

if __name__ == '__main__':
    app.run(debug=True)
