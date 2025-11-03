from flask import Flask, render_template, request, redirect
from dash import Dash, html, dcc, Input, Output, dash_table
import pandas as pd
import plotly.express as px
import pickle, re
from datetime import datetime

app = Flask(__name__)

# ================== Load Model ==================
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ================== Flask Routes ==================
@app.route('/')
def index():
    # Redirect ke main dashboard Dash
    return redirect('/dash/')

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

@app.route("/hello/<name>")
def hello_there(name):
    now = datetime.now().strftime("%A, %d %B, %Y at %X")
    match_object = re.match("[a-zA-Z]+", name)
    clean_name = match_object.group(0) if match_object else "Friend"
    return f"Hello there, {clean_name}! It's {now}"

@app.route("/pyramid/<height>")
def pyramid(height):
    height = int(height)
    pyramid = ""
    for i in range(height):
        pyramid += " " * (height - i - 1)
        pyramid += "* " * (2 * i + 1)
        pyramid += "<br>"
    return pyramid


# ================== DASH 1: Main Dashboard ==================
def register_dash_main(server):
    dash_app = Dash(__name__, server=server, routes_pathname_prefix='/dash/')
    df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')

    dash_app.layout = html.Div([
        html.H2('📊 Visualisasi Data Gapminder', style={'textAlign': 'center'}),
        html.Label('Pilih Negara:'),
        dcc.Dropdown(df.country.unique(), 'Indonesia', id='dropdown-country', style={'width': '50%'}),
        dcc.Graph(id='line-chart'),
        html.Hr(),
        html.P('Dashboard ini menampilkan perkembangan GDP per Kapita tiap negara dari dataset Gapminder.')
    ])

    @dash_app.callback(Output('line-chart', 'figure'), Input('dropdown-country', 'value'))
    def update_graph(selected_country):
        dff = df[df.country == selected_country]
        return px.line(dff, x='year', y='gdpPercap', title=f'Perkembangan GDP per Kapita - {selected_country}')

    return dash_app


# ================== DASH 2: Controls ==================
def register_dash_controls(server):
    dash_app = Dash(
        __name__,
        server=server,
        routes_pathname_prefix='/dash/controls/',
        external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"],
        suppress_callback_exceptions=True
    )

    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv")

    dash_app.layout = html.Div([
        html.H3("My First App with Data, Graph, and Controls",
                style={"textAlign": "center", "color": "blue", "fontSize": 25}),
        dcc.RadioItems(
            options=[
                {"label": "Population (pop)", "value": "pop"},
                {"label": "Life Expectancy (lifeExp)", "value": "lifeExp"},
                {"label": "GDP per capita (gdpPercap)", "value": "gdpPercap"},
            ],
            value="lifeExp",
            inline=True,
            id="my-radio-buttons-final",
        ),
        html.Div([
            dash_table.DataTable(data=df.to_dict("records"), page_size=11, style_table={"overflowX": "auto"}),
            dcc.Graph(id="histo-chart-final", figure={})
        ])
    ])

    @dash_app.callback(Output("histo-chart-final", "figure"), Input("my-radio-buttons-final", "value"))
    def update_graph(col_chosen):
        fig = px.histogram(df, x="continent", y=col_chosen, histfunc="avg",
                           title=f"Average {col_chosen} by Continent")
        return fig

    return dash_app


# ================== DASH 3: Unfiltered ==================
def register_dash_unfiltered(server):
    dash_app = Dash(__name__, server=server,
                    routes_pathname_prefix='/dash/unfiltered/',
                    suppress_callback_exceptions=True)
    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv")
    options = [{"label": c, "value": c} for c in sorted(df["country"].unique())]

    dash_app.layout = html.Div([
        html.H2("Population Dashboard", style={"textAlign": "center"}),
        dcc.Dropdown(options=options, value="Canada", id="dropdown-selection"),
        dcc.Graph(id="graph-content")
    ])

    @dash_app.callback(Output("graph-content", "figure"), Input("dropdown-selection", "value"))
    def update_graph(value):
        dff = df[df.country == value]
        return px.line(dff, x="year", y="pop", title=f"Population in {value} by Year")

    return dash_app


# ================== Redirect helper (fix 404 without slash) ==================
@app.route('/dash/controls')
def redirect_controls():
    return redirect('/dash/controls/')

@app.route('/dash/unfiltered')
def redirect_unfiltered():
    return redirect('/dash/unfiltered/')


# ================== MAIN ==================
if __name__ == '__main__':
    register_dash_main(app)
    register_dash_controls(app)
    register_dash_unfiltered(app)
    app.run(debug=True, host='0.0.0.0')
