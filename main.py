from flask import Flask
import pickle
from flask import Blueprint, render_template, request, flash, jsonify
import pandas as pd
from meteostat import Point, Daily
from datetime import datetime
import numpy


app = Flask(__name__)
# app.config['SECRET_KEY'] = 'JHKBJD,N7hbjnKLJDFNn,+_098y5uh'

# from website.views import views
# app.register_blueprint(views, url_prefix='/')


@app.route('/', methods=['GET', 'POST'])
def home():

    prediction = 0

    if request.method == "POST":
        product_type = request.form.get("product_type")
        year = request.form.get("year")
        month = request.form.get("month")
        day_of_month = request.form.get("day_of_month")
        procentage = request.form.get("procentage")

        prediction = get_prediction(product_type, procentage, year, month, day_of_month)

    if type(prediction) != numpy.ndarray:
        return render_template("home.html", prediction=prediction)
    else:
        return render_template("home.html", prediction=prediction[0])


def _data_preprocessing(product_type, procentage, year, month, day_of_month):
    try:
        year = int(year)
        procentage = float(procentage)
        month = int(month)
        day_of_month = int(day_of_month)
        feature_vector = []
        season = -1
        types = ["ЙОГУРТ", "КЕФИР", "МАСЛО", "МОЛОКО", "МОРОЖЕНОЕ", "СЛИВКИ", "СМЕТАНА", "СЫР", "ТВОРОГ"]
        day_of_week = datetime(year, month, day_of_month).weekday() + 1

        if product_type.upper() not in types:
            product_type = "ДРУГОЕ"
        if 1 <= int(month) <= 2 or int(month) == 12:
            season = 1
        if 3 <= int(month) <= 5:
            season = 2
        if 6 <= int(month) <= 8:
            season = 3
        if 9 <= int(month) <= 11:
            season = 4

        tavg = get_tavg(day_of_month, month, year)

        X = pd.DataFrame(columns=['procentage', 'tavg', 'season_1', 'season_2', 'season_3', 'season_4',
                                  'product_type_ДРУГОЕ', 'product_type_ЙОГУРТ', 'product_type_КЕФИР',
                                  'product_type_МАСЛО',
                                  'product_type_МОЛОКО', 'product_type_МОРОЖЕНОЕ', 'product_type_СЛИВКИ',
                                  'product_type_СМЕТАНА', 'product_type_СЫР', 'product_type_ТВОРОГ', 'day_of_month_1',
                                  'day_of_month_2', 'day_of_month_3', 'day_of_month_4', 'day_of_month_5',
                                  'day_of_month_6',
                                  'day_of_month_7', 'day_of_month_8', 'day_of_month_9', 'day_of_month_10',
                                  'day_of_month_11',
                                  'day_of_month_12', 'day_of_month_13', 'day_of_month_14', 'day_of_month_15',
                                  'day_of_month_16', 'day_of_month_17', 'day_of_month_18', 'day_of_month_19',
                                  'day_of_month_20', 'day_of_month_21', 'day_of_month_22', 'day_of_month_23',
                                  'day_of_month_24', 'day_of_month_25', 'day_of_month_26', 'day_of_month_27',
                                  'day_of_month_28', 'day_of_month_29', 'day_of_month_30', 'day_of_month_31',
                                  'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7',
                                  'month_8', 'month_9', 'month_10', 'month_11', 'month_12', 'day_of_week_1',
                                  'day_of_week_2', 'day_of_week_3', 'day_of_week_4', 'day_of_week_5', 'day_of_week_6',
                                  'day_of_week_7'])

        for column in X.columns.tolist():
            if column == "procentage":
                feature_vector.append(procentage)
            elif column == "tavg":
                feature_vector.append(tavg)
            else:
                feature_vector.append(0)

        X.loc[0] = feature_vector

        X.loc[0, f"product_type_{product_type.upper()}"] = 1
        X.loc[0, f"season_{season}"] = 1
        X.loc[0, f"day_of_month_{day_of_month}"] = 1
        X.loc[0, f"month_{month}"] = 1
        X.loc[0, f"day_of_week_{day_of_week}"] = 1

        for column in X.columns.tolist():
            X[column] = X[column].astype(int)

        return X.iloc[[0]]
    except Exception as _:
        return "SOME ERROR"


def get_prediction(product_type, procentage, year, month, day_of_month):

    try:
        big_sales_regression_model = pickle.load(open("big_sales_model.pkl", "rb"))
        small_sales_regression_model = pickle.load(open("small_sales_model.pkl", "rb"))
        anomaly_classifier_model = pickle.load(open("classification_model.pkl", "rb"))

        X = _data_preprocessing(product_type, procentage, year, month, day_of_month)
        is_anomaly = anomaly_classifier_model.predict(X)

        if is_anomaly[0] == 1:
            prediction = big_sales_regression_model.predict(X)
        else:
            prediction = small_sales_regression_model.predict(X)

        return prediction
    except Exception as _:
        return "SOME ERROR"


def get_tavg(day, month, year):
    start = end = datetime(year, month, day)
    location = Point(53.893009, 27.567444, 70)
    data = Daily(location, start, end)
    data = data.fetch()
    return data.tavg[0]


if __name__ == "__main__":
    app.run(debug=True)

