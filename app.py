import numpy as np
import pandas as pd
import datetime as dt
import joblib

import plotly.express as px

import streamlit as st


st.title("Prediction of Sales 📈")
st.write("")
st.write("This app allows you to visualize the predicted sales for a store.") 
st.write("Since the available database only goes from 2015/01/01 to 2015/07/31 yet, \
    you can only get the predictions for a range between these two dates.")

st.write("By default the maximum date will be 6 weeks ahead of the minimum date you enter \
     (with a limit on 2015/07/31) but you can choose another date.")
st.write("")

store_id = int(st.text_input(label="Store Id (between 1 and 1115)", value="1"))

min_date = st.date_input(
    label = "Minimum date for prediction",
    value = dt.date(2015, 1, 1),
    min_value = dt.date(2015, 1, 1),
    max_value = dt.date(2015, 7, 31)
)

max_date = st.date_input(
    label = "Maximum date of prediction",
    value = min(min_date + dt.timedelta(weeks=6), dt.date(2015, 7, 31)),
    min_value = min_date,
    max_value = dt.date(2015, 7, 31)
)


features = ['Store', 'DayOfWeek', 'Day', 'Month', 'WeekOfYear', 'Year', 'Promo', 'StateHoliday', \
    'SchoolHoliday', 'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionAge', \
    'IsPromo2Applied',
]

pipeline = joblib.load('model/pipeline.pkl')

df = pd.read_csv('data/2015.csv',  dtype={"StateHoliday": str})
df['Date'] = pd.to_datetime(df['Date'])


def predicted_sales(store_id, min_date, max_date):

    min_date = pd.to_datetime(min_date)
    max_date = pd.to_datetime(max_date)

    df1 = df[(df['Store'] == store_id) & (df['Date'] >= min_date) & (df['Date'] <= max_date)]

    try : 
        df1.loc[df1["Open"] == 1, "Predicted Sales"] = pipeline.predict(df1.loc[df1["Open"] == 1, features])
    except:
        pass

    try :
        df1.loc[df1["Open"] == 0, "Predicted Sales"] = 0
    except :
        pass

    fig = px.line(
            df1, x='Date', y='Predicted Sales', markers=True,
            title=f"Sales for store {store_id} from {min_date.date()} to {max_date.date()}"
    )

    fig.update_layout(
        autosize=False,
        width=2000,
        height=500,
        titlefont=dict(size=30),
    )

    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)

    return fig


# Plot!
st.write("")
st.plotly_chart(predicted_sales(store_id, min_date, max_date), use_container_width=True)