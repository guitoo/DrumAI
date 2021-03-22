from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import plotly.graph_objs as go
from database.objects import Sample, SampleClass,SampleSubClass

import pandas as pd

import streamlit as st

if __name__ == "__main__":

    engine = create_engine("postgresql+psycopg2://dev:dev@postgres/data")
    Session = sessionmaker(bind=engine)
    session = Session()

    query_classes = (
        session.query(
            SampleClass.sample_class.label('classe'),
    #         SampleSubClass.sample_subclass
        ).select_from(Sample)
        .join(Sample.sample_class)
        .join(Sample.sample_subclass)
    )


    data_classes = pd.read_sql(query_classes.statement, engine)
    class_df = data_classes.groupby('classe').size().sort_values(ascending=False)

    all_classes = list(class_df.index)


    

    # st.sidebar.text("Menu")
    # tmp = st.number_input('choisir un nombre')
    # st.text(tmp)
    selected_classes = st.radio('Classes:', ["all classes", "top 9 classes"]) #st.multiselect('Classes', all_classes, default=all_classes)

    if selected_classes == "top 9 classes":
        class_df=class_df[class_df.index[0:9]]

    fig = go.Figure([go.Bar(x=class_df.index, y=class_df)])
    st.plotly_chart(fig)