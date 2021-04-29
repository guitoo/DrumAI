from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func
import plotly.graph_objs as go
import plotly.express as px
from umap import UMAP 
from database.objects import Sample, SampleClass,SampleSubClass, Features, UserClassVote, User

import pandas as pd

import streamlit as st

@st.cache(ttl=3600)
def umap_timbral(df):
    timbral_features = ['hardness', 'depth', 'brightness', 'roughness', 'warmth', 'sharpness', 'boominess']

    df[['embed_x', 'embed_y']] = UMAP(random_state=42).fit_transform(df[timbral_features])

    return df

def get_timbral_df():
    
    query = (
        session.query(
            SampleClass.sample_class.label('classe'),
            Features.hardness,
            Features.depth,
            Features.brightness,
            Features.roughness,
            Features.warmth,
            Features.sharpness,
            Features.boominess,
        ).select_from(Sample)
        .join(Sample.sample_class)
        .join(Sample.features)
    ).statement

    df = pd.read_sql(query, engine)

    return df

if __name__ == "__main__":

    global engine
    # engine = create_engine("postgresql+psycopg2://dev:dev@localhost/data")
    engine = create_engine("postgresql+psycopg2://dev:dev@postgres/data")
    Session = sessionmaker(bind=engine)
    global session
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

    st.title('Répartition des classes')

    selected_classes = st.radio('Classes:', ["all classes", "top 9 classes"]) #st.multiselect('Classes', all_classes, default=all_classes)

    if selected_classes == "top 9 classes":
        class_df=class_df[class_df.index[0:9]]

    fig = go.Figure([go.Bar(x=class_df.index, y=class_df)])
    st.plotly_chart(fig)

    timbral_features = ['hardness', 'depth', 'brightness', 'roughness', 'warmth', 'sharpness', 'boominess']

    timbral_df = get_timbral_df()

    st.title('Espace timbral')
    col_a, col_b, col_c = st.beta_columns(3)

    x_axis = col_a.radio('Axis X:', timbral_features)
    y_axis = col_b.radio('Axis Y:', timbral_features, index=1)
    hue = col_c.radio('Color:', ['classe'] + timbral_features)


    
    fig3 = px.scatter(
            timbral_df, x=x_axis, y=y_axis, 
            color=hue,
            hover_data=timbral_features,)
    st.plotly_chart(fig3)

    st.title('Espace timbral: Umap  ')
    umap_df = umap_timbral(timbral_df)

    fig2 = px.scatter(
            umap_df, x="embed_x", y="embed_y", 
            color="classe",
            hover_data=timbral_features)
    st.plotly_chart(fig2)

    st.title('Répartition des classes (utilisateurs)')

    freq = (
        session.query(
            UserClassVote.sample_class,
            func.count(UserClassVote.user_id).label('freq'),
            UserClassVote.sample_id
        ).select_from(UserClassVote)
        .group_by(UserClassVote.sample_id)
        .group_by(UserClassVote.sample_class)
        .subquery("freq")
    )
    query = (
        session.query(
            freq.c.sample_class,
            freq.c.sample_id
        ).distinct(freq.c.sample_id)
        .order_by(freq.c.sample_id, freq.c.freq.desc())
    )

    user_class = pd.read_sql(query.statement,engine)#.set_index('sample_id')
    user_class = user_class.groupby('sample_class').size().sort_values(ascending=False)

    fig4 = go.Figure([go.Bar(x=user_class.index, y=user_class)])
    st.plotly_chart(fig4)

    st.title('Corrections')

    query_user_classes = (
        session.query(
            UserClassVote.sample_class.label('classe'),
            User.name,
            Sample.id
    #         SampleSubClass.sample_subclass
        ).select_from(UserClassVote)
        .join(UserClassVote.user)
        .join(UserClassVote.sample)
        # .join(Sample.sample_subclass)
    )
    df_user = pd.read_sql(query_user_classes.statement, engine)
    st.write(df_user)