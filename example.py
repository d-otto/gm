# -*- coding: utf-8 -*-

from . import gm3s
import plotly.express as px


if __name__ == '__main__':
    model = gm3s(mode='b')
    df_p = model.to_tidy()
    fig = px.line(df_p, x='t', y='value', facet_row='variable')
    fig.show()
    
    
    