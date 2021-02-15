#%matplotlib inline
import numpy as np
# from scipy import stats
# import matplotlib.pyplot as plt
# import seaborn as sns
# from warnings import filterwarnings
# import itertools
# import math

import pymc3 as pm
# from pymc3 import floatX, model_to_graphviz, forestplot, traceplot
# import theano
# import theano.tensor as tt
# from pymc3.theanof import set_tt_rng, MRG_RandomStreams
#
# import sklearn
# from sklearn import datasets
# from sklearn.preprocessing import scale
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, maxabs_scale, RobustScaler
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import make_moons
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# from sklearn.metrics import accuracy_score, r2_score, confusion_matrix
#
# import networkx as nx

import pickle

import pandas as pd

# import chart_studio.plotly as py
# import plotly.graph_objs as go
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# import ipywidgets as widgets
# import plotly.express as px
# import plotly.figure_factory as ff
from plotly.colors import n_colors

# import time
# import random
# import unicodedata

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
#import dash_table
#import dash_daq as daq
from dash.dependencies import Input, Output, State
#import plotly.tools as tls
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

#-----------------------------------------------------------------------------------------------------------------
COSTAIN_LOGO = "https://www.costain.com/assets/img/logo.png"
COSTAIN_BACKGROUND = 'https://github.com/Raimondo-M/CarbonLibrary-app-test/blob/master/Costain_background.png?raw=true'

COSTAIN_stylesheets = 'https://www.costain.com/assets/css/Costain/font-awesome/css/font-awesome.min.css'

font_family='-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,"Noto Sans",sans-serif,"Apple Color Emoji","Segoe UI Emoji","Segoe UI Symbol","Noto Color Emoji"'

external_stylesheets = [dbc.themes.BOOTSTRAP, COSTAIN_stylesheets,]# 'https://www.costain.com/assets/css/styles.css?v=1.5']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.tittle = 'REVIS - Sustainable Product Development tool'

server = app.server
app.config.suppress_callback_exceptions = True
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True


#-----------------------------------------------------------------------------------------------------------------
Needs_df = pd.read_excel('Survey_REVIS_RM.xlsx', sheet_name = 'Needs')
Feats_df = pd.read_excel('Survey_REVIS_RM.xlsx', sheet_name = 'Features')
categories = {0: 'Transportation sector', 1: 'Academic', 2: 'Costain', 3: 'Overall'}
    
#model_fpath = 'REVIS_surveymodel_' + 'Revis_Survey_V4_June 16, 2020_02.05'
model_fpath = 'REVIS_surveymodel_Revis_Survey_V5_October 21, 2020_10.23'
#model_fpath = 'REVIS_surveymodel_test2_20200253'
with open(model_fpath, 'rb') as buff:
    u = pickle._Unpickler(buff)
    u.encoding = 'latin1'
    data = u.load()
    #data = pickle.load(buff)
    
priorities_needs_df =  pd.DataFrame(data['priorities_needs_df'])
priorities_feats_df =  pd.DataFrame(data['priorities_feats_df'])

model_Q2 = data['model_Q2']
trace_Q2 = data['trace_Q2']
#model_Q3 = data['model_Q3']
trace_Q3 = data['trace_Q3_hier']
#model_Q31 = data['model_Q31']
#trace_Q31 = data['trace_Q31']

color_scale_feat = px.colors.n_colors('rgb(162, 192, 55)','rgb(116,182,174)', 9, colortype='rgb')
color_costain_bar = ['#00234C', 'rgb(35, 77, 97)', '#4F868E', '#69599E', '#4D306B', '#74B6AD']
                     
#px.colors.sequential.Viridis#[[0, '#a1bf37'],  [1.0, '#74b6ad']]#[0.5, '#a7e3bc'],
                #['#a1bf37', '#a2c037', '#9cca5c', '#9ad27c', '#9cd999', '#a5dfb3', '#a7e3bc',
                # '#aae6c5', '#adeace', '#a2eece', '#96f2cf', '#88f5d1', '#77f9d3']

#------------------------------------------------------------------------------------------------------------------
class BNN_network():
    def __init__(self, needs_library = Needs_df, feats_library = Feats_df, trace = trace_Q3, scale = True):
        
        self.trace = trace
        
        self.feats_df = feats_library
        self.needs_df = needs_library
        
        self.priorities_needs_df = priorities_needs_df
        self.priorities_feats_df = priorities_feats_df
        
        #self.w_in_1 = trace.get_values('w_in_1').mean(axis=0).reshape((self.feats_df.shape[0],self.needs_df.shape[0]))
        #self.w_2_out = trace.get_values('w_2_out').mean(axis=0).reshape(-1, 1)
        
        self.pos = {}
        
        self.df_nodes = self.feats_df.rename(columns={"Features": "label"})
        self.df_nodes['node'] = self.df_nodes.index.map(lambda x: '{}_{}'.format('feat', x))
        self.df_nodes['type'] = 'feat'
        self.df_nodes['color'] = self.df_nodes.index.map(lambda x: color_scale_feat[x])#'rgb(162, 192, 55)'
        
        #self.df_nodes['color_scale'] = self.df_nodes.index.map(lambda x: [x, color_scale[x]])

        
        _ = self.needs_df.rename(columns={"Need": "label"})
        _['node'] = _.index.map(lambda x: '{}_{}'.format('need', x))
        _['type'] = 'need'
        _['color'] = 'rgb(0, 156, 222)'
        
        self.df_nodes = self.df_nodes.append(_)
        
        self.df_nodes = self.df_nodes.append({'label': 'Overall satisfaction', 'node': 'out__0', 'type' : 'out_', 'color': 'rgb(222, 124, 0)',
                                              'Description' :'', 'code':'',
                                             'ID':0}, ignore_index=True)
        
        #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        #    print(self.df_nodes)
        
        self.df_edges = self.trace_to_df(trace_Q3, names_sorce = self.feats_df['Features'].to_list(), 
                                         names_target = self.needs_df['Need'].to_list(), 
                                         sorce_type = 'feat', target_type= 'need', x = ['w_f_n'])
        self.df_edges = self.df_edges.append(self.trace_to_df(trace_Q2,  names_sorce = self.needs_df['Need'].to_list(), 
                                                              names_target = ['Construction site'], 
                                                              sorce_type = 'need', target_type= 'out_'))
                 
    def trace_to_df(self, trace, x = ['w_in_1'], names_sorce = [], names_target = [], sorce_type = '', target_type= '', scale = True, sort = True):
            
        df = pm.summary(trace, var_names=[x[0]], credible_interval=0.95)
        
        #df.head()
        n_var = len(names_sorce)
        n_group = int(df.shape[0]/n_var)
        if n_group != len(names_target):
            print('error, list of names sorce/taget not fine')
            return ''
            
        #df['ID'] = df.index().apply(lambda x: names_target[x])
        df['var_type'] = x[0]
        df['sorce_type'] = sorce_type
                             
        df['source'] = list(map(int,(range(n_var) * np.ones((n_group,1))).flatten('F') ))
        df['source_label'] = df['source'].apply(lambda x: names_sorce[int(x)])
        df['source_ID'] =  df['source'].apply(lambda x: '{}_{}'.format(sorce_type, x))
        df['source'] = df['source_ID'].apply(lambda x: self.df_nodes.index[self.df_nodes['node'] == x].values[0])
                                    
        df['target'] = list(map(int, list(range(n_group)) * n_var))
        df['target_label'] = df['target'].apply(lambda x: names_target[int(x)])
        df['target_ID'] =  df['target'].apply(lambda x: '{}_{}'.format(target_type, x))
        df['target'] = df['target_ID'].apply(lambda x: self.df_nodes.index[self.df_nodes['node'] == x].values[0])
        
        
        #sort_list = df['var_titles'].to_list()
        
        #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        #    print(df)
        
        if scale == True:
            df = self.gen_scores(df, trace, sorce_type)
            if sort == True:                                            
                df = df.sort_values(by ='mean_scaled')
        
        if len(x)>1:
            df = df.append(self.trace_to_df(trace, x[1:], names_sorce))
        
        return df
    
    def gen_scores(self, df, trace, var, level = False, norm = True, variable = 'mean', selection = 'all'):
        #Qui
        labels = Feats_df['Features'].array
        categories = {0: 'Transportation sector', 1: 'Academic', 2: 'Costain', 3: 'Overall'} 
        cat1 , cat2, cat3, catG = 0, 1, 2, 3  
    
        n_ = trace_Q3.get_values('w_n_out').reshape(( -1, 4, Needs_df.shape[0] ))#np.exp()#.mean(axis=0)
        d = trace_Q3.get_values('w_f_n').reshape(( -1, Feats_df.shape[0], Needs_df.shape[0] ))

        if norm == True: norm = '_norm' 

        c1 = categories[cat1] + '_' + variable + norm
        c2 = categories[cat2] + '_' + variable + norm
        c3 = categories[cat3] + '_' + variable + norm
        cG = categories[catG] + '_' + variable + norm

        def df_by_cat(cat, c):
            df = pd.DataFrame({})
            scores = np.zeros((d.shape[0], Feats_df.shape[0]))  

            for s in range(d.shape[0]):
                n = n_[s,cat,:].flatten()#
                data = d[s,:,:]#.reshape(( Feats_df.shape[0], Needs_df.shape[0])).T
                score = np.exp(np.dot(data, n))#
                if selection != 'all': 
                    score = score/score[selection].sum()
                else:
                    score = score/score.sum()
                scores[s, :] = score

            df[c] = scores.mean(axis=0)
            df[categories[cat] + '_std' + norm] = scores.std(axis=0)
            df['lable'] = labels
            df = df.set_index('lable')
            return df
        
        
        if ( var == 'need' ) or (level == True):
            e_w = np.exp(trace.get_values('w_in_1'))
            e_w_sum = e_w.mean(axis=0).sum()
            e_w_norm = e_w/e_w_sum

            df['mean_scaled'] = e_w_norm.mean(axis=0)
            df['sd_scaled'] = e_w_norm.std(axis=0)
            df['mc_error_scaled'] = np.sqrt(np.mean((e_w_norm-e_w_norm.mean(axis=0))**2, axis=0))
            df['hpd_2.5_scaled'] = np.percentile(e_w_norm, 2.5, axis=0)
            df['hpd_97.5_scaled'] = np.percentile(e_w_norm, 97.5, axis=0)
        elif var == 'feat':
            dfG = df_by_cat(catG, cG)
            
            e_w = np.exp(d)
            e_w_sum = e_w.mean(axis=0).sum()
            e_w_norm = e_w/e_w_sum

            df['mean_scaled'] = e_w_norm.mean(axis=0).flatten()
            df['sd_scaled'] = e_w_norm.std(axis=0).flatten()
            df['mc_error_scaled'] = np.sqrt(np.mean((e_w_norm-e_w_norm.mean(axis=0))**2, axis=0)).flatten()
            df['hpd_2.5_scaled'] = np.percentile(e_w_norm, 2.5, axis=0).flatten()
            df['hpd_97.5_scaled'] = np.percentile(e_w_norm, 97.5, axis=0).flatten()
        
        #print(e_w.shape, e_w_sum.shape, e_w_norm.shape)
        
        return df
    
    def alpha_df_edges(self, alpha):
        df = self.df_edges
        #G.trace = self.trace
        
        w_2_out = df.loc[df['sorce_type'] == 'need', 'mean'].values
        lim = np.percentile(w_2_out, alpha*100)
        ind_to_rem = (df['sorce_type'] == 'need') & (df['mean'] < lim)
        needs_droppped = df.loc[ind_to_rem, 'source_ID'].to_list()
        df = df.drop(df[ind_to_rem].index)
        
        df = df.drop(df[df['target_ID'].isin(needs_droppped)].index)
                
        return df
    
    def materiality_table_FEATURES_plotly(self, short = '', variable = 'mean',  norm = '', std = True, n_areas = 2, selection = 'all', type = 'Feat'):
        cat1 , cat2, cat3, catG = 0, 1, 2, 3
        
        if type == 'Feat':
            df = self.priorities_feats_df
        else:
            df = self.priorities_needs_df

        if norm == True: norm = '_norm' 
        
        if short != '':
            df = df.loc[short]

        c1 = categories[cat1] + '_' + variable + norm
        c2 = categories[cat2] + '_' + variable + norm
        c3 = categories[cat3] + '_' + variable + norm
        cG = categories[catG] + '_' + variable + norm

        bord_low = min(df[c1].min(), df[c2].min(), df[c3].min(), df[cG].min())
        bord_hig = max(df[c1].max(), df[c2].max(), df[c3].max(), df[cG].min())
        d = (bord_hig-bord_low)/10
        bord_low = bord_low - d
        bord_hig = bord_hig + d

        if selection != 'all' :   
            df = df.iloc[selection]

        df = df.sort_values(by = cG )

        hoverinfo = {3: 'Must have', 2: 'Key to discuss', 1: 'Potential added value', 0: 'To exclude'}

        for c in [cG, c1, c2, c3]:
            df[c] = df[c].map(lambda val: int(round((val - df[c].min()*0.9)*3/(df[c].max()*1.1 - df[c].min()*0.9))) )
            df[c + 'hi'] = df[c].map(lambda val: hoverinfo[val])

        colors = n_colors('rgb(255, 200, 200)', 'rgb(200, 0, 0)', 9, colortype='rgb')
        
        
        fig = ff.create_annotated_heatmap(df[[cG, c1, c2, c3]].values,
                    #labels=dict(x=df_.index, y=[cG, c1, c2, c3], color="w"),
                    x=[categories[3], categories[0], categories[1], categories[2]],
                    y=df.index.array,
                    hovertemplate = "%{y}:<br>%{text}",
                    #hoverlabel=dict(font=dict(family='sans-serif', size=10)),
                    text = df[[cG + 'hi', c1+ 'hi', c2+ 'hi', c3+ 'hi']].values,
                    annotation_text=df[[cG + 'hi', c1+ 'hi', c2+ 'hi', c3+ 'hi']].values,
                    colorscale='YlGnBu',#px.colors.diverging.Portland,#px.colors.sequential.Viridis,
                    #reversescale=True,
                    xgap = 5,
                    ygap = 5,
                   )
        fig.update_layout(
                    xaxis=dict(showgrid=False),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height = 38 * df.shape[0] + 60,
                    font=dict(size=10,)
                                 )
        
        fig.add_traces(self.add_score_bar(df, cat = 0))
        fig.add_traces(self.add_score_bar(df, cat = 1))
        fig.add_traces(self.add_score_bar(df, cat = 2))
        #fig.add_traces(self.add_score_bar(df, cat = 1))
        
        # initialize xaxis2 and yaxis2
        fig['layout']['xaxis2'] = {}
        fig['layout']['yaxis2'] = {}

        # Edit layout for subplots
        fig.layout.xaxis.update({'domain': [0, 0.64]})
        fig.layout.xaxis2.update({'domain': [0.65, 1.]})
        fig.layout.yaxis2.update({'anchor': 'x2', 'visible': False})
        fig.layout.margin.update({'t':40, 'b':20})
        
        return fig
        # iplot(materiality_table_FEATURES_plotly(priorities_feats_df, variable = 'mean', norm = True, selection = 'all'))
    
    def add_score_bar(self, df, needs = 'all', cat = 3):
        c = categories[cat] + '_' + 'mean' + '_norm'
        c_sd = categories[cat] + '_' + 'std' + '_norm'
        c_sam = categories[cat] + '_' + 'sample' + '_norm'

        #df = df.sort_values(by = c )

        #colors = n_colors('rgb(200, 10, 10)', 'rgb(5, 200, 200)',  df.shape[0], colortype='rgb')
        #df['color'] = colors

        bar = go.Bar(#df, 
                     x=df[c].values, y = df.index.array,#df.index.array,
                     #title='REVIS - Value Proposition features',
                     #labels={"mean": 'Prioroty scores', 'var_titles' : ''},
                     orientation='h', 
                     xaxis='x2', yaxis='y2',
                     name=categories[cat],
                     marker =dict(
                            color=color_costain_bar[cat],)
#                             line=dict(
#                                 color='rgba(50, 171, 96, 1.0)',
#                                 width=1))
                    )

#         bar.update_layout(title =  'REVIS - Value Proposition features',
#                           yaxis=dict(
#                                     showgrid=False,
#                                     showline=False,
#                                     showticklabels=True,
#                                     #domain=[0, 0.85],
#                                 ),
#                              xaxis=dict(
#                                     zeroline=False,
#                                     showline=False,
#                                     showticklabels=True,
#                                     showgrid=True,
#                                     #domain=[0, 0.42],
#                                 ),

#                           #legend=dict(x=0.029, y=1.038, font_size=10),
#                             #margin=dict(l=100, r=20, t=70, b=70),
#                           showlegend =False,
#                             paper_bgcolor='rgb(248, 248, 255)',
#                             plot_bgcolor='rgb(248, 248, 255)',
#                          )
#         iplot(bar)

        return bar
    
    def materiality_map_plotly(self, df = priorities_needs_df, 
                               scale = True, barerr = '95%', cat1 = 0, cat2 = 1, n_areas = 2):
        c1 = categories[cat1] + '_' + 'mean' + '_norm'
        c2 = categories[cat2] + '_' + 'mean' + '_norm'
        c1_sd = categories[cat1] + '_' + 'std' + '_norm'
        c2_sd = categories[cat2] + '_' + 'std' + '_norm'

        #df = df.sort_values(by = c )

        bord_low = min(df[c1].min(), df[c2].min())*0.9
        bord_hig = max(df[c1].max(), df[c2].max())
        d = (bord_hig-bord_low)/10
        bord_low = bord_low - d
        bord_hig = bord_hig + d

        areas = [i*(bord_hig - bord_low)/n_areas for i in range(1,n_areas+1)]

        fig = px.scatter(df, 
                         x=c1, 
                         y=c2,
                         text = 'Needs_',
                         color = 'Needs', 
                         labels= {c1,
                                  c2},
                         error_x = c1_sd,
                         error_y = c2_sd,            
                        )
        fig.update_traces(textposition='middle right')
        fig.update_traces(marker=dict(
                            size = 16,
                             line=dict(
                                 color='rgb(255, 255, 255)',
                                 width=2))
         )
        shapes_ = [dict(type="circle",
                                    xref="x",
                                    yref="y",
                                    fillcolor='rgba(108, 171, 134, 0.1)',
                                    line_color='rgba(50, 171, 96, 0)',
                                    x0=bord_hig+i,
                                    y0=bord_hig+i,
                                    x1=bord_hig-i,
                                    y1=bord_hig-i,
                                ) for i in areas]    
        fig.update_layout(title =  'REVIS - Needs materiality map',
                          yaxis=dict(
                                    showgrid=True,
                                    showline=False,
                                    showticklabels=True,
                                    range=[bord_low, bord_hig],
                                ),
                             xaxis=dict(
                                    zeroline=False,
                                    showline=False,
                                    showticklabels=True,
                                    showgrid=True,
                                    #scaleanchor="y", 
                                    #scaleratio=1,
                                    range=[bord_low, bord_hig],
                                ),
                            paper_bgcolor='rgb(248, 248, 255)',
                            plot_bgcolor='rgb(248, 248, 255)',
                          shapes=shapes_,
                          font=dict(size=10,)
                         )

        if barerr == 'std':     
            fig.update_layout(shapes=shapes_ + [dict(type="circle",
                                    xref="x",
                                    yref="y",
                                    line_color ='rgba(108, 171, 134, 0.1)',
                                    fillcolor = fig['data'][np.where(df.index.values == i)[0][0]]['marker']['color'],
                                    opacity=0.1,
                                    x0=r[c1] - r[c1_sd],
                                    y0=r[c2] - r[c2_sd],
                                    x1=r[c1] + r[c1_sd],
                                    y1=r[c2] + r[c2_sd],
                                ) for i, r in df.iterrows()])
        return fig

    def plot_forestplot_plotly(self, x = ['w_in_1'], names = None, scale = True, barerr = 'std', alpha = None):
        
        if names == None: names = self.needs_df['Need'].array
            
        if alpha == None:
            df = self.df_edges
        else:
            df = self.alpha_df_edges(alpha)
            
        df = df.loc[df['sorce_type'] == 'need']
        df['name'] = 'Q3'
        
#         df_Q2 = pm.summary(trace_Q2, var_names=['w_in_1'], credible_interval=0.95)
#         df_Q2['name'] = 'Q2'
#         df_Q2['source_label'] = self.needs_df['Need'].to_list()
        
#         if scale == True:
#             df_Q2 = self.gen_scores(df_Q2, trace_Q2, 'w_in_1', level = True)

        #if len(x)>1:
        #    df_ = pm.summary(self.trace, var_names=[x[1]], credible_interval=0.95)
        #    df_['name'] = x[1]
        #    df_['var_titles'] = names
            #if scale == True:
            #    df_['mean'] = scaler.fit_transform(df_['mean'].values.reshape(-1,1))
            #    df_["hpd_97.5"] = scaler.transform(df_["hpd_97.5"].values.reshape(-1,1))
            #    df_["hpd_2.5"] = scaler.transform(df_["hpd_2.5"].values.reshape(-1,1))

        #    df = df.append(df_)
            
        #df = df.append(df_Q2)
        
        #df["hpd_97.5"] = - df["mean"].to_numpy() + df["hpd_97.5"].to_numpy()
        #df["hpd_2.5"] =  df["mean"].to_numpy() - df["hpd_2.5"].to_numpy()
    
        err_label = {'95%':{'p': - df["mean_scaled"].to_numpy() + df["hpd_97.5_scaled"].to_numpy(), 
                            'n':   df["mean_scaled"].to_numpy() - df["hpd_2.5_scaled"].to_numpy()},
                     'std':{'p': df["sd_scaled"].to_numpy(), 'n': df["sd_scaled"].to_numpy()}}

        fig = px.scatter(df, 
                     x="mean_scaled", y = 'source_label',# orientation='h',
                     error_x = err_label[barerr]['p'],
                     error_x_minus = err_label[barerr]['n'],
                     #hover_data=["tip", "size"],
                     title='Forestplot - Confident Interval = {}'.format(barerr),
                     color = 'name',)

        if len(x)>1:    fig.data[1].visible = False
        #fig.data[1].visible = False
        
        
        fig.update_layout( updatemenus=[ dict( buttons=list([
                           dict(label="Only Q3",
                                 method="update",
                                 args=[{"visible": [True, False]},
                                       ]),
                              dict(label="Both",
                                 method="update",
                                 args=[{"visible": [True, True]},
                                       ])
        ]),
                        direction="down",  pad={"r": 10, "t": 10},  showactive=True,
                        x=1.01, xanchor="left", y = 1.2, yanchor="top"
                    ) ,

             dict( buttons=list([
                             dict(label="std",
                                 method="restyle",
                                 args=[{"error_x": [{'array': df.loc[df["name"] == 'Q3']["sd_scaled"].to_numpy()},
                                                    {'array': df.loc[df["name"] == 'Q2']["sd_scaled"].to_numpy()}]}
                                       ]),
                             dict(label="95%",
                                 method="restyle",
                                 args=[{"error_x": [{'array': df.loc[df["name"] == x[0], "hpd_97.5_scaled"].to_numpy(),  
                                                    'arrayminus': df.loc[df["name"] == x[0], "hpd_2.5_scaled"].to_numpy()},
                                                   {'array': df.loc[df["name"] == 'Q2', "hpd_97.5_scaled"].to_numpy(),  
                                                    'arrayminus': df.loc[df["name"] == 'Q2', "hpd_2.5_scaled"].to_numpy()}]},
                                       ]),

        ]),
                        direction="down",  pad={"r": 10, "t": 10},  showactive=True,
                        x=0.81, xanchor="left", y = 1.2, yanchor="top"
                    ),] )
        
        
        
        return fig    
    
    def plot_shanky(self, alpha = None):
        
        if alpha == None:
            df = self.df_edges
        else:
            df = self.alpha_df_edges(alpha)
        
        #print(df['source'])
        #print(df['mean_scaled'])
        
        data_trace = dict(type='sankey',
                   domain = dict(x = [0,1], y = [0,1]),
                   orientation = 'h',
                   valueformat = '.0f',
                   node = dict(
                       pad = 10, thickness = 30,
                       line = dict( 
                           color = 'black',
                           width = 0.5,),
                        label = self.df_nodes['label'].to_list(),
                        color = self.df_nodes['color'].to_list(),
                           
                       ),
                   link = dict(
                           source = df['source'],
                           target = df['target'],
                           value = df['mean_scaled'].apply(lambda x: 0.00001 if x == 0 else x),#df['value'],#
                           color = df['mean_scaled'].apply(lambda x: 'rgba(0,0,0,{})'.format((3*x)))
                           )
                   )
        
        self.plot_init()
        
        layout = self.layout
        
#         layout['updatemenus']=[dict( buttons=list([
#                              dict(label="Normilised",
#                                  method="restyle",
#                                  args=[{'link': {"value": df['mean_scaled'].apply(lambda x: 0.00001 if x == 0 else x).values, } }]),
#                              dict(label="Not Normilised",
#                                  method="restyle",
#                                  args=[{'link': {"value": df['mean_scaled'].apply(lambda x: 0.00001 if x == 0 else x).values, } } ])
#                                         ]),
#                             direction="down",  pad={"r": 10, "t": 10},  showactive=True,
#                             x=1.01, xanchor="left", y = 1.2, yanchor="top") ,
#                                        ]
        
        fig = dict(data=[data_trace], layout=layout)
        #print(fig)
        return fig
    
    def heatmap_feat_vs_needs(self):
        #e_w = np.exp(trace_Q31.get_values('w_in_1')).reshape((self.feats_df.shape[0] , self.needs_df.shape[0], -1))
        #e_w_sum = e_w.mean(axis=2).sum(axis=1)
        #print(e_w.mean(axis=2))
        #e_w_norm = np.vstack([e_w[i, :, :]/s for i, s in enumerate(e_w_sum)]).T

        
        #data = e_w_norm.mean(axis=0).reshape((self.feats_df.shape[0] , self.needs_df.shape[0])).T
        
        #print(data.shape)
        
        data = self.get_values('w_in_1').mean(axis=0).reshape((self.needs_df.shape[0] ,self.feats_df.shape[0] ))
        #data = np.exp(data)
        fig = ff.create_annotated_heatmap(data,
                        #labels=dict(x="Needs", y="Features", color="w"),
                        y=self.needs_df['Need'].array,
                        x=self.feats_df['Features'].array,
                        annotation_text=np.around(data, decimals=2),
                        colorscale=px.colors.sequential.Viridis,
                       )
        fig.update_xaxes(side="top")
        fig.update_layout(height = 722)
        
        return fig
    
    def sunburst(self):
        
        val = 0.2
        need = 'Workers healt and Safety'
        fig = go.Figure({
          'type': "pie",
          'showlegend': False,
          'hole': 0.8,
          'rotation': 90,
          'values': [ val*100, (1-val)*100, 100],
          'text': ["", "", ''],
          #'direction': "clockwise",
          'textinfo': "text",
          'textposition': "inside",
          'marker': {
            'colors': ["rgb(162, 192, 55)", "rgba(0,0,0,0.1)", "rgba(0,0,0,0)"]
          },
          #'labels': ["0-10", "10-50", "50-200", "200-500", "500-2000", ""],
          #'hoverinfo': "label"
        })
    
        fig.update_layout(height = 250, width = 300,
                          title = {'text': "Top prioty need"},
                          
            #xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,  fixedrange= True,),
            #yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.3,1], fixedrange= True,),
            #displayModeBar =False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin = go.layout.Margin(l = 25, r = 25, b = 0, t= 40,),
            hoverlabel=dict(font=dict(family=font_family, size=12)),
            annotations=[
                        dict(x=0.5,y=0.55,
                            text=val,
                            font=dict(family=font_family, size=36),
                             showarrow=False,
                            #textposition='top center'
                        ),
                    dict(x=0.5,y=0.4,
                            text=need,
                            font=dict(family=font_family, size=14),
                         showarrow=False,
                            #textposition='bottom center'
                        )],
                                
             )
        print(fig)
        return fig
    
    def Waffle_feat_vs_needs(self, needs = 'all'):
        score = self.df_edges.loc[self.df_edges['sorce_type'] == 'need', "mean_scaled"].values

        waffle_score = np.rint(np.sort(score*100))

        waffle_score_i = np.argsort(score*100)
        

        squares = np.zeros((10,10))
        list_ = self.feats_df['Features'].values
        label =  []
        annotation = ['']*100
        k = 0
        for i in range(10):
            for j in range(10):
                if waffle_score[k] == 0:
                    if k+1 != len(list_):
                        k = k+1
                
                waffle_score[k] = waffle_score[k]-1
                squares[i,j] = waffle_score_i[k]   #k
                label.append( '{}<br>{:d}%'.format(
                    list_[waffle_score_i[k]] ,
                    int(round(score[waffle_score_i[k]]*100) ))
                            )

        label = list(zip(*[iter(label[::-1])]*10))
        annotation = list(zip(*[iter(annotation)]*10)) 
        
        fig = ff.create_annotated_heatmap(squares[::-1, ::-1],
                        #labels=dict(x="Needs", y="Features", color="w"),
                        y=[i for i in range(10)],#[Needs_df['Need'].array[i] for i in r]  +["Tot Score"], #  ,# 
                        x=[i for i in range(10)],#Feats_df['Features'].array,
                        #annotation_text=np.around(data, decimals=4),
                        #hoverlabel=label,
                        annotation_text=annotation,
                        text=label,
                        hovertemplate = "<b>%{text}</b>"+ "<extra></extra>",
                        colorscale=px.colors.sequential.Viridis,#[[0.0, 'rgb(162, 192, 55)'],
                                   # [1.0, 'rgb(116,182,174)']],#px.colors.sequential.Viridis,
                        xgap = 2,
                        ygap = 2,
                       )
        #print(fig)
        fig.update_layout(height = 300, width = 300,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,  fixedrange= True,),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,  fixedrange= True,),
                    #displayModeBar =False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin = go.layout.Margin(l = 20, r = 5, b = 0, t= 30,),
                    hoverlabel=dict(font=dict(family=font_family, size=12),
                                    ),
                    )
        
#         legend = px.scatter(self.feats_df, x=[0]*len(score), y=0.5*waffle_score_i, 
#                             text='Features', color='Features',
#                            )
        y = np.linspace(0, 10, len(score))
        y_ = np.zeros(len(score))
        for i, e in enumerate(waffle_score_i): 
            y_[e] = y[i]
            
        legend = go.Figure(data=go.Scatter(
                    x=[0]*len(score),
                    y=y_,#[y[i] for j,i in enumerate(waffle_score_i)],
                    mode='markers+text',
                    text=self.feats_df['Features'].values,
                    textposition='middle right',
                    hoverinfo='skip',
                    marker=dict(size=[16]*len(waffle_score_i),
                                color=self.df_nodes['color'].values)#[self.df_nodes.iloc[i]['color'] for i in waffle_score_i])  #fig['data'][0]['colorscale'][i+1][1]
                ), )
        legend.update_layout(
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,  range=[-0.27,10], fixedrange= True,),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,  fixedrange= True,),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    #modebar = dict(displayModeBar =False),
                    width = 300,
                    height = 320,
                    margin = go.layout.Margin(l = 20, r = 5, b = 0, t= 4,),
                    font=dict(family=font_family, size=12),)
                            
        return fig, legend

    #    iplot(Waffle_feat_vs_needs(trace_Q3_sig, 1))
    
    def edge_init(self):
        return go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5,color='rgba(255,0,0,1)'),
            hoverinfo='text',
            mode='lines',
            text=[],
            )
    
    def plot_init(self):
        self.edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5,color='rgba(0,0,0,0)'),
            hoverinfo='text',
            mode='lines',
            text=[],
            )
        
        self.node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                opacity = 0.8,
                #showscale=True,
                # colorscale options
                #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                #colorscale='YlGnBu',
                #reversescale=True,
                color=[],
                size=[],
                #colorbar=dict(
                #    thickness=15,
                #    title='Node Connections',
                #    xanchor='left',
                #    titleside='right'
                #),
                line=dict(width=2)))
        
        self.layout=go.Layout(
            title='Air pollution monitoring system',
            titlefont=dict(size=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            hovermode='closest',
            height = 500,
            width = 900,
            font = dict(family=font_family, 
                        size=11),
            #margin=dict(b=20,l=20,r=20,t=40),
            #annotations=[ dict(
            #    text="Python code: <a href='https://plot.ly/ipython-notebooks/network-graphs/'> https://plot.ly/ipython-notebooks/network-graphs/</a>",
            #    showarrow=False,
            #    xref="paper", yref="paper",
            #   x=0.005, y=-0.002 ) ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range  = [-1, 4.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin = go.layout.Margin(l= 50, r= 15, b = 0, t = 40,),
        )
    
BNN_Q4 = BNN_network(Needs_df, Feats_df, trace_Q3)

#------------------------------------------------------------------------------------------------------------------
def navbar():
    return dbc.Navbar([
       dbc.NavLink(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=COSTAIN_LOGO, height="40px"), width="auto"),
                        dbc.Col(dbc.NavbarBrand("SPD - Decision Support System", className="ml-2"), width="auto"),
                    ],
                    align="center",
                    no_gutters=True,
                    className="flex-nowrap",
                ),
                href="/h/",
            ),        
        #bills_drop,
        #log,
        #html.Div(user_name),
    ],
        color="dark",
        dark=True,
        style={"position": "fixed", 'top':'0', 'z-index':'10', 'width':'100%'},
    )

# ----------------------------------------------------------------------------------------------------
# print([ {"label": 'All', "value": 'All'}] + 
#                     [
#                         {"label": Needs_df.loc[Needs_df['Need'] == col]['Need_short'].values[0], 
#                                            "value": col} for col in BNN_Q4.priorities_needs_df.index.array
#                     ])
controls_needs = dbc.FormGroup(
            [
                dbc.Label("Select Needs", width=2),
                dbc.Button('All', id="need-all", color = 'primary', className="mr-1", block=False),
                dbc.Col([
                    dcc.Dropdown( id="need-selection",
                        options=  [ {"label": Needs_df.loc[Needs_df['Need'] == col]['Need_short'].values[0], 
                                               "value": col} for col in BNN_Q4.priorities_needs_df.index.array
                        ]  ,
                        value=None,
                        multi=True
                    )
                ],
                 width=9,
                ),
            ],
            row=True,
        )

controls_feats = dbc.FormGroup(
            [
                dbc.Label("Select VP Features", width=2),
                dbc.Button('All', id="feat-all", color = 'primary', className="mr-1", block=False),
                dbc.Col([
                    dcc.Dropdown( id="feat-selection",
                        options=  [ {"label": Feats_df.loc[Feats_df['Features'] == col]['Feat_short'].values[0], 
                                               "value": col} for col in BNN_Q4.priorities_feats_df.index.array
                        ]  ,
                        value=None,
                        multi=True
                    )
                ],
                 width=9,
                ),
            ],
            row=True,
        )

controls_map = dbc.FormGroup(
            [
                dbc.Label("X axes"),
                dcc.Dropdown( id="x-map",
                    options=  [ {"label": item, "value": key} for key, item in categories.items()
                    ]  ,
                    value=0,
                    clearable=False,
                ),
                dbc.Label("Y axes"),
                dcc.Dropdown( id="y-map",
                    options=  [ {"label": item, "value": key} for key, item in categories.items()
                    ]  ,
                    value=1,
                    clearable=False,
                ),
            ],
        )

page_title = '''
# Air Quality Monitoring Service
##### In support of the global crisis around our air quality, in Costain, we are developing a service to visulalise, manage and improve air quality in our environment.
'''
intro_text_1 = '''
Air pollution is the largest environmental risk to public health in urban areas. 
Investing in cleaner air and tackling air pollution can deliver £1.6 billion to the UK economy and prevent almost 17.000 
premature deaths each year[(1)](https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/770715/clean-air-strategy-2019.pdf). 
That is why the UK has adopted tougher, legally binding ceilings for national emissions of air pollutants for 2020 and 2030.

The problem is particularly enhanced in towns and urban environments. 
Accurate and wide monitoring of the air pollutants across urban areas is necessary to tackle the challenge. 
It can enhance better urban planning prioritising the wellbeing of the citizen and providing safe and healthy places in 
which to live, as well as enabling better decisions to operate inside law limits and identify polluters. 

The COSTAIN Air Quality Monitoring Service will enable solutions to reduce NO2 and other pollutants by accurately measuring 
real-time and predictive emission levels in such areas. 

![](https://www.revismonitors.co.uk/assets/img/revisArchitecture.png)


#### Understending the challenge

Due to the extensive use of modern road vehicles, we face a significant challenge in monitoring usage and hence the air 
quality caused by usage in the public highway. 
Specifically, NO2 concentrations are the statutory air quality limit that the UK is currently failing to meet. 
Many partners have a role to play. Local authorities, industries, asset operators, universities and citizens.

A solution for this challenge needs to meet their interest to be adopted and scale-up. 
We run the Sustainable Product Development analysis aiming to understand their expectation and find the optimal 
compromise to satisfy them.

This interactive report shows the results of the analysis for designing an AQ Monitoring Service according to the 
sustainability principles.
The analysis's scope is to inspire this service's exploitation plan to satisfy all the actors and the environment.
                                                
#### Needs to address
                                                '''
revis_arct_img = 'https://www.revismonitors.co.uk/assets/img/revisArchitecture.png'

def need_card(i):
    return dbc.Card(
                html.Div([
                    dbc.CardImg(
                        src = Needs_df.iloc[i]['Img'],#"https://image.shutterstock.com/image-vector/eco-city-background-transport-people-260nw-1209178480.jpg",
                        top = True),
                    dbc.CardBody(Needs_df.iloc[i]['Need'].replace('<br>',''))#html.H5(),
                    ],
                style = {'cursor': 'pointer'},
                id = f"card-needs-{i}",),
            color="info", inverse=True,
            #style = {'backgroundColor': 'rgba(255,255,255,0.4)', },
            style = {'margin-left': '7px', 'margin-right': '7px'}
            )

def need_card_collapse(i):
    return dbc.Collapse(
                    dbc.Card(
                        dbc.CardBody(Needs_df.iloc[i]['Need'].replace('<br>','')),
                        color="info", outline = True,
                        style = { 'margin-left': '7px', 'margin-right': '7px'}),
                    id=f"collapse-needs-{i}",
                        )

intro_card = [dbc.Label(""),
                #dbc.Card([
                        dbc.Row([
                            dbc.Col(html.Img(src = COSTAIN_LOGO, height = "120px"), width = {"size": 3, "offset": 0},
                                style = {'textAlign': 'right', 'padding': '8px'}),
                            dbc.Col(dcc.Markdown(page_title, id = 'page_title'), width = {"size": 6, "offset": 0})]),
                        dbc.Label(""),
                        dbc.Row(
                             dbc.Col(dcc.Markdown(intro_text_1, id = 'intro_text_1'), width = {"size": 6, "offset": 3})),
                        dbc.Col([dbc.Label(""),
                                ], width={"size": 12, "offset": 0.5}),
                #         ],
                #         body=True,
                #         style = {'backgroundColor':'rgba(255,255,255,0.4)'},
                # ),
                dbc.Col([
                        dbc.CardDeck([need_card(i) for i in range(6)]),
                        html.Div([need_card_collapse(i) for i in range(6)]),
                        dbc.Label(""),
                        dbc.CardDeck([need_card(i) for i in range(6,12)]),
                        html.Div([need_card_collapse(i) for i in range(6,12)]),
                    ]
                    , width = {"size": 10, "offset": 1}),
              ]


main_layout =  dbc.Col(intro_card +
                       [    dbc.Label(""),
                            dbc.Card(
                                    [
                                    dbc.Row(dbc.Col(controls_needs, md=10)),
                                    dbc.Col([dcc.Graph(id='table_needs', 
                                                       figure=BNN_Q4.materiality_table_FEATURES_plotly(type = 'Need'),
                                                       config={'displayModeBar': False}
                                                      ),
                                            ], width={"size": 12, "offset": 0.5}),
                                    ], 
                                    body=True,
                                    style = {'backgroundColor':'rgba(255,255,255,0.4)'},
                            ),

                            dbc.Label(""),
                            dbc.Card(
                                    [
                                    dbc.Row(dbc.Col(controls_feats, md=10)),
                                    dbc.Col([dcc.Graph(id='table_feats', 
                                                       figure=BNN_Q4.materiality_table_FEATURES_plotly(type = 'Feat'),
                                                       config={'displayModeBar': False}
                                                      ),
                                            ], width={"size": 12, "offset": 0.5}),
                                    ], 
                                    body=True,
                                    style = {'backgroundColor':'rgba(255,255,255,0.4)'},
                            ),
                        
                            dbc.Label(""),
                            dbc.Card(
                                    [
                                    dbc.Row([dbc.Col(controls_map, md=2),
                                        dbc.Col([dcc.Graph(id='materialityplot', 
                                                           figure = BNN_Q4.materiality_map_plotly(priorities_needs_df, 
                                                                          cat1 = 2, cat2 = 0, scale = True, barerr = 'std',  ),
                                                           config={'displayModeBar': False}
                                                          ),
                                                ], width={"size": 10, "offset": 0.5}),
                                                ],
                                        align="center"),
                                    ], 
                                    body=True,
                                    style = {'backgroundColor':'rgba(255,255,255,0.4)'},
                            ),

                        dcc.Graph(id='waffle',
                                   figure = BNN_Q4.Waffle_feat_vs_needs()[0]),
                        dcc.Graph(id='waffle_leg',
                                   figure = BNN_Q4.Waffle_feat_vs_needs()[1]),

                        dcc.Graph(id='network-graph',
                                   figure = BNN_Q4.plot_shanky()  ),#.alpha_subgraph(0.7) ), #

                        dcc.Slider(id='alpha-slider', min=0,  max=1, step=0.1, value=0,),
                        dcc.Graph(id='forestplot',
                                       figure = BNN_Q4.plot_forestplot_plotly(['w_2_out'])  
                                 ),

#     iplot(materiality_map_plotly(priorities_needs_df, cat1 = 2, cat2 = 1, names = Needs_df['Need'].array, x = ['w_in_1_h'], scale = True, barerr = 'std',  ), filename='networkx')
#     iplot(materiality_map_plotly(priorities_needs_df, cat1 = 1, cat2 = 0, names = Needs_df['Need'].array, x = ['w_in_1_h'], scale = True, barerr = 'std',  ), filename='networkx'),
                        
#                         dcc.Graph(id='heatmap',
#                            figure = BNN_Q4.heatmap_feat_vs_needs( )),

                                    dbc.Col([
                                        dbc.Label("Scenario Variables"),
                                        dbc.Label(" "),
                                        dbc.Label(" "),
                                        dbc.Label("Sensors accuracy", color="secondary",),
                                        dcc.Slider(id='state_var_0', min=0,  max=1, step=0.01, value=0.33,),
                                        #dbc.Tooltip(f"Tooltip on 0", target=f"state_var_0",placement='bottom'),
                                        dbc.Label("Data analytics", color="secondary",),
                                        dcc.Slider(id='state_var_1', min=0,  max=1, step=0.01, value=0.33,),
                                        dbc.Label("Cost of the system", color="secondary",),
                                        dcc.Slider(id='state_var_2', min=0,  max=1, step=0.01, value=0.33,),
                                        dbc.Label("Brands name ", color="secondary",),
                                        dcc.Slider(id='state_var_3', min=0,  max=1, step=0.01, value=0.33,),
                                        dbc.Label("Mainenance frequency", color="secondary",),
                                        dcc.Slider(id='state_var_4', min=0,  max=1, step=0.01, value=0.33,),
                                        dbc.Label("Problematic preferences", color="secondary",),
                                        dcc.Slider(id='state_var_5', min=0,  max=1, step=0.01, value=0.33,),
                                        dbc.Label("Unclear technology", color="secondary",),
                                        dcc.Slider(id='state_var_6', min=0,  max=1, step=0.01, value=0.33,),
                                        dbc.Label("Fluid participation", color="secondary",),
                                        dcc.Slider(id='state_var_7', min=0,  max=1, step=0.01, value=0.33,),
                                ], width={"size": 4, "offset": 0.5}),
    ])


# ----------------------------------------------------------------------------------------------------

app.layout = html.Div(children=[
    
    html.Div(navbar(), id='nav-bar'),
    
    html.Div([], style={'padding-bottom': '72px'}),
    
    dbc.Row(dbc.Col(html.Img(src=COSTAIN_BACKGROUND),
               style={"position": "fixed", 'textAlign':'right', 'z-index':'-10', })),
        
    dcc.Location(id='url', refresh=False),
    
    html.Div(id='page-content', children = main_layout),
    
    html.Div(children = 'OUT', id='log-state', style={'display': 'none'}),

])

@app.callback(
    [Output(f"collapse-needs-{i}", "is_open") for i in range(12)],
    [Input(f"card-needs-{i}", "n_clicks") for i in range(12)],
    #[State(f"collapse-needs-top", "is_open"),State(f"collapse-needs-bottom", "is_open")],# for i in range(1, 4)],
)
def toggle_collapse(n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11):#, is_open_t, is_open_b):
    ans = [False]*12
    if not any([n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11]):
        return ans

    ctx = dash.callback_context
    if not ctx.triggered:
        return ans
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    i = int(button_id.split("-")[-1])
    ans[i] = True
    return ans


@app.callback(
    Output("need-selection", 'value'),
    [Input("need-all", 'n_clicks')])
def all_need_button_click(all):
    return list(BNN_Q4.priorities_needs_df.index.array)


@app.callback(
    Output('table_needs', 'figure'),
    [Input("need-selection", 'value')])
def update_need_table(list):
    if (list == None) or (list == -1) or (list == []):
        list = BNN_Q4.priorities_needs_df.index.array
        
    fig = BNN_Q4.materiality_table_FEATURES_plotly(type = 'Need', short = list)
    return fig

@app.callback(
    Output("feat-selection", 'value'),
    [Input("feat-all", 'n_clicks')])
def all_feat_button_click(all):
    return list(BNN_Q4.priorities_feats_df.index.array)


@app.callback(
    Output('table_feats', 'figure'),
    [Input("feat-selection", 'value')])
def update_feat_table(list):
    if (list == None) or (list == [-1]) or (list == []):
        list = BNN_Q4.priorities_feats_df.index.array
        
    fig = BNN_Q4.materiality_table_FEATURES_plotly(type = 'Feat', short = list)
    return fig

@app.callback(
    Output('materialityplot', 'figure'),
    [Input('x-map', 'value'), Input('y-map', 'value')])
def update_map(x, y):
    return BNN_Q4.materiality_map_plotly(priorities_needs_df, 
                  cat1 = x, cat2 = y, scale = True, barerr = 'std',  )

# @app.callback(
#     [Output('network-graph', 'figure'), Output('waffle', 'figure'), Output('waffle_leg', 'figure')],
#     [Input('alpha-slider', 'value')])
# def update_output(value):
#     if float(value) == 0:
#         a = BNN_Q4.plot_shanky()
#         b, c = BNN_Q4.Waffle_feat_vs_needs()
#     else:
#         a = BNN_Q4.plot_shanky(float(value))
#         b, c= BNN_Q4.Waffle_feat_vs_needs(int((1-float(value))*11)+1)
#     return a, b, c

# -----------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    app.run_server(debug=True)