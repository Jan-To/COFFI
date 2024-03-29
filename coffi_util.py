import shap.datasets
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

import numpy as np
import pandas as pd
import panel as pn
import seaborn as sns
import time
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, ColorBar, WheelZoomTool, CDSView, BasicTicker, FixedTicker, PrintfTickFormatter, FuncTickFormatter
from bokeh.plotting import figure
from bokeh.transform import linear_cmap
from bokeh.models.widgets.tables import TableColumn, HTMLTemplateFormatter
from matplotlib.colors import LinearSegmentedColormap

from bisect import bisect, bisect_left
from scipy import linalg

def classify(x, bounds):
    for i, b in enumerate(bounds):
        if x < b:
            return i
    return len(bounds)

def load_dataset(name):
    if name=="iris":
        dataset = sklearn.datasets.load_iris()
        data = dataset.data
        target = dataset.target
        features = dataset.feature_names
        classes = dataset.target_names
        categories = get_categories(data, [False, False, False, False])
    if name=="breast":
        dataset = sklearn.datasets.load_breast_cancer()
        data = dataset.data[:,:18]
        target = dataset.target
        features = dataset.feature_names[:18]
        classes = dataset.target_names
        categories = get_categories(data, [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    if name=="income":
        X, y  = shap.datasets.adult()
        data = X.to_numpy()[:5000,:]
        target = y[:5000]
        features = X.columns
        classes = ["<=50k", ">50k"]
        categories = get_categories(data, [False, True, True, True, True, True, True, True, False, False, False, True])
    if name=="diabetes":
        X = pd.read_csv("./data/diabetes.csv")
        data = X.to_numpy()[:,:-1]
        target = np.invert(np.array(X.to_numpy()[:,-1], dtype=bool))
        features = X.columns[:-1]
        classes = ["diabetes","no diabetes"]
        categories = get_categories(data, [False, False, False, False, False, False, False, False])
    if name=="heart-failure":
#         select = [0,4,7,8,11,-1]
        select = [0,1,2,3,4,5,6,7,8,9,10,12]
        X = pd.read_csv("./data/heart_failure_clinical_records_dataset.csv").iloc[:,select]
        data = X.to_numpy()[:,:-1]
        target = np.invert(np.array(X.to_numpy()[:,-1], dtype=bool))
        features = X.columns[:-1]
        classes = ["fatal", "non-fatal"]
        is_cat = [False, True, False, True, False, True, False, False, False, True, True, False]
        categories = get_categories(data, [is_cat[i] for i in select[:-1]]) 
    if name=="shuttle":
        X = np.loadtxt("./data/shuttle/shuttle-train.txt")
        data = X[:,:-1]
        target = X[:,-1].astype(int)-1
        features = ["1","2","3","4","5","6","7","8","9"]
        classes = ["Rad Flow", "Fpv Close", "Fpv Open", "High", "Bypass", "Bpv Close", "Bpv Open"]
        categories = get_categories(data, 9*[False])
    if name=="robot24":
        X = pd.read_csv("./data/robot/sensor_readings_24.txt")
        data = X.to_numpy()[:,:-1]
        target = X.to_numpy()[:,-1].astype(int)-1
        features = ["180°","-165°","-150°","-135°","-120°","-105°","-90°","-75°","-60°","-45°","-30°","-15°","0°",
                    "15°","30°","45°","60°","75°","90°","105°","120°","135°","150°","165°"]
        classes = ["Move-Forward", "Slight-Right-Turn", "Sharp-Right-Turn", "Slight-Left-Turn"]
        categories = get_categories(data, 24*[False])
    if name=="robot4":
        X = pd.read_csv("./data/robot/sensor_readings_4.txt")
        data = X.to_numpy()[:,:-1]
        target = X.to_numpy()[:,-1].astype(int)-1
        features = ["front","left","right","back"]
        classes = ["Move-Forward", "Slight-Right-Turn", "Sharp-Right-Turn", "Slight-Left-Turn"]
        categories = get_categories(data, 4*[False])
    if name=="u-classes":
        X = pd.read_excel("./data/thermo.xlsx")
        data = X[["Name","u1","u2","u3","u4"]].to_numpy()
        target = X['Group_ID'].to_numpy()
        feats = ["u1","u2","u3","u4"]
        classes = X['Group'].value_counts().index.to_numpy()
        categories = get_categories(data[:,1:], 4*[False])
        return pd.DataFrame(data, columns=['Name']+feats), target, feats, classes, categories, None
    if name in ["u1","u2","u3","u4"]:
        X = pd.read_excel("./data/thermo.xlsx")
        feats = ['Dipole Mom.', 'Polarizab.', 'Anisotr.', 'Norm. Aniso.', 'H-Bond Acc.', 'H-Bond Don.', 
                  'HomoLumoGap', 'IonizationEnergy', 'ElectronAffinity', 'Molar Mass']
        data = X[['Name']+feats].to_numpy()
        target_raw = X[name].to_numpy()
        histo_data = np.histogram(target_raw, bins=20)
        if name=="u1":
            bounds = histo_data[1][[12,15]]
        if name=="u2":
            bounds = histo_data[1][[7,11,15]]
        if name=="u3":
            bounds = histo_data[1][[9,14]]
        if name=="u4":
            bounds = histo_data[1][[9,14]]
        histo_data = bounds, *histo_data
        target = np.array([classify(u, bounds) for u in target_raw])
        classes = [name+' < '+str(round(b,1)) for b in bounds] + [name+' > '+str(round(bounds[-1],1))]
        categories = get_categories(data[:,1:], len(feats)*[False])
        return pd.DataFrame(data, columns=['Name']+feats), target, feats, classes, categories, histo_data

    if not isinstance(features, list):
        features = features.tolist()
    
    if name == "shuttle" or name=="robot24" or name=="robot4":
        n = 400
        new_data = np.zeros((0,data.shape[1]))
        new_target = np.zeros(0, dtype=int)
        np.random.seed(0)
        for i in range(np.unique(target).shape[0]):
            possible_inds = np.where(target == i)[0]
            rnd_ind = np.random.choice(possible_inds, min(n,possible_inds.shape[0]), replace=False)
            new_data = np.append(new_data, data[rnd_ind,:], axis=0)
            new_target = np.append(new_target, target[rnd_ind])
        data = new_data
        target = new_target

    return pd.DataFrame(data, columns=features), target, features, classes, categories, None


def get_categories(data, which):
    cat_data = data[:,which]
    categories = []
    c = 0
    for i in which:
        if i:
            categories += [np.arange(np.amax(np.unique(cat_data[:,c]))+1)]
            c += 1
        else:
            categories += [None]
    return categories

class Shifter():
    def _reset(self):
        self.by = []
        
    def set_by(self, by):
        self._reset()
        self.by = by
        
    def set_bounds(self, bounds):
        self.bounds = bounds
    
    def fit(self, X, y=None, sample_weight=None):
        self._reset()
        return self
    
    def transform(self, X):
        return X
    
    def inverse_transform(self, X):
        # shift by support vector
        if len(self.by) > 0:
            X += self.by
        # clamp to be in bounds
        X = X.clip(self.bounds[0], self.bounds[1])
        return X
    
class SVD():
    components_ = np.zeros((2,0))
    V = None
    
    def fit(self, X, y=None):
        if X.shape[1] > 2:
            _, _, V = linalg.svd(X, full_matrices=False)
            # flip eigenvectors' sign to enforce deterministic output
            max_abs_rows = np.argmax(np.abs(V), axis=1)
            signs = np.sign(V[range(V.shape[0]), max_abs_rows])
            V *= signs[:, np.newaxis]
            
            self.V = V
            self.components_ = V[:2]
        elif X.shape[1] == 2:
            self.V = np.array([[1.0,0.0],[0.0,1.0]])
            self.components_ = np.array([[1.0,0.0],[0.0,1.0]])
        elif X.shape[1] == 1:
            self.V = np.array([[0.7071],[0.7071]])
            self.components_ = np.array([[0.7071],[0.7071]])
        return self
            
    def transform(self, X):
        X = np.dot(X, self.components_.T)
        return X
    
    def inverse_transform(self, X):
        X = np.dot(X, self.components_)
        return X

def update_table(tbl, src, dataset, predictor, embedding, params, full=True):
    point = pd.DataFrame([dataset.point[dataset.features]])
    lin_cm = LinearSegmentedColormap.from_list("mycmap", \
                    ["white", params.palette[predictor.predict(point)[0]]])
    
    # update table source
    red_influence = np.linalg.norm(embedding.emb['pca'].components_, axis=0)
    red_influence /= np.sum(red_influence)
    
    for i, f in enumerate(dataset.features):
        if f not in dataset.selected_features:
            red_influence = np.insert(red_influence, i, 0)
    
    if full:
        importances = permutation_importance(predictor, dataset.data.loc[embedding.nn_ind,dataset.features], 
                                             np.argmax(np.array(dataset.data.loc[embedding.nn_ind,'prob'].tolist()), axis=1),
                                             n_repeats=5, random_state=0).importances_mean
        if np.any((importances != 0)):
            importances /= np.sum(importances)
        df = pd.DataFrame((100*importances).round(0), columns=["Permutation Importance"], index=dataset.features)
    else:
        df = pd.DataFrame(src.data["Permutation Importance"], columns=["Permutation Importance"], index=dataset.features)
    df["Reduction Influence"] = (100*red_influence).round(0)
        
    highest = df.select_dtypes(exclude=['object']).max().max()
    df["color_red"] = [rgb_to_hex(lin_cm(i/highest)) for i in df["Reduction Influence"].tolist()]
    df["color_imp"] = [rgb_to_hex(lin_cm(i/highest)) for i in df["Permutation Importance"].tolist()]
    src.data = df
    
    # update table
    row_height = (params.total_height-8)//len(dataset.features)
    template_red=f'''
    <div style="background:<%=color_red%>; 
        height: {row_height}px;
        width: {params.tbl_width//2}px;
        text-align: center;
        vertical-align: middle;
        line-height: {row_height}px;
        top: -1px;
        left: -4px;
        position: relative;">
    <%= value+"%" %></div>
    '''
    template_imp=f'''
    <div style="background:<%=color_imp%>; 
        height: {row_height}px;
        width: {params.tbl_width//2}px;
        text-align: center;
        vertical-align: middle;
        line-height: {row_height}px;
        top: -1px;
        left: -4px;
        position: relative;">
    <%= value+"%" %></div>
    '''
    columns = [
        TableColumn(field='Reduction Influence', title='Emb.Imp.', 
                    formatter=HTMLTemplateFormatter(template=template_red)),
        TableColumn(field='Permutation Importance', title='Mod.Imp.', 
                    formatter=HTMLTemplateFormatter(template=template_imp))]
    tbl.columns = columns
    tbl.row_height = row_height
    
def update_df(tbl, dataset, params):
    template_prob=f'''
    <div style="background:<%=color%>; 
        height: 27px;
        line-height: 25px;
        top: -1px;
        position: relative;">
    <%= value.toFixed(2) %></div>
    '''
    columns = [TableColumn(field="maxprob", title="Model Pred.",
               formatter=HTMLTemplateFormatter(template=template_prob))]
    if dataset.name in ["u1", "u2", "u3", "u4", "u-classes"]:
        columns += [TableColumn(field="Name")]
    columns += [TableColumn(field=f) for f in dataset.features]
    tbl.columns = columns
    

# compute conditional expectation in bounds for all classes
# predict should expect a single instance and 
#         return either a single prediction probability or an array thereof
def partial_dependence(predict, x, bounds, categories, features, res=20):
    pds_x = []
    pds_y = []
    
    for f in range(x.size):
        # init
        ref = x.copy()
        if categories[f] is not None:
            pd_x = []
            pd_y = []
            for c in categories[f]:
                ref[f] = c
                # predict test instance
                ref_pred = predict(pd.DataFrame(ref, index=features).T)[0]
                # save
                pd_x.append(ref[f])
                pd_y.append(ref_pred)
        else:
            # create test instances
            arr = np.full((res,len(ref)),ref)
            arr[:,f] = [bounds[0][f] + (i / (res-1)) * (bounds[1][f] - bounds[0][f]) for i in range(res)]
            # insert prediction at x to stay consistent over features
            arr = np.insert(arr, bisect(arr[:,f], x[f]), x, axis=0)
            # predict instances
            ref_pred = predict(pd.DataFrame(arr, columns=features))
            # save
            pd_x = arr[:,f]
            pd_y = ref_pred
        # save
        pds_x.append(pd_x)
        pds_y.append(pd_y)
    
    return pds_x, pds_y

def update_horizons( plots, dataset, params ):
    return
    for f,p in zip(dataset.features, plots):
        for i in range(len(dataset.classes)):
            if f in dataset.selected_features:
                p.object.select('a'+str(i*2)).glyph.fill_color = params.palette[int(i*5+5*0.3)]
                p.object.select('a'+str(i*2+1)).glyph.fill_color = params.palette[int(i*5+5*0.7)]
            else:
                p.object.select('a'+str(i*2)).glyph.fill_color = 'lightgray'
                p.object.select('a'+str(i*2+1)).glyph.fill_color = 'gray'
            

def plot_horizons( predict_fn, dataset, params, pt_source, cf_source):
    plots = pn.Column()
    
    pds_x, pds_y = partial_dependence(predict_fn, dataset.point[dataset.features].to_numpy(), 
                                      dataset.bounds, dataset.categories, dataset.features, params.pdp_res)
    df_pdp = pd.DataFrame({'x': pds_x, 'y': pds_y}, index=dataset.features)

    for i,f in enumerate(df_pdp.index):
        # handle categorical 'horizons' as stacked bar charts
        if dataset.categories[i] is not None:
            p = figure(height=int(params.total_height/len(df_pdp)), width=params.hor_width, 
                   y_range=(0,0.25), x_range=(df_pdp.loc[f,'x'][0]-0.5,df_pdp.loc[f,'x'][-1]+0.5), 
                   tools=[], x_axis_location='above', min_border_left=0, min_border_right=0)
            max_x = df_pdp.loc[f,'x'][-1] + 1
            p.xaxis.ticker = FixedTicker(ticks=[i for i in range(int(max_x))])
            
            for i in range(len(df_pdp.loc[f,'y'][0])):
                p.vbar(x=df_pdp.loc[f,'x'], top=[v[i]-0.5 for v in df_pdp.loc[f,'y']], width=1-(max_x*0.003),
                        color=params.palette[int(i*5+2)], name='a'+str(i*2))
                p.vbar(x=df_pdp.loc[f,'x'], top=[v[i]-0.75 for v in df_pdp.loc[f,'y']], width=1-(max_x*0.003),
                        color=params.palette[int(i*5+4)], name='a'+str(i*2+1))
        else:
        # plot horizons
            p = figure(height=int(params.total_height/len(df_pdp)), width=params.hor_width, 
                   y_range=(0,0.25), x_range=(df_pdp.loc[f,'x'][0],df_pdp.loc[f,'x'][-1]), 
                   tools=[], x_axis_location='above', min_border_left=0, min_border_right=0)
            
            for i in range(len(df_pdp.loc[f,'y'][0])):
                p.varea(x=df_pdp.loc[f,'x'], y1=[0]*len(df_pdp.loc[f,'x']), y2=[v[i]-0.5 for v in df_pdp.loc[f,'y']], 
                        color=params.palette[int(i*5+2)], name='a'+str(i*2))
                p.varea(x=df_pdp.loc[f,'x'], y1=[0]*len(df_pdp.loc[f,'x']), y2=[v[i]-0.75 for v in df_pdp.loc[f,'y']], 
                        color=params.palette[int(i*5+4)], name='a'+str(i*2+1))

        for color, src, name in zip(['#444444','black'], [cf_source,pt_source], ['cf','pt']):
            p.line(x=f, y='y', color=color, line_width=2, name=name+'_line', source=src.line)
            p.text(x=f, y='y', color=color, text=f+"_label", name=name+'_text', text_font_size='8pt', x_offset=f+'_x_offset', y_offset=11, text_align=f+'_align', source=src.text)
        
        p.ygrid.visible=False
        p.xgrid.visible=False
        p.yaxis.visible=False
        p.xaxis.fixed_location=0.0
        axis_color = "black"
        p.xaxis.axis_line_color       =axis_color
        p.xaxis.major_tick_line_color =axis_color
        p.xaxis.minor_tick_line_color =axis_color
        p.xaxis.major_label_text_color=axis_color
        p.xaxis.major_label_text_font_size="7pt"
        p.xaxis.major_label_standoff=0
        p.xaxis.major_tick_out=4
        p.xaxis.minor_tick_out=2
        p.xaxis.major_tick_in=4
        p.xaxis.minor_tick_in=2
        p.toolbar_location = None

        plots.append(p)
    
    return plots


def update_horizon_lines(horizons, features, point, name="pt"):
    for i,f in enumerate(features):
        horizons[i].object.select(name=name).data_source.data['x'] = [point[i], point[i]]
        horizons[i].object.select(name=name+"_label").data_source.data['x'] = [point[i]]
        horizons[i].object.select(name=name+"_label").data_source.data['text'] = ['{:3.1f}'.format(point[i])]

        
def update_embedding(p, params, dataset, embedding, predictor, average, plot_range=None):
    '''Adjust the embedding view to new PCA.'''
    
    # update plot datasource
    dataset.datasource.stream(dataset.data, rollover=dataset.data.shape[0])
    
    # set up background image bound
    if plot_range is None:
        x_lef = min(dataset.data.loc[embedding.nn_ind,'x'])
        x_rig = max(dataset.data.loc[embedding.nn_ind,'x'])
        y_bot = min(dataset.data.loc[embedding.nn_ind,'y'])
        y_top = max(dataset.data.loc[embedding.nn_ind,'y'])
        bx = x_rig-x_lef
        by = y_top-y_bot
        if bx == 0 or by == 0 :
            print('ERROR: plot ranges collapsed to 1D', x_lef, x_rig, y_bot, y_top, embedding.nn_ind)
        x_min = x_lef - (max(bx,by)/bx -1) * bx/2
        x_max = x_rig + (max(bx,by)/bx -1) * bx/2
        y_min = y_bot - (max(bx,by)/by -1) * by/2
        y_max = y_top + (max(bx,by)/by -1) * by/2
    else:
        # check if double call based on plot range change listener
        if p.select('image').data_source.data['x'] == [plot_range[0]]:
            return
        x_min = plot_range[0]
        x_max = plot_range[1]
        y_min = plot_range[2]
        y_max = plot_range[3]
    bounds_range = max(x_max-x_min, y_max-y_min)
    
    N = params.red_res
    def compute_and_predict_grid(dataset, inv_tf_fn, predict_fn, N, x_min, x_max, y_min, y_max):
        px = np.linspace(x_min, x_max, num=N)
        py = np.linspace(y_min, y_max, num=N)
        
        xx, yy = np.meshgrid(px,py)

#         tic = time.perf_counter()
        invp = inv_tf_fn(np.c_[xx.ravel(), yy.ravel()])
#         tac = time.perf_counter()

        for i,f in enumerate(dataset.features):
            # insert fixed features
            if f not in dataset.selected_features:
                invp = np.insert(invp, i, [dataset.point[i]]*len(invp), axis=1)
                
            # fix categorical values to integers
            if dataset.categories[i] is not None:
                invp[:,i] = np.around(invp[:,i])
        
        probs = predict_fn(pd.DataFrame(invp, columns=dataset.features))
#         toc = time.perf_counter()
#         print("Compute inv_points:",tac-tic,"s")
#         print("With Predict:",toc-tic,"s")
        return probs, invp
    
    # iterate zoom until a decision boundary is within the window
    for i in range(1,6):    
        probs, invp = compute_and_predict_grid(dataset, embedding.emb.inverse_transform, predictor.predict_proba, N, x_min, x_max, y_min, y_max)
        visible_class_count = np.count_nonzero(np.amax(probs, axis=0) > 0.5)
        if visible_class_count >= 2 or plot_range is not None: 
            break
        x_min -= (i/2.5)*bounds_range
        x_max += (i/2.5)*bounds_range
        y_min -= (i/2.5)*bounds_range
        y_max += (i/2.5)*bounds_range
    
    img_src = (np.clip(np.max(probs, axis=1),0.501,0.999) - 0.5) * 2 + np.argmax(probs, axis=1) 
    
    source = p.select('image').data_source
    source.data['image'] = [img_src.reshape((N,N))]
    source.data['x'] = [x_min]
    source.data['y'] = [y_min]
    source.data['dw'] = [x_max-x_min]
    source.data['dh'] = [y_max-y_min]
    
    p.select('image').glyph.color_mapper=LinearColorMapper(palette=params.palette, low=0, high=params.num_colors)
    
    for i, f in enumerate(dataset.features):
        source.data[f] = [invp[:,i].reshape((N,N))]
    
    # update plot ranges (triggers plot range listener)
    if plot_range is None:
        p.x_range.update(start=x_min, end=x_max)
        p.y_range.update(start=y_min, end=y_max)
    
    # update axes
    axis_x = []
    axis_y = []
    point = dataset.point[dataset.selected_features].to_frame().T
    point_xy = embedding.emb.transform(point)[0]
    s = find_axes_scaling(point_xy, embedding.emb['pca'].components_, [x_min, x_max, y_min, y_max])
    axis_x = embedding.emb['pca'].components_[0]*s + point_xy[0]
    axis_y = embedding.emb['pca'].components_[1]*s + point_xy[1]
    p.select('axes').data_source.data = dict(xs=[[point_xy[0],v] for v in axis_x], ys=[[point_xy[1],v] for v in axis_y])
    #p.select('axes').data_source.data = dict(xs=[[0,v] for v in embedding.emb['pca'].components_[0]], ys=[[0,v] for v in embedding.emb['pca'].components_[1]])
    text_y = axis_y
    # diab 667
#    text_y[7] = text_y[7]+0.05
#    text_y[1] = text_y[1]-0.35
#    text_y[6] = text_y[6]-0.2
    # teaser
#     text_y[1] = text_y[1]+0.05
#     text_y[3] = text_y[3]-0.3
#     text_y[6] = text_y[6]-0.3
#     text_y[8] = text_y[8]-0.3
    p.select('text').data_source.data = dict(x=axis_x, y=text_y, text=dataset.selected_features)
    
    # update hover tools
    if params.tooltip_pt:
        p.select(name='hover_emb')[0].tooltips = None
        if dataset.name in ["u1", "u2", "u3", "u4", "u-classes"]:
            p.select(name='hover_pt')[0].tooltips = [("Name", "@Name"), ('prob', "@maxprob{0%}")] + [(f, "@{"+f+"}") for f in dataset.features]
        else:
            p.select(name='hover_pt')[0].tooltips = [("id", "$index"), ('prob', "@maxprob{0%}")] + [(f, "@{"+f+"}") for f in dataset.features]
    else:
        p.select(name='hover_pt')[0].tooltips = None
        p.select(name='hover_emb')[0].tooltips = [(f, "@{"+f+"}{0.0}") for f in dataset.features]
    
def embedding_view(params, dataset, embedding, predictor, avg, cf_source):
    '''Create the embedding view.'''
    p = figure(height=params.emb_width-2, width=params.emb_width+30, 
               tools="pan,tap,box_select,lasso_select", 
               x_range=(-1,1), y_range=(-1,1), min_border=0)
    
    p.add_tools(HoverTool(names=["scatter"], name='hover_pt'),
                HoverTool(names=["image"], name='hover_emb'),
                WheelZoomTool(zoom_on_axis=False))
    
    p.cross(source=cf_source, x='x', y='y', color='#444444', size=12, line_width=3, name='cf', 
             nonselection_fill_alpha=1, nonselection_line_alpha=1)
    mapper = linear_cmap(field_name='sat_color', palette=params.palette, low=0, high=params.num_colors)
    
    p.scatter(source=dataset.datasource, x='x', y='y', color='sat_color', line_color='line_color',
              size='size', name="scatter", nonselection_fill_alpha=0.0, nonselection_line_alpha=0.0)
    
    p.scatter(source=dataset.datasource, x='x', y='y', fill_color='target_color',
              line_color='target_color', size='size', name="wrong_scatter",
              nonselection_fill_alpha=0.0, nonselection_line_alpha=0.0, marker='cross',
              view=CDSView(source=dataset.datasource, filters=[dataset.wrong_filter]))

    p.image(image=[[]], color_mapper=LinearColorMapper(palette=params.palette, low=0, high=params.num_colors), name='image',
                 x=[0], y=[0], dw=[0], dh=[0], level='underlay')
    p.multi_line([], [], color='#555555', name='axes', nonselection_line_alpha=1)
    p.text(x=[], y=[], text=[], color='#555555', text_align="center", name='text', nonselection_alpha=1)

    update_embedding(p, params, dataset, embedding, predictor, avg)

    # minimalistic style
    p.axis.visible = False
    p.xgrid.visible = False
    p.ygrid.visible = False
#     p.outline_line_color = None
    p.toolbar_location = "right"
    p.toolbar.logo = None
    p.toolbar.active_scroll = p.select_one(WheelZoomTool)
    
    return pn.pane.Bokeh(p)

def colorbar_view(params, dataset):
    if dataset.name in ["u1", "u2", "u3", "u4"]:
        bounds, hist, edges = dataset.histo_data
        colors = [params.color_list[classify(e, bounds)] for e in edges[:-1]]
        p = figure(height=150, width=200, toolbar_location=None, min_border=5, y_axis_location='right')
        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color=colors, line_color="white")
        p.xaxis.axis_label=dataset.name
        p.yaxis.axis_label='#solutes'
    else:
        p = figure(height=150, width=200, toolbar_location=None, min_border=5, x_range=[50,100], y_range=dataset.classes,
                   y_axis_location='right')
        p.rect(x=len(dataset.classes)*[55+10*i for i in range(5)], y=np.repeat(dataset.classes,5), 
               width=10, height=1, color=params.palette[:5*len(dataset.classes)])
        p.axis.axis_line_color = None
        
    p.xaxis.minor_tick_line_color = None
    p.yaxis.minor_tick_line_color = None
    p.yaxis.major_label_standoff = -1
    p.xaxis.major_label_standoff = -1
    p.yaxis.axis_label_standoff = -2
    p.xaxis.axis_label_standoff = -3
    p.xaxis.major_tick_in = 0
    p.yaxis.major_tick_in = 0

    return pn.pane.Bokeh(p)


def rgb_to_hex(c):
    h = '#'
    for i in c:
        h = h + '{0:02X}'.format(int(i*255))
    return h

from umap import UMAP
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from bokeh.models import Range1d
from sklearn.manifold import TSNE, MDS

def update_non_linear_view(p, dataset, predictor, method="MDS", neighbors=20, sel_feats_only=False):
    # reduce data with non-linear DR
    if method == "UMAP":
        nl_pip = Pipeline([('scaler', StandardScaler()), ('reducer', UMAP(n_neighbors=neighbors, random_state=42))])
    elif method == "t-SNE":
        nl_pip = Pipeline([('scaler', StandardScaler()), ('reducer', TSNE(perplexity=neighbors, random_state=42))])
    elif method == "MDS":
        nl_pip = Pipeline([('scaler', StandardScaler()), ('reducer', MDS(n_components=2, random_state=42))])
    if sel_feats_only:
        reduced = nl_pip.fit_transform(dataset.data[dataset.selected_features])
    else:
        reduced = nl_pip.fit_transform(dataset.data[dataset.features])
    
    dataset.data['x1'] = reduced[:,0]
    dataset.data['y1'] = reduced[:,1]
    
    # update plot bounds
    bx = max(dataset.data['x1'])-min(dataset.data['x1'])
    by = max(dataset.data['y1'])-min(dataset.data['y1'])
    x_min = min(dataset.data['x1'])-0.1*bx
    x_max = max(dataset.data['x1'])+0.1*bx
    y_min = min(dataset.data['y1'])-0.1*by
    y_max = max(dataset.data['y1'])+0.1*by
    p.x_range.update(start=x_min, end=x_max)
    p.y_range.update(start=y_min, end=y_max)
    
    # update plot datasource
    dataset.datasource.stream(dataset.data, rollover=dataset.data.shape[0])


def non_linear_view(params, dataset, predictor):
    p = figure(height=params.bot_height, width=params.bot_height+30, tools="pan,tap,box_select,lasso_select", x_range=(-1,1), y_range=(-1,1))
    
    if dataset.name in ["u1", "u2", "u3", "u4", "u-classes"]:
        p.add_tools(HoverTool(names=["scatter"], name='hover', tooltips = [("Name", "@Name")]), WheelZoomTool(zoom_on_axis=False))
    else:
        p.add_tools(HoverTool(names=["scatter"], name='hover', tooltips = [("id", "$index")]), WheelZoomTool(zoom_on_axis=False))
    
    update_non_linear_view(p, dataset, predictor)
    
    mapper = linear_cmap(field_name='sat_color', palette=params.palette, low=0, high=params.num_colors)
    p.scatter(source=dataset.datasource, x='x1', y='y1', color='sat_color', line_color='line_color', 
              size='size', name="scatter", nonselection_fill_alpha=0.1, nonselection_line_alpha=0.1)
    
    p.scatter(source=dataset.datasource, x='x1', y='y1', fill_color='target_color',
              line_color='target_color', size='size', name="wrong_scatter",
              nonselection_fill_alpha=0.0, nonselection_line_alpha=0.0, marker='cross',
              view=CDSView(source=dataset.datasource, filters=[dataset.wrong_filter]))
    
    
    # minimalistic style
    p.axis.visible = False
    p.xgrid.visible = False
    p.ygrid.visible = False
    p.toolbar_location = "right"
    p.toolbar.logo = None
    p.toolbar.active_scroll = p.select_one(WheelZoomTool)
    
    return pn.pane.Bokeh(p)

def find_axes_scaling(origin, components, bounds):
    
    def line(p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0]*p2[1] - p2[0]*p1[1])
        return A, B, -C

    def intersect(L1, L2):
        D  = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if D != 0:
            x = Dx / D
            y = Dy / D
            return x,y
        else:
            return False
        
    # extend axis to boundary
    def directed_boundary_intersection(o, d, start, end):
        # create lines
        axis = line(o, d)
        bound = line(start, end)
        # intersect axis and boundary
        i = intersect(bound, axis)
        if not i:
            return False
        # check if within bound segment
        z = np.nonzero(end-start)[0][0]
        if i[z] < start[z] or end[z] < i[z]:
            return False
        # correct direction
        if (np.greater((i-o).round(3), 0) != np.greater((d-o).round(3), 0)).any():
            return False
        return i
    
    s = np.inf
    # intersect each axis with each embedding-bound-side
    for i in range(components.shape[1]):
        # bot, left, top, right
        inds = [(0,1),(0,2),(2,3),(1,3)]
        corners = np.array([[bounds[0],bounds[2]],[bounds[1],bounds[2]],[bounds[0],bounds[3]],[bounds[1],bounds[3]]])
        for x in inds:
            ext = directed_boundary_intersection(origin, origin+components[:,i], corners[x[0]], corners[x[1]])
            if ext != False: break
        if ext is False: # origin was not within corners
#             print("error computing axes scaling.", origin, components[:,i], bounds) 
            continue
        s_ = (np.linalg.norm(ext-origin) / np.linalg.norm(components[:,i])) * 0.9
        if s_ < s: s = s_
    return s
        