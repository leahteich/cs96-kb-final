from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import geopandas as gpd
from mgwr.gwr import GWR, MGWR 
from mgwr.sel_bw import Sel_BW 
from mgwr.utils import shift_colormap, truncate_colormap
import mgwr
from geopandas import GeoDataFrame
from shapely.geometry import Polygon
import os
import seaborn as sns

class Model: 
    def __init__(self, outcome, geo=False): 
        # merge with the outcome and the geographical thing 
        join_cols = ['SUBDISTRICT', 'DISTRICT']
        features = pd.read_csv('static/merged.csv')
        features['DISTRICT'] = features['DISTRICT'].apply(str.upper)

        self.merged = pd.merge(features, outcome[outcome.duplicated(subset=join_cols, keep='first') == False],  on=join_cols)
        self.merged.dropna(inplace=True)
        
        self.merged = gpd.GeoDataFrame(self.merged, geometry=self.polygons_from_custom_xy_string(self.merged["geometry"]))

        self.y = self.merged.iloc[:,-1:]
        self.X = self.merged.iloc[:,4:]
        self.X = self.X.iloc[:,:-1]

    def polygons_from_custom_xy_string(self, df_column):
        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]
        
        def xy_list_from_string(s):
            # 'x y x y ...' -> [[x, y], [x, y], ...]
            s = s.split('((')[1]
            s = s.split('))')[0]
            s = s.replace(',','')
            return list(chunks([float(i) for i in s.split()], 2))
        
        def poly(s):
            """ returns shapely polygon from point list"""
            ps = xy_list_from_string(s)
            return Polygon([[p[0], p[1]] for p in ps])

        polygons = [poly(r) for r in df_column]

        return polygons

class RandomForestModel(Model): 
    def __init__(self, outcome, geo=False): 
        super().__init__(outcome, geo)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=0)
        self.model = RandomForestRegressor(n_estimators=100, random_state=10)
        self.plot_dir = 'static/plots'

        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

    @staticmethod
    def proximity_matrix(model, X, normalize=True):
        terminals = model.apply(X)
        n_trees = terminals.shape[1]
        
        # Initialize an empty matrix
        prox_mat = np.zeros((X.shape[0], X.shape[0]))
        
        # Iterate through all trees and update proximity matrix
        for tree in range(n_trees):
            terminals_tree = terminals[:, tree]
            
            for i in range(X.shape[0]):
                for j in range(i, X.shape[0]):
                    if terminals_tree[i] == terminals_tree[j]:
                        prox_mat[i, j] += 1
                        if i != j:
                            prox_mat[j, i] += 1
                            
        if normalize:
            prox_mat = prox_mat / n_trees

        return prox_mat

    def save_proximity_matrix_plot(self):
        prox_mat = RandomForestModel.proximity_matrix(self.model, self.X_test)
        plt.figure(figsize=(7, 5))
        sns.heatmap(prox_mat, cmap='viridis')
        plt.title("Proximity Matrix Heatmap")
        plot_path = os.path.join(self.plot_dir, 'proximity_matrix.png')
        plt.savefig(plot_path)
        plt.close()
        return plot_path

    def save_feature_importances_plot(self):
        importances = self.model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)
        forest_importances = pd.Series(importances, index=self.X.columns)

        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()

        plot_path = os.path.join(self.plot_dir, 'feature_importances.png')
        plt.savefig(plot_path)
        plt.close()
        return plot_path

    def run(self):
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        proximity_plot_path = self.save_proximity_matrix_plot()
        feature_importances_plot_path = self.save_feature_importances_plot()
        return r2, proximity_plot_path, feature_importances_plot_path

class MGWRModel(Model):
    def __init__(self, outcome, geo=True): 
        super().__init__(outcome, geo)
        y = self.y.values.reshape((-1, 1))
        X = self.X.values 
        Zy = (y - y.mean(axis=0)) / y.std(axis=0)
        ZX = (X - X.mean(axis=0)) / X.std(axis=0)

        self.merged['centroids'] = self.merged.centroid
        self.merged['X'] = self.merged['centroids'].x 
        self.merged['Y'] = self.merged['centroids'].y 
        u, v = self.merged['X'], self.merged['Y']
        coords = list(zip(u, v))

        mgwr_selector = Sel_BW(coords, Zy, ZX, multi=True)
        mgwr_bw = mgwr_selector.search()
        self.model = MGWR(coords, Zy, ZX, mgwr_selector)

    def run(self): 
        result = self.model.fit()
        return result.summary()
    
# Multiple linear regression class that returns R^2 score and a plot of coefficients
class MultipleLinearRegression(Model):
    def __init__(self, outcome, geo=False): 
        super().__init__(outcome, geo)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=0)
        self.model = LinearRegression()
        self.X_train_sm = sm.add_constant(self.X_train)
        self.model_sm = sm.OLS(self.y_train, self.X_train_sm)

    def save_coefficients_plot(self):
        features = list(self.X_train.columns.values)
        fig = plt.figure(figsize=(7, 5))
        plt.plot(features, self.model.coef_.flatten(), linestyle='none', marker='o', markersize=4, color='blue', zorder=7)
        plt.title("Linear Regression Coefficients")
        plt.xticks(rotation=90)
        plt.margins(0.2)
        plt.subplots_adjust(bottom=0.25)
        fig_path = 'static/plots/multiple_linear_regression_coefficients.png'
        fig.savefig(fig_path)
        plt.close()
        return fig_path

    def save_residual_plot(self):
        y_pred = self.model.predict(self.X_test)
        residuals = self.y_test - y_pred
        fig = plt.figure(figsize=(7, 5))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        fig_path = 'static/plots/residual_plot.png'
        fig.savefig(fig_path)
        plt.close()
        return fig_path

    def run(self): 
        self.model.fit(self.X_train, self.y_train)
        result = self.model_sm.fit()
        r2 = self.model.score(self.X_test, self.y_test)
        coef_plot_path = self.save_coefficients_plot()
        res_plot_path = self.save_residual_plot()
        summary = result.summary()
        return r2, coef_plot_path, res_plot_path, summary
    
class RidgeRegression(Model):
    def __init__(self, outcome, geo=False):
        super().__init__(outcome, geo)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=0)
        self.model = RidgeCV(alphas = [0.0001, 0.001,0.01, 0.1, 1, 5, 10, 20], store_cv_values=True)

    def save_coefficients_plot(self):
        features = list(self.X_train.columns.values)
        fig = plt.figure(figsize=(7, 5))
        plt.plot(features, self.model.coef_.flatten(), linestyle='none', marker='o', markersize=4, color='blue', zorder=7)
        plt.title("Ridge Regression Coefficients")
        plt.xticks(rotation=90)
        plt.margins(0.2)
        plt.subplots_adjust(bottom=0.25)
        fig_path = 'static/plots/ridge_regression_coefficients.png'
        fig.savefig(fig_path)
        plt.close()
        return fig_path

    def save_alpha_plot(self):
        cv_mean = self.model.cv_values_.mean(axis=0).flatten()
        fig = plt.figure(figsize=(7, 5))
        plt.plot(self.model.alphas, cv_mean)
        plt.xscale('log')
        plt.xlabel("Alpha Value")
        plt.ylabel("Mean Squared Error")
        plt.title("Ridge Regression - Alpha Value Selection")
        fig_path = 'static/plots/ridge_alpha_selection.png'
        fig.savefig(fig_path)
        plt.close()
        return fig_path

    def save_residual_plot(self):
        y_pred = self.model.predict(self.X_test)
        residuals = self.y_test - y_pred
        fig = plt.figure(figsize=(7, 5))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        fig_path = 'static/plots/residual_plot.png'
        fig.savefig(fig_path)
        plt.close()
        return fig_path

    def run(self):
        self.model.fit(self.X_train, self.y_train)
        r2 = self.model.score(self.X_test, self.y_test)
        coef_plot_path = self.save_coefficients_plot()
        alpha_plot_path = self.save_alpha_plot()
        res_plot_path = self.save_residual_plot()
        return r2, coef_plot_path, alpha_plot_path, res_plot_path
    
class LassoRegression(Model):
    def __init__(self, outcome, geo=False):
        super().__init__(outcome, geo)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=0)
        self.model = LassoCV(alphas = [0.0001, 0.001,0.01, 0.1, 1, 5, 10, 20])

    def save_coefficients_plot(self):
        features = list(self.X_train.columns.values)
        fig = plt.figure(figsize=(7, 5))
        plt.plot(features, self.model.coef_.flatten(), linestyle='none', marker='o', markersize=4, color='blue', zorder=7)
        plt.title("Lasso Regression Coefficients")
        plt.xticks(rotation=90)
        plt.margins(0.2)
        plt.subplots_adjust(bottom=0.25)
        fig_path = 'static/plots/lasso_regression_coefficients.png'
        fig.savefig(fig_path)
        plt.close()
        return fig_path

    def save_alpha_plot(self):
        cv_mean = self.model.mse_path_.mean(axis=1)
        fig = plt.figure(figsize=(7, 5))
        plt.plot(self.model.alphas_, cv_mean)
        plt.xscale('log')
        plt.xlabel("Alpha Value")
        plt.ylabel("Mean Squared Error")
        plt.title("Lasso Regression - Alpha Value Selection")
        fig_path = 'static/plots/lasso_alpha_selection.png'
        fig.savefig(fig_path)
        plt.close()
        return fig_path

    def save_residual_plot(self):
        y_pred = self.model.predict(self.X_test)
        residuals = self.y_test.values.flatten() - y_pred
        fig = plt.figure(figsize=(7, 5))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        fig_path = 'static/plots/residual_plot.png'
        fig.savefig(fig_path)
        plt.close()
        return fig_path

    def run(self):
        self.model.fit(self.X_train, self.y_train)
        r2 = self.model.score(self.X_test, self.y_test)
        coef_plot_path = self.save_coefficients_plot()
        alpha_plot_path = self.save_alpha_plot()
        res_plot_path = self.save_residual_plot()
        return r2, coef_plot_path, alpha_plot_path, res_plot_path
