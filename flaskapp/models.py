from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
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
from scipy.stats import t
import time


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
        
        self.variables = self.X.columns.tolist()
        mgwr_selector = Sel_BW(coords, Zy, ZX, multi=True)
        mgwr_bw = mgwr_selector.search()
        self.model = MGWR(coords, Zy, ZX, mgwr_selector)
        self.gdf = gpd.read_file('static/shapefiles/Rajasthan_Blocks.shp')

    def mgwr_coefficient_plot(self,coefficients):
        coefficients_df = pd.DataFrame(coefficients[:,1:], index=self.merged.index, columns=self.X.columns)
        merged_gdf = pd.merge(coefficients_df, self.gdf, left_index=True, right_index=True)
        merged_gdf = gpd.GeoDataFrame(merged_gdf, geometry='geometry')

 
        fig, axes = plt.subplots(int(np.ceil(len(self.variables) / 3)), ncols=3, figsize=(12,8))

        for i, var in enumerate(self.variables):
            row, col = divmod(i, 3)
            ax = axes[row, col]
            merged_gdf.plot(column=var, cmap='coolwarm', linewidth=0.05, scheme='FisherJenks', k=5, legend=True, legend_kwds={'fontsize': 8, 'bbox_to_anchor': (1.01, 1.01)}, ax=ax)
            ax.set_title(f'{var} plot', fontsize=12)
            ax.set_axis_off()
        
        plt.tight_layout()
        file_path=f'static/plots/coefficient_plot_{int(time.time())}.png'
        fig.savefig(file_path)
        plt.close()
        return file_path
    
    # def mgwr_pvalue_plot(self, mgwr_results):
    #     print(mgwr_results.pvalues)
    #     p_values_df = pd.DataFrame(mgwr_results.pvalues, index=self.merged.index, columns=self.X.columns)
    #     merged_gdf = pd.merge(p_values_df, self.gdf, left_index=True, right_index=True)
    #     merged_gdf = gpd.GeoDataFrame(merged_gdf, geometry='geometry')

    #     fig, axes = plt.subplots(int(np.ceil(len(self.variables) / 3)), ncols=3, figsize=(18,6))

    #     for i, var in enumerate(self.variables):
    #         row, col = divmod(i, 3)
    #         ax = axes[row, col]
    #         merged_gdf.plot(column=var, cmap='coolwarm', linewidth=0.05, scheme='FisherJenks', k=7, legend=True, legend_kwds={'bbox_to_anchor': (1.10, 0.96)}, ax=ax)
    #         ax.set_title(f'{var} plot', fontsize=12)
    #         ax.set_axis_off()
        
    #     plt.tight_layout()
    #     file_path='static/plots/p_value_plot.png'
    #     fig.savefig(file_path)
    #     plt.close()
    #     return file_path


    # this is currently t-value plot as kb requested
    def mgwr_coefft_plot(self, mgwr_results, coefficients):
        coefficients_df = pd.DataFrame(coefficients[:,1:], index=self.merged.index, columns=self.X.columns)
        merged_gdf = pd.merge(coefficients_df, self.gdf, left_index=True, right_index=True)
        merged_gdf = gpd.GeoDataFrame(merged_gdf, geometry='geometry')

        mgwr_filtered_t = mgwr_results.filter_tvals(alpha = 0.05)
        n = self.X.shape[0]
        # p_values = np.array([[2 * (1 - t.cdf(abs(t_val), n - 2)) for t_val in t_val_row] for t_val_row in t_values])
        mgwr_filtered_series = pd.DataFrame(mgwr_filtered_t[:, 1])

        fig, axes = plt.subplots(int(np.ceil(len(self.variables) / 3)), ncols=3, figsize=(12,8))

        for i, var in enumerate(self.variables):
            row, col = divmod(i, 3)
            ax = axes[row, col]
            merged_gdf.plot(column=var, cmap='coolwarm', linewidth=0.05, scheme='FisherJenks', k=5, legend=True, legend_kwds={'fontsize': 8, 'bbox_to_anchor': (1.01, 1.01)}, ax=ax)
            merged_gdf[mgwr_filtered_t[:,1] == 0].plot(color='white', linewidth=0.05, edgecolor='black', ax=ax)
            ax.set_title(f'{var} plot', fontsize=12)
            ax.set_axis_off()
        
        plt.tight_layout()
        file_path=f'static/plots/filetered_coefficient.png'
        fig.savefig(file_path)
        plt.close()
        return file_path
    
    def mgwr_multicollinearity_test(self,mgwr_results):
        mgwrCN, mgwrVDP = mgwr_results.local_collinearity()
        self.gdf['mgwr_CN'] = mgwrCN

        fig, ax = plt.subplots(figsize=(6, 6))
        self.gdf.plot(column='mgwr_CN', cmap = 'coolwarm', linewidth=0.01, scheme = 'FisherJenks', k=5, legend=True, legend_kwds={'bbox_to_anchor':(1.10, 0.96)},  ax=ax)
        ax.set_title('Local multicollinearity (CN > 30)?', fontsize=12)
        ax.axis("off")
        #plt.savefig('myMap.png',dpi=150, bbox_inches='tight')
        plt.show()
    

    # def mgwr_r2_plot(self, models.R2):
    #     r2 = pd.DataFrame(R2, index=self.merged.index, columns=self.X.columns)

    #     merged_gdf = pd.merge(p_values_df, self.gdf, left_index=True, right_index=True)
    #     merged_gdf = gpd.GeoDataFrame(merged_gdf, geometry='geometry')

    #     fig, axes = plt.subplots(int(np.ceil(len(variables) / 3)), ncols=3, figsize=(18,6))

    #     for i, var in enumerate(self.X.columns):
    #         row, col = divmod(i, 3)
    #         ax = axes[row, col]
    #         merged_gdf.plot(column=var, cmap='coolwarm', linewidth=0.05, scheme='FisherJenks', k=7, legend=True, legend_kwds={'bbox_to_anchor': (1.10, 0.96)}, ax=ax)
    #         ax.set_title(f'{var} plot', fontsize=12)
    #         ax.set_axis_off()
        
    #     plt.tight_layout()
    #     file_path='static/plots/local_r2_plot.png'
    #     fig.savefig(file_path)
    #     plt.close()
    #     return file_path

    def run(self): 
        model = self.model.fit()
        summary = model.summary()
        mgwr_coefficient_plot = self.mgwr_coefficient_plot(model.params)
        # mgwr_pvalue_plot = self.mgwr_pvalue_plot(model)
        mgwr_coefft_plot = self.mgwr_coefft_plot(model, model.params)
        # mgwr_localr_plot = self.mgwr_localr_plot(model)
        
        # mgwr_multicollinearity_test = self.mgwr_multicollinearity_test(model)
        # mgwr_r2_plot = self.mgwr_r2_plot(model.R2)
        return mgwr_coefficient_plot, mgwr_coefft_plot,summary

    
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

class DecisionTreeModel(Model): 
    def __init__(self, outcome, geo=False): 
        super().__init__(outcome, geo)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model = DecisionTreeRegressor(criterion="friedman_mse", max_depth=5, random_state=42)
        self.plot_dir = 'static/plots'

        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

    def save_feature_importances_plot(self):

        importances = self.model.feature_importances_
        indices = np.argsort(importances)
        features = self.X.columns
        
        plt.figure(figsize=(7, 5))
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.title('Feature Importances')
        
        plot_path = os.path.join(self.plot_dir, 'feature_importances.png')
        plt.savefig(plot_path)
        plt.close()
        return plot_path

    def run(self):
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        feature_importances_plot = self.save_feature_importances_plot()
        return r2, feature_importances_plot
