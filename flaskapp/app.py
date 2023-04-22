from distutils.log import debug
from fileinput import filename
import pandas as pd
from flask import *
import os
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import models
import seaborn as sns
import matplotlib.pyplot as plt

UPLOAD_FOLDER = 'static/uploads'
 
# Define allowed files
ALLOWED_EXTENSIONS = {'csv'}
 
app = Flask(__name__)
 
# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
app.secret_key = 'This is your secret key to utilize session in Flask'
 
@app.route('/index')
@app.route('/')
def index():
  return render_template("index.html")

@app.route('/howto')
def howto():
  return render_template("howto.html")

@app.route('/about')
def about():
  return render_template("about.html")

@app.route('/results')
def results():
  return render_template("results.html")

@app.route('/mgwr', methods=['GET', 'POST'])
def runMGWR():
  if request.method == 'POST':
    # upload file flask
    f = request.files.get('file')

    # Extracting uploaded file name
    data_filename = secure_filename(f.filename)

    f.save(os.path.join(app.config['UPLOAD_FOLDER'],
                        data_filename))

    session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
    # Uploaded File Path
    data_file_path = session.get('uploaded_data_file_path', None)
    # read csv
    uploaded_df = pd.read_csv(data_file_path,
                            encoding='unicode_escape')
    
    # Converting to html Table
    uploaded_df_html = uploaded_df.to_html()
    # merged df
    outcome = pd.read_csv(data_file_path)
    
    # X is all the features and y is the output from the csv that user input
    model = models.MGWRModel(outcome)
    r2 = model.run()
    return render_template('show.html',
                        data_var=uploaded_df_html, r2=r2)

  return render_template("model.html",name="mgwr")

@app.route('/randomforest', methods=['GET', 'POST'])
def runRandomForestTree():
  if request.method == 'POST':
    # upload file flask
    f = request.files.get('file')

    # Extracting uploaded file name
    data_filename = secure_filename(f.filename)

    f.save(os.path.join(app.config['UPLOAD_FOLDER'],
                        data_filename))

    session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
    # Uploaded File Path
    data_file_path = session.get('uploaded_data_file_path', None)
    # read csv
    uploaded_df = pd.read_csv(data_file_path,
                            encoding='unicode_escape')
    
    # Converting to html Table
    uploaded_df_html = uploaded_df.to_html()
    # merged df
    outcome = pd.read_csv(data_file_path)
    
    # X is all the features and y is the output from the csv that user input
    model = models.RandomForestModel(outcome)
    r2, proximity_plot_path, feature_importances_plot_path = model.run()
    return render_template('show.html', data_var=uploaded_df_html, r2=r2, proximity_plot=proximity_plot_path, feature_importances_plot=feature_importances_plot_path)

  return render_template("model.html",name="randomforest")


@app.route('/multiplelinearregression', methods=['GET', 'POST'])
def runMLR():
  if request.method == 'POST':
    # upload file flask
    f = request.files.get('file')

    # Extracting uploaded file name
    data_filename = secure_filename(f.filename)

    f.save(os.path.join(app.config['UPLOAD_FOLDER'],
                        data_filename))

    session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
    # Uploaded File Path
    data_file_path = session.get('uploaded_data_file_path', None)
    # read csv
    uploaded_df = pd.read_csv(data_file_path,
                            encoding='unicode_escape')
    
    # Converting to html Table
    uploaded_df_html = uploaded_df.to_html()
    # merged df
    outcome = pd.read_csv(data_file_path)
    
    # X is all the features and y is the output from the csv that user input
    model = models.MultipleLinearRegression(outcome)
    r2, mlr_coef, res_plot, summary = model.run()
    return render_template('show.html',
                        data_var=uploaded_df_html, r2=r2, multiple_linear_regression_coefficients=mlr_coef, residual_plot=res_plot, summary=summary)

  return render_template("model.html", name="multiplelinearregression")


@app.route('/ridgeregression', methods=['GET', 'POST'])
def runRidge():
  if request.method == 'POST':
    # upload file flask
    f = request.files.get('file')

    # Extracting uploaded file name
    data_filename = secure_filename(f.filename)

    f.save(os.path.join(app.config['UPLOAD_FOLDER'],
                        data_filename))

    session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
    # Uploaded File Path
    data_file_path = session.get('uploaded_data_file_path', None)
    # read csv
    uploaded_df = pd.read_csv(data_file_path,
                            encoding='unicode_escape')
    
    # Converting to html Table
    uploaded_df_html = uploaded_df.to_html()
    # merged df
    outcome = pd.read_csv(data_file_path)
    
    # X is all the features and y is the output from the csv that user input
    model = models.RidgeRegression(outcome)
    r2, ridge_coef, alpha_plot, res_plot = model.run()
    return render_template('show.html',
                        data_var=uploaded_df_html, r2=r2, ridge_regression_coefficients=ridge_coef, alpha_plot=alpha_plot, residual_plot=res_plot)


  return render_template("model.html",name="ridgeregression")


@app.route('/lassoregression', methods=['GET', 'POST'])
def runLasso():
  if request.method == 'POST':
    # upload file flask
    f = request.files.get('file')

    # Extracting uploaded file name
    data_filename = secure_filename(f.filename)

    f.save(os.path.join(app.config['UPLOAD_FOLDER'],
                        data_filename))

    session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
    # Uploaded File Path
    data_file_path = session.get('uploaded_data_file_path', None)
    # read csv
    uploaded_df = pd.read_csv(data_file_path,
                            encoding='unicode_escape')
    
    # Converting to html Table
    uploaded_df_html = uploaded_df.to_html()
    # merged df
    outcome = pd.read_csv(data_file_path)
    
    # X is all the features and y is the output from the csv that user input
    model = models.LassoRegression(outcome)
    r2, lasso_coef, alpha_plot, res_plot = model.run()
    return render_template('show.html',
                        data_var=uploaded_df_html, r2=r2, lasso_regression_coefficients=lasso_coef, alpha_plot=alpha_plot, residual_plot=res_plot)

  return render_template("model.html",name="lassoregression")

@app.route('/decisiontree', methods=['GET', 'POST'])
def runDecisionTree():
  if request.method == 'POST':
    # upload file flask
    f = request.files.get('file')

    # Extracting uploaded file name
    data_filename = secure_filename(f.filename)

    f.save(os.path.join(app.config['UPLOAD_FOLDER'],
                        data_filename))

    session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
    # Uploaded File Path
    data_file_path = session.get('uploaded_data_file_path', None)
    # read csv
    uploaded_df = pd.read_csv(data_file_path,
                            encoding='unicode_escape')
    
    # Converting to html Table
    uploaded_df_html = uploaded_df.to_html()
    # merged df
    outcome = pd.read_csv(data_file_path)
    
    # X is all the features and y is the output from the csv that user input
    dt_model = models.DecisionTreeModel(outcome)
    r2, feature_importances_plot = dt_model.run()
    return render_template('show.html', data_var=uploaded_df_html, r2=r2, feature_importances_plot=feature_importances_plot)
    # return render_template('show.html', data_var=uploaded_df_html, r2=r2, proximity_plot=proximity_plot_path, feature_importances_plot=feature_importances_plot_path)
  return render_template("model.html",name="decisiontree")

 
if __name__ == '__main__':
    app.run(debug=True)

