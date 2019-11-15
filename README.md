# ml_test_app
Web-interface for getting statistics after testing machine learning models on unseen data.

XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. 
This tool is developed to perform testing stage of  XGBoost model based on user-suggested settings. 

The tool consists of 2 modules: 

test_app.py – main module, predict scores and calculate statistics. 
calc.py – additional module, contains tools for processing .xml configuration file, creating test dataset. 

Before running tool, be sure, that you have installed Python 3.6 and necessary libraries: 
xgboost (0.90 version), pandas (0.25.0 version), numpy (1.17.2 version), plotly (4.2.0 version)

INPUT FILES:

1).xml configuration file. User should edit configuration file before running tool by specifying working directory full path,  
name of.csv config file located in the working directory, name of the .aux file for test data located in the working directory,
name of the .bin file located in the working directory 

2) .auc  file. File contains a list of pathes to .dat files with data from detectors. 

3) .csv file. File contains a list of parameters from detector. 

4) .bin file with xgboost model. 

All files should be placed in a working directory. 

RUNNING TOOL:
After installing all the necessary software and putting all files in working directory, 
run tool by typing in ubuntu terminal (in working directory): 

python3 main_file.py --confg_name full_path_to_xml_config_file 

OUTPUT FILES:
1) .csv file (model_file_name.csv) with statistics for fixed threshold (in case you are running terminal version)

2)  .html files with ROC-AUC curves.  

All files will be placed in the working directory. 
