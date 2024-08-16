# Health-Data-Analytics
This repository contains a comprehensive Health Data Analytics project utilizing datasets sourced from [Kaggle](https://www.kaggle.com/datasets/muhammadehsan000/healthcare-dataset-2019-2024?resource=download). The primary goal of this project is to analyze and visualize health-related data to uncover valuable insights that can aid healthcare professionals and organizations in making informed decisions. By leveraging the power of Python libraries and Tableau, this project provides an interactive and intuitive dashboard for exploring the data. 
View the interactive dashboard [here](https://public.tableau.com/views/HealthInsightDashboard/HealthInsightDashboard?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

### Project Description
The Health Data Analytics Project is designed to analyze a wide range of health-related data, focusing on demographic, medical, and financial variables. This project is divided into several key components:

#### Data Preprocessing:
- Loading and cleaning the data using Python libraries such as pandas and numpy.
- Handling missing values, encoding categorical variables, and normalizing numerical features.

#### Exploratory Data Analysis (EDA):
- Conducting univariate and bivariate analyses to understand the distribution and relationships between different variables.
- Visualizing the data using Python libraries like matplotlib and seaborn.

Predictive Modeling:
- Building predictive models to forecast key metrics such as billing amount and length of hospital stay.
- Evaluating the models using metrics like Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).

Clustering and Segmentation:
- Implementing clustering algorithms (e.g., K-Means) to segment patients based on demographic and medical characteristics.
- Visualizing clusters to identify distinct patient groups and their characteristics.

Time Series Analysis and Forecasting:
- Analyzing trends and patterns in admission and discharge dates to understand seasonality and peaks in hospital admissions.
- Applying time series forecasting techniques to predict future admissions and billing amounts.

Visualizations Using Python Libraries and Tableau
- Trends in admissions and discharges over time.
- Distribution of billing amounts across different admission types and medical conditions.
- Correlation between demographic variables and medical outcomes.
- Clustering of patients based on demographic and medical characteristics.
- The dashboard on Tableau can be accessed via the following link:
  [Health Data Analytics Dashboard](https://public.tableau.com/views/HealthInsightDashboard/HealthInsightDashboard?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

Tools and Technologies
- Python: Used for data preprocessing, exploratory data analysis, predictive modeling, and clustering.
- Pandas, NumPy: Essential libraries for data manipulation and numerical computations.
- Matplotlib, Seaborn, plotly, bokeh: Libraries used for creating static visualizations during the EDA phase.
- Dash: Python libary used to build an app illustrating Age distribution
- Scikit-learn: Used for predictive modeling and clustering.
- Tableau: Tool used for creating the interactive dashboard.

Repository Structure
data/: Contains the raw and processed datasets used in the project.
notebooks/: Jupyter notebooks used for data analysis and visualization.
models/: Contains the scripts and models used for predictive analysis.
dashboard/: Tableau files and related assets for the interactive dashboard.
README.md: Overview and instructions for the repository.

Getting Started
To explore the project, clone the repository and navigate through the Jupyter notebooks to understand the analysis and visualizations. The Tableau dashboard provides an interactive way to delve deeper into the insights derived from the data.

bash
git clone https://github.com/yourusername/health-data-analytics.git
cd health-data-analytics

Conclusion
This Health Data Analytics project serves as a valuable resource for healthcare professionals and data analysts to explore and understand health-related data. The interactive dashboard and detailed analyses offer actionable insights that can improve decision-making processes in healthcare settings.
