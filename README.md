# Health-Data-Analytics
This repository contains a comprehensive Health Data Analytics project utilizing datasets sourced from [Kaggle](https://www.kaggle.com/datasets/muhammadehsan000/healthcare-dataset-2019-2024?resource=download). The primary goal of this project is to analyze and visualize health-related data to uncover valuable insights that can aid healthcare professionals and organizations in making informed decisions. By leveraging the power of Python libraries and Tableau, this project provides an interactive and intuitive dashboard for exploring the data. 
View the interactive dashboard [here](https://public.tableau.com/views/HealthInsightDashboard/HealthInsightDashboard?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

<img src = "https://github.com/Frances-Odunaiya/Health-Data-Analytics/blob/main/Visualizations%20snippet/HeatMap%20of%20Gender%20vs%20Medical%20Condition.png" alt ="Visuals1" width ="250px" height = "250px"> <img src = "" alt ="" width ="" height = ""> <img src = "" alt ="" width ="" height = "">

### Project Description
The Health Data Analytics Project is designed to analyze a wide range of health-related data, focusing on demographic, medical, and financial variables. This project is divided into several key components:

#### Data Preprocessing:
- Loading and cleaning the data using Python libraries such as pandas and numpy.
- Handling missing values, encoding categorical variables, and normalizing numerical features.

#### Exploratory Data Analysis (EDA):
- Conducting univariate and bivariate analyses to understand the distribution and relationships between different variables.
- Visualizing the data using Python libraries like matplotlib and seaborn.

#### Predictive Modeling:
- Building predictive models to forecast key metrics such as billing amount and length of hospital stay.
- Evaluating the models using metrics like Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).

#### Clustering and Segmentation:
- Implementing clustering algorithms (e.g., K-Means) to segment patients based on demographic and medical characteristics.
- Visualizing clusters to identify distinct patient groups and their characteristics.

#### Time Series Analysis and Forecasting:
- Analyzing trends and patterns in admission and discharge dates to understand seasonality and peaks in hospital admissions.
- Applying time series forecasting techniques to predict future admissions and billing amounts.

#### Visualizations Using Python Libraries and Tableau
- Trends in admissions and discharges over time.
- Distribution of billing amounts across different admission types and medical conditions.
- Correlation between demographic variables and medical outcomes.
- Clustering of patients based on demographic and medical characteristics.
- The dashboard on Tableau can be accessed via the following link:
  [Health Data Analytics Dashboard](https://public.tableau.com/views/HealthInsightDashboard/HealthInsightDashboard?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

#### Tools and Technologies
- Python: Used for data preprocessing, exploratory data analysis, predictive modeling, and clustering.
- Pandas, NumPy: Essential libraries for data manipulation and numerical computations.
- Matplotlib, Seaborn, plotly, bokeh: Libraries used for creating static visualizations during the EDA phase.
- Dash: Python libary used to build an app illustrating Age distribution
- Scikit-learn: Used for predictive modeling and clustering.
- Tableau: Tool used for creating the interactive dashboard.

#### Getting Started
To explore the project, clone the repository and navigate through the Jupyter notebooks to understand the analysis and visualizations. The Tableau dashboard provides an interactive way to delve deeper into the insights derived from the data.

bash
git clone https://github.com/yourusername/health-data-analytics.git
cd health-data-analytics

#### Insights
- Gender Distribution: Is there an imbalance in the gender distribution? The dataset reveals a balanced gender distribution, with males constituting X% and females Y% of the total patient population. This near-equal representation ensures that any analysis derived from the dataset is not skewed by gender imbalance, allowing for more accurate and reliable conclusions regarding healthcare outcomes and demographic trends.

- Blood Type Distribution: Which blood types are most or least common? Upon reviewing the dataset, we found that [Blood Type A+] is the most common in females, followed by [Blood Type B-] in males . The least common blood type is [Blood Type A+] in males. This distribution is not consistent with general population trends, where certain blood types are naturally more prevalent. Understanding the distribution of blood types can help in planning for blood bank inventories and managing transfusion needs efficiently.

- Medical Condition Distribution: What is the prevalence of different medical conditions? The dataset highlights a diverse range of medical conditions among the patients. The most prevalent condition is Diabetes in both genders followed by Cancer in Males and Arthritis in Females. This distribution provides valuable insights into the healthcare demands of the patient population and can guide resource allocation and treatment prioritization to address the most significant health concerns effectively.

- Is there a significant difference in the Age and between Billing Amount?
The t-test analysis reveals there is no significant difference in average billing amount between admission types. On average, All age group pay the same amount as it relates to their medical condition. Understanding these can help healthcare providers tailor their care plans and resource allocation to meet the specific needs of each gender.

#### Conclusion
This Health Data Analytics project serves as a valuable resource for healthcare professionals and data analysts to explore and understand health-related data. The interactive dashboard and detailed analyses offer actionable insights that can improve decision-making processes in healthcare settings.
