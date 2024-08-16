#!/usr/bin/env python
# coding: utf-8

# In[330]:


pip install plotly


# In[334]:


pip install dash


# In[337]:


pip install bokeh


# In[338]:


# import Libaries
import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, html, dcc
from bokeh.io import show
from bokeh.plotting import figure
from bokeh.layouts import column
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[226]:


# Load the dataset
df = pd.read_csv('healthcare_dataset.csv')
df.head()


# In[227]:


# get general info on dataset
df.info()


# In[228]:


# check for duplicates and drop if any 
df.drop_duplicates()


# In[229]:


# drop or remove the name column for privacy concerns
# assign the datetime datatype to the date columns
df = df.drop('Name',axis=1)
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])


# In[230]:


df.info()


# In[231]:


# To get the statistical summary in numerical columns and transpose as headers for ease of understanding 
df.describe().T


# In[232]:


# check for nulls and sum it up 
df.isnull().sum()


# In[233]:


# rounding up the Billing Amount to 2 decimal place
df['Billing Amount'] = df['Billing Amount'].round(2)
df


# Frequency Analysis

# In[234]:


# Frequency of Gender
gender_counts = df['Gender'].value_counts()
print(gender_counts)


# In[235]:


# Bar plot for Gender Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Gender', data=df)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


# In[236]:


# Frequency of Blood Type
blood_type_counts = df['Blood Type'].value_counts()
print(blood_type_counts)


# In[237]:


# Bar plot for Blood Type Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Blood Type', data=df)
plt.title('Blood Type Distribution')
plt.xlabel('Blood Type')
plt.ylabel('Count')
plt.show()


# In[238]:


# Frequency of Medical Condition
medical_condition_counts = df['Medical Condition'].value_counts()
print(medical_condition_counts)


# In[239]:


# Bar plot for Medical Condition distribution
plt.figure(figsize=(8, 4))
sns.countplot(x='Medical Condition', data=df)
plt.title('Medical Condition Distribution')
plt.xlabel('Medical Condition')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[240]:


# Frequency of Insurance Provider
insurance_provider_counts = df['Insurance Provider'].value_counts()
print(insurance_provider_counts)


# In[241]:


# Bar plot for Insurance Provider distribution
plt.figure(figsize=(8, 4))
sns.countplot(x='Insurance Provider', data=df)
plt.title('Insurance Provider Distribution')
plt.xlabel('Insurance Provider')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# Combined Analysis with Subplots

# In[242]:


# Combined analysis with subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Gender Distribution
sns.countplot(x='Gender', data=df, ax=axs[0, 0])
axs[0, 0].set_title('Gender Distribution')
axs[0, 0].set_xlabel('Gender')
axs[0, 0].set_ylabel('Count')

# Blood Type Distribution
sns.countplot(x='Blood Type', data=df, ax=axs[0, 1])
axs[0, 1].set_title('Blood Type Distribution')
axs[0, 1].set_xlabel('Blood Type')
axs[0, 1].set_ylabel('Count')

# Medical Condition Distribution
sns.countplot(x='Medical Condition', data=df, ax=axs[1, 0])
axs[1, 0].set_title('Medical Condition Distribution')
axs[1, 0].set_xlabel('Medical Condition')
axs[1, 0].set_ylabel('Count')
axs[1, 0].tick_params(axis='x', rotation=45)

# Insurance Provider Distribution
sns.countplot(x='Insurance Provider', data=df, ax=axs[1, 1])
axs[1, 1].set_title('Insurance Provider Distribution')
axs[1, 1].set_xlabel('Insurance Provider')
axs[1, 1].set_ylabel('Count')
axs[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# Age Distribution

# In[243]:


df['Age'].describe()


# In[332]:


# Create interactive histogram
fig = px.histogram(df, x='Age', nbins=10, title='Age Distribution')
fig.show()


# In[333]:


# Create interactive scatter plot
fig = px.scatter(df, x='Age', y='Billing Amount', title='Age vs. Billing Amount')
fig.show()


# In[339]:


# Create a histogram using bokeh
hist, edges = np.histogram(df['Age'], bins=10)

p = figure(title='Age Distribution', x_axis_label='Age', y_axis_label='Frequency')
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color='blue', line_color='white')

show(p)


# In[340]:


# Create a scatter plot
p = figure(title='Age vs. Billing Amount', x_axis_label='Age', y_axis_label='Billing Amount')
p.scatter(df['Age'], df['Billing Amount'], size=10, color='blue', alpha=0.5)

show(p)


# In[244]:


# Age distribution using histograms 
plt.figure(figsize=(8, 6))
plt.hist(df['Age'], bins=15, edgecolor='black', alpha=0.7)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()


# In[245]:


# Age distribution using density plot
plt.figure(figsize=(8, 6))
sns.kdeplot(df['Age'], fill=True, color='blue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Density')
plt.grid(axis='y', alpha=0.75)
plt.show()


# In[246]:


# Age distribution using histograms and density plots
plt.figure(figsize=(8, 6))
sns.histplot(df['Age'], bins=15, kde=True, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()


# Gender Distribution

# In[247]:


# Gender vs Medical Condition
plt.figure(figsize=(10, 6))
sns.countplot(x='Medical Condition', hue='Gender', data=df)
plt.title('Gender vs Medical Condition')
plt.xlabel('Medical Condition')
plt.ylabel('Count')
plt.legend(title='Gender')
plt.show()


# In[248]:


# Gender vs. Blood Type
plt.figure(figsize=(10, 6))
sns.countplot(x='Blood Type', hue='Gender', data=df)
plt.title('Gender vs Blood Type')
plt.xlabel('Blood Type')
plt.ylabel('Count')
plt.legend(title='Gender')
plt.show()


# Time Series Analysis

# In[249]:


# Create a DataFrame for Admission Dates
admission_df = df[['Date of Admission']].copy()
admission_df['Count'] = 1


# In[250]:


# Create a DataFrame for Discharge Dates
discharge_df = df[['Discharge Date']].copy()
discharge_df['Count'] = 1


# In[251]:


# Resample the data by month
admission_counts = admission_df.resample('M', on='Date of Admission').sum()
discharge_counts = discharge_df.resample('M', on='Discharge Date').sum()


# In[252]:


# Rename the columns for clarity
admission_counts.rename(columns={'Count': 'Admissions'}, inplace=True)
discharge_counts.rename(columns={'Count': 'Discharges'}, inplace=True)


# In[253]:


# Plotting the trends in admissions and discharges
plt.figure(figsize=(12, 6))
plt.plot(admission_counts.index, admission_counts['Admissions'], label='Admissions', marker='o')
plt.plot(discharge_counts.index, discharge_counts['Discharges'], label='Discharges', marker='o')

plt.title('Trends in Admissions and Discharges')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Dual Axis Plot

# In[254]:


fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot admissions
color = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('Admissions', color=color)
ax1.plot(admission_counts.index, admission_counts['Admissions'], color=color, marker='o', label='Admissions')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

# Plot discharges on the second y-axis
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Discharges', color=color)
ax2.plot(discharge_counts.index, discharge_counts['Discharges'], color=color, marker='o', label='Discharges')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

plt.title('Trends in Admissions and Discharges')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# Seasonality and Trends

# In[255]:


# Decompose the admissions time series
decomposition = seasonal_decompose(admission_counts['Admissions'], model='additive', period=12)

# Plot the decomposition
decomposition.plot()
plt.show()


# Correlation Matrix

# In[256]:


# Select only the numeric columns
numeric_df = df.select_dtypes(include=['int64', 'float64'])


# In[257]:


# Calculate the correlation matrix
corr_matrix = numeric_df.corr()


# In[258]:


# Drop column room number 
corr_matrix = corr_matrix.drop(columns=['Room Number'])


# In[261]:


# Drop row room number 
corr_matrix = corr_matrix.drop(index=['Room Number'])


# In[262]:


# Display the correlation matrix
print(corr_matrix)


# Correlation Matrix Using a Heatmap

# In[263]:


# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_matrix, cmap=cmap, annot=True, fmt=".2f", linewidths=.5, cbar_kws={"shrink": .5})

# Add title and labels
plt.title('Correlation Matrix Heatmap', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# Bivariate Analysis

# In[264]:


# Plotting box plot
plt.figure(figsize=(8, 6))
sns.boxplot(x='Admission Type', y='Billing Amount', data=df)
plt.title('Billing Amount vs. Admission Type')
plt.xlabel('Admission Type')
plt.ylabel('Billing Amount')
plt.show()


# Cross-Tabulation

# In[266]:


# Cross-tabulation of Gender and Medical Condition
gender_medical_condition = pd.crosstab(df['Gender'], df['Medical Condition'])
print(gender_medical_condition)


# In[267]:


# Cross-tabulation of Gender and Blood Type
gender_blood_type = pd.crosstab(df['Gender'], df['Blood Type'])
print(gender_blood_type)


# In[268]:


# Cross-tabulation of Age and Medical Condition
age_condition_ct = pd.crosstab(df['Age'], df['Medical Condition'])
print(age_condition_ct)


# Heatmap for Cross-Tabulation

# In[269]:


# Heatmap for Gender vs Medical Condition
plt.figure(figsize=(8, 6))
sns.heatmap(gender_medical_condition, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Heatmap of Gender vs Medical Condition')
plt.ylabel('Gender')
plt.xlabel('Medical Condition')
plt.show()


# In[270]:


# Heatmap for Gender vs Blood Type
plt.figure(figsize=(8, 6))
sns.heatmap(gender_blood_type, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Heatmap of Gender vs Blood Type')
plt.ylabel('Gender')
plt.xlabel('Blood Type')
plt.show()


# In[271]:


# Heatmap for Age vs Medical Condition
# Plotting bar plot
age_condition_ct.plot(kind='bar', stacked=True, figsize=(10, 7))
plt.title('Age vs. Medical Condition')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(title='Medical Condition')
plt.show()


# Insurance Provider vs. Billing Amount

# In[274]:


# Plotting scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Insurance Provider', y='Billing Amount', data=df, hue='Insurance Provider', s=100)
plt.title('Insurance Provider vs. Billing Amount')
plt.xlabel('Insurance Provider')
plt.ylabel('Billing Amount')
plt.legend(title='Insurance Provider', loc='upper right')
plt.show()


# Hypothesis Testing

# In[275]:


# Separate the billing amounts by admission type
emergency_billing = df[df['Admission Type'] == 'Emergency']['Billing Amount']
elective_billing = df[df['Admission Type'] == 'Elective']['Billing Amount']
urgent_billing = df[df['Admission Type'] == 'Urgent']['Billing Amount']


# In[276]:


# Combine all billing amounts into one list and create a list of corresponding group labels
all_billing = [emergency_billing, elective_billing, urgent_billing]
group_labels = ['Emergency', 'Elective', 'Urgent']


# In[277]:


# Perform one-way ANOVA
f_stat, p_value = stats.f_oneway(*all_billing)
print(f"F-statistic: {f_stat:.2f}")
print(f"P-value: {p_value:.4f}")


# In[278]:


# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference in average billing amount between admission types.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in average billing amount between admission types.")


# Association between Gender and Medical Condition

# In[281]:


# Perform Chi-Square test
chi2, p, dof, expected = stats.chi2_contingency(gender_medical_condition)

print(f"Chi-Square Statistic: {chi2:.2f}")
print(f"P-value: {p:.4f}")

# Interpretation
alpha = 0.05
if p < alpha:
    print("Reject the null hypothesis: There is an association between gender and medical condition.")
else:
    print("Fail to reject the null hypothesis: There is no association between gender and medical condition.")


# Predictive Modelling

# In[282]:


df_predictive = df


# In[283]:


# Convert date columns to numeric features (e.g., days since the start)
df_predictive['Days in Hospital'] = (df_predictive['Discharge Date'] - df_predictive['Date of Admission']).dt.days


# In[284]:


df_filtered = df_predictive[['Age', 'Billing Amount', 'Days in Hospital']]


# In[285]:


# Extract features and target
X = df_filtered[['Age', 'Days in Hospital']]  # Features
y = df_filtered['Billing Amount']  # Target


# In[286]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[314]:


# Initialize and fit the model
model = LinearRegression()
model.fit(X_train, y_train)


# In[288]:


# Make predictions
lr_predictions = model.predict(X_test)

# Evaluate the model
print("Mean Squared Error:", mean_squared_error(y_test, lr_predictions))
print("R^2 Score:", r2_score(y_test, lr_predictions))


# In[289]:


# Plot histograms
plt.figure(figsize=(15, 5))

# Plot Age Distribution
plt.subplot(1, 3, 1)
sns.histplot(df_filtered['Age'], kde=True, color='blue')
plt.title('Age Distribution')

# Plot Billing Amount Distribution
plt.subplot(1, 3, 2)
sns.histplot(df_filtered['Billing Amount'], kde=True, color='green')
plt.title('Billing Amount Distribution')

# Plot Days in Hospital Distribution
plt.subplot(1, 3, 3)
sns.histplot(df_filtered['Days in Hospital'], kde=True, color='orange')
plt.title('Days in Hospital Distribution')

# Show the plots
plt.tight_layout()  # Adjust subplots to fit into figure area.
plt.show()


# In[290]:


# Calculate residuals
residuals = y_test - lr_predictions

# Plot residuals
plt.figure(figsize=(10, 6))
plt.scatter(lr_predictions, residuals, color='blue', alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Billing Amount')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.show()


# In[291]:


# Plotting Actual vs. Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, lr_predictions, color='blue', alpha=0.6)
plt.xlabel('Actual Billing Amount')
plt.ylabel('Predicted Billing Amount')
plt.title('Linear Regression: Actual vs Predicted')
plt.show()


# Decision Trees

# In[315]:


# Initialize and fit the Decision Tree model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)


# In[293]:


# Make predictions
dt_predictions = dt_model.predict(X_test)

# Evaluate the model
print("Decision Tree - Mean Squared Error:", mean_squared_error(y_test, dt_predictions))
print("Decision Tree - R^2 Score:", r2_score(y_test, dt_predictions))


# Random Forest

# In[316]:


# Initialize and fit the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
rf_predictions = rf_model.predict(X_test)


# In[317]:


# Evaluate the model
print("Random Forest - Mean Squared Error:", mean_squared_error(y_test, rf_predictions))
print("Random Forest - R^2 Score:", r2_score(y_test, rf_predictions))


# In[318]:


import matplotlib.pyplot as plt

# Plot for Decision Tree
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, dt_predictions, color='blue', alpha=0.5)
plt.xlabel('Actual Billing Amount')
plt.ylabel('Predicted Billing Amount')
plt.title('Decision Tree: Actual vs Predicted')

# Plot for Random Forest
plt.subplot(1, 2, 2)
plt.scatter(y_test, rf_predictions, color='green', alpha=0.5)
plt.xlabel('Actual Billing Amount')
plt.ylabel('Predicted Billing Amount')
plt.title('Random Forest: Actual vs Predicted')

plt.tight_layout()
plt.show()


# In[319]:


importances = rf_model.feature_importances_
feature_names = X.columns

# Create a DataFrame for visualization
importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot Feature Importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importances_df, palette='viridis')
plt.title('Feature Importances (Random Forest)')
plt.show()


# In[320]:


# Convert feature names to a list
feature_names_list = X.columns.tolist()

# Plot Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=feature_names_list, filled=True, rounded=True)
plt.title('Decision Tree Visualization')
plt.show()


# Medical condition and admission type

# In[321]:


sns.countplot(x='Medical Condition', hue='Admission Type', data=df)
plt.show()


# Billing amount range for each medical condition

# In[ ]:


sns.boxplot(x='Medical Condition', y='Billing Amount', data=df)
plt.show()


# Medications count for each medical condition

# In[322]:


df['Medication'].unique()


# In[323]:


sns.countplot(x='Medical Condition', hue='Medication', data=df)
plt.show()


# Stacked bar plot for Gender vs. Medical Condition

# In[324]:


df.groupby(['Medical Condition', 'Gender']).size().unstack().plot(kind='bar', stacked=True)
plt.title('Gender vs. Medical Condition')
plt.show()


# Scatterplot for test results with medical conditions and bills

# In[325]:


sns.scatterplot(x='Test Results', y='Billing Amount', hue='Medical Condition', data=df)
plt.title('Test Results vs. Billing Amount by Medical Condition')
plt.show()


# Gender vs Age

# In[326]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='Age', data=df)
plt.title('Age by Gender')
plt.show()


# Pairplot between Medical condition and [age, billing amount]

# In[327]:


# Example plotting with Seaborn
sns.histplot(df['Age'])
plt.tight_layout()  # Manually adjust the layout
plt.show()


# How long does patient stays

# In[328]:


plt.figure(figsize=(10, 6))
sns.histplot(df['Days in Hospital'], kde=True)
plt.title('Distribution of Length of Stay')
plt.show()


# Billing Amount, Age, Length stay

# In[329]:


# Group by 'Medical Condition' and calculate the mean
grouped = df.groupby('Medical Condition').agg({
    'Billing Amount': 'mean',
    'Age': 'mean',
    'Days in Hospital': 'mean'
})

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot for Billing Amount
color = 'tab:blue'
ax1.set_xlabel('Medical Condition')
ax1.set_ylabel('Mean Billing Amount', color=color)
ax1.bar(grouped.index, grouped['Billing Amount'], color=color, alpha=0.6, label='Billing Amount')
ax1.tick_params(axis='y', labelcolor=color)

# Create a second y-axis for Age and Days in Hospital
ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Mean Age', color=color)
ax2.plot(grouped.index, grouped['Age'], color=color, marker='o', label='Age')
ax2.tick_params(axis='y', labelcolor=color)

ax3 = ax1.twinx()
color = 'tab:orange'
ax3.spines['right'].set_position(('outward', 60))  # Move the third y-axis outward
ax3.set_ylabel('Mean Days in Hospital', color=color)
ax3.plot(grouped.index, grouped['Days in Hospital'], color=color, marker='s', linestyle='--', label='Days in Hospital')
ax3.tick_params(axis='y', labelcolor=color)

# Adding titles and legends
plt.title('Mean Billing Amount, Age, and Length of Stay by Medical Condition')
fig.tight_layout()  # Adjust layout to fit elements

# Show the plot
plt.show()


# In[336]:


# Create a Dash app
app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(
        id='scatter-plot',
        figure=px.scatter(df, x='Age', y='Billing Amount', title='Age vs. Billing Amount')
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:




