#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[4]:


df = pd.read_csv('C:\\Users\\Administrator\\Desktop\\DATASET\\HR_comma_sep.csv')
df


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


def column_summary(df):
    summary_data = []

    for col_name in df.columns:
        col_dtype = df[col_name].dtype
        num_of_nulls = df[col_name].isnull().sum()
        num_of_non_nulls = df[col_name].notnull().sum()
        num_of_distinct_values = df[col_name].nunique()

        if num_of_distinct_values <= 10:
            distinct_values_counts = df[col_name].value_counts().to_dict()
        else:
            top_10_values_counts = df[col_name].value_counts().head(10).to_dict()
            distinct_values_counts = {k: v for k, v in sorted(top_10_values_counts.items(), key=lambda item: item[1], reverse=True)}

        summary_data.append({
            'col_name': col_name,
            'col_dtype': col_dtype,
            'num_of_nulls': num_of_nulls,
            'num_of_non_nulls': num_of_non_nulls,
            'num_of_distinct_values': num_of_distinct_values,
            'distinct_values_counts': distinct_values_counts
        })

    summary_df = pd.DataFrame(summary_data)
    return summary_df

# Usage
summary_df = column_summary(df)
display(summary_df)




# In[9]:


categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print("Categorical variables:")
print(categorical_cols)

numeric_cols = df.select_dtypes(include=['int64', 'float64','int32','float32']).columns.tolist()
print("numeric variables:")
print(numeric_cols)


# In[10]:


df.isnull().sum()


# In[11]:


df.duplicated().sum()


# In[12]:


df=df.drop_duplicates()


# In[13]:


df.duplicated().sum()


# In[14]:


df.dtypes


# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()


# In[16]:


categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    print(f"\n{col} value counts:\n{df[col].value_counts()}")


# In[17]:


df_numeric = df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(10, 8))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# In[18]:


sns.boxplot(x='left', y='satisfaction_level', data=df)
plt.title('Satisfaction vs Employee Exit')
plt.show()


# In[19]:


sns.boxplot(x='left', y='average_montly_hours', data=df)
plt.title('Monthly Hours vs Employee Exit')
plt.show()


# In[20]:


sns.boxplot(x='left', y='number_project', data=df)
plt.title('number_project vs Employee Exit')
plt.show()


# In[21]:


sns.barplot(x='left', y='Work_accident', data=df)
plt.title('Work_accident vs Employee Exit')
plt.show()


# In[22]:


sns.barplot(x='left', y='time_spend_company', data=df)
plt.title('time_spend_company vs Employee Exit')
plt.show()


# In[23]:


categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    if 'left' in df.columns:
        sns.countplot(x=col, hue='left', data=df)
        plt.title(f"{col} vs Employee Exit")
        plt.xticks(rotation=45)
        plt.show()


# In[24]:


for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()


# In[25]:


from scipy.stats import zscore

z_scores = df[numeric_cols].apply(zscore)
outliers = (abs(z_scores) > 3).sum()
print(outliers)


# In[26]:


# Calculate Z-scores for numeric columns
z_scores = df[numeric_cols].apply(zscore)

# Filter out rows where any numeric column has a Z-score > 3 or < -3
df = df[(abs(z_scores) < 3).all(axis=1)]

# Confirm new shape
print(df.shape)


# In[27]:


x=df.drop(columns=['left'],axis=1)
y=df['left']


# In[28]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[29]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


numeric_features = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years']
categorical_features = ['Department', 'salary']


preprocessor = make_column_transformer(
    (OneHotEncoder(), categorical_features),
    (StandardScaler(), numeric_features),
    remainder='passthrough'
)


pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('Classifier', XGBClassifier())
])

# Hyperparameter tuning 
param_grid = {
    'Classifier__n_estimators': [100, 200],
    'Classifier__max_depth': [3, 5, 7],
    'Classifier__learning_rate': [0.01, 0.1, 0.2],
    'Classifier__subsample': [0.8, 1],
    'Classifier__colsample_bytree': [0.8, 1],
    'Classifier__gamma': [0, 1],
    'Classifier__min_child_weight': [1, 5]
}
grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)
print("Best Params:", grid.best_params_)
print("Best Score:", grid.best_score_)


# In[30]:


best_params = grid.best_params_
pipeline_best_param = Pipeline([
    ('preprocessor', preprocessor),
    ('Classifier', XGBClassifier(
        n_estimators=best_params['Classifier__n_estimators'],
        max_depth=best_params['Classifier__max_depth'],
        learning_rate=best_params['Classifier__learning_rate'],
        subsample=best_params['Classifier__subsample'],
        colsample_bytree=best_params['Classifier__colsample_bytree'],
        gamma=best_params['Classifier__gamma'],
        min_child_weight=best_params['Classifier__min_child_weight'],
    ))
])


# In[31]:


pipeline_best_param.fit(X_train,y_train)
pipeline_best_param.score(X_test,y_test)


# In[32]:


import pandas as pd
import matplotlib.pyplot as plt


ohe = pipeline.named_steps['preprocessor'].transformers_[0][1]
ohe_feature_names = ohe.get_feature_names_out(categorical_features)

all_feature_names = list(ohe_feature_names) + numeric_features

xgb_model = pipeline_best_param.named_steps['Classifier']

importances = xgb_model.feature_importances_

feat_importance = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feat_importance.plot(kind='bar')
plt.title("XGBoost Feature Importance")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()


# In[75]:


import pickle

with open('modell_001.pkl', 'wb') as file:
    pickle.dump(pipeline_best_param, file)


# In[ ]:




