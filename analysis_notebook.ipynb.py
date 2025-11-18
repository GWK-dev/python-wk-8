# analysis_notebook.ipynb content:

# Cell 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
import re
from collections import Counter

# Cell 2: Load and explore data
print("Loading CORD-19 dataset...")
# In practice, you would use: df = pd.read_csv('metadata.csv')
# For this example, we'll create sample data
def create_sample_data():
    np.random.seed(42)
    dates = pd.date_range('2019-11-01', '2021-08-01', freq='D')
    return pd.DataFrame({
        'title': [f'COVID-19 Research Paper {i}' for i in range(100)],
        'abstract': [f'Abstract for paper {i}' for i in range(100)],
        'publish_time': np.random.choice(dates, 100),
        'journal': np.random.choice(['The Lancet', 'Nature', 'Science', 'JAMA'], 100),
        'has_pdf_parse': np.random.choice([True, False], 100)
    })

df = create_sample_data()
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
df.head()

# Cell 3: Basic exploration
print("Dataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nBasic Statistics:")
print(df.describe())

# Cell 4: Data cleaning
df_clean = df.copy()
df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'])
df_clean['year'] = df_clean['publish_time'].dt.year
df_clean['abstract_word_count'] = df_clean['abstract'].str.split().str.len()
print("Data cleaning completed!")
print("New columns:", ['year', 'abstract_word_count'])

# Cell 5: Basic analysis
# Publications by year
year_counts = df_clean['year'].value_counts().sort_index()
print("Publications by year:")
print(year_counts)

# Top journals
top_journals = df_clean['journal'].value_counts().head(5)
print("\nTop 5 journals:")
print(top_journals)

# Cell 6: Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Publications by year
year_counts.plot(kind='bar', ax=axes[0,0], color='skyblue')
axes[0,0].set_title('Publications by Year')

# Plot 2: Top journals
top_journals.plot(kind='barh', ax=axes[0,1], color='lightcoral')
axes[0,1].set_title('Top Publishing Journals')

# Plot 3: Word cloud
wordcloud = WordCloud(width=400, height=300, background_color='white').generate(' '.join(df_clean['title']))
axes[1,0].imshow(wordcloud, interpolation='bilinear')
axes[1,0].axis('off')
axes[1,0].set_title('Title Word Cloud')

# Plot 4: Data availability
data_avail = df_clean['has_pdf_parse'].value_counts()
axes[1,1].pie(data_avail.values, labels=['PDF Available', 'No PDF'], autopct='%1.1f%%')
axes[1,1].set_title('PDF Availability')

plt.tight_layout()
plt.show()

# Cell 7: Insights and conclusions
print("KEY INSIGHTS:")
print("1. Temporal distribution shows research evolution over time")
print("2. Multiple prestigious journals contributing to COVID-19 research")
print("3. Good data accessibility with PDF availability")
print("4. Diverse research topics covered in publications")