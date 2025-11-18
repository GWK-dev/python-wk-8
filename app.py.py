# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from wordcloud import WordCloud
import re
from collections import Counter

# Set page configuration
st.set_page_config(
    page_title="CORD-19 Research Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_sample_data():
    """Load enhanced sample data that mimics real CORD-19 structure"""
    np.random.seed(42)
    dates = pd.date_range('2019-12-01', '2021-06-01', freq='D')
    journals = ['The Lancet', 'Nature', 'Science', 'JAMA', 'BMJ', 'NEJM', 'PLoS One', 'BMC Medicine']
    
    sample_data = {
        'title': [
            'COVID-19 transmission dynamics and control measures in urban populations',
            'Clinical characteristics and outcomes of coronavirus patients in ICU',
            'Vaccine development for SARS-CoV-2: Current status and challenges',
            'Effectiveness of social distancing in pandemic control: A meta-analysis',
            'Global economic impact of COVID-19 lockdown measures on GDP',
            'Mental health consequences during pandemic isolation periods',
            'Treatment protocols and management of severe COVID-19 cases',
            'Viral mutation patterns and implications for vaccine efficacy',
            'Asymptomatic transmission of SARS-CoV-2 in community settings',
            'Healthcare system preparedness for pandemic respiratory diseases',
            'Remdesivir and dexamethasone in COVID-19 treatment protocols',
            'Long-term effects of COVID-19 on pulmonary function',
            'Mask-wearing effectiveness in reducing viral transmission',
            'Pediatric cases of COVID-19: Epidemiology and clinical features',
            'Telemedicine adoption during the COVID-19 pandemic'
        ],
        'abstract': [
            'Comprehensive study of transmission patterns and effectiveness of various control measures in urban environments with high population density.',
            'Detailed analysis of clinical features, treatment outcomes, and prognostic factors in intensive care unit patients with confirmed COVID-19.',
            'Systematic review of current vaccine development approaches, clinical trial results, and challenges in global distribution and acceptance.',
            'Meta-analysis evaluating the impact of social distancing policies on infection rates across multiple countries and demographic groups.',
            'Economic modeling assessment of lockdown consequences on global GDP, employment rates, and industry-specific impacts.',
            'Longitudinal examination of mental health effects, anxiety, depression, and coping strategies during extended isolation periods.',
            'Evidence-based development of treatment guidelines and clinical management protocols for severe and critical COVID-19 cases.',
            'Genomic investigation of viral mutation rates, variant emergence, and their potential impact on current vaccine effectiveness.',
            'Epidemiological study of asymptomatic transmission routes, detection challenges, and implications for public health policies.',
            'Evaluation of healthcare system capacity, resource allocation, and preparedness strategies for respiratory pandemic scenarios.',
            'Clinical trial results and real-world evidence for antiviral treatments and immunomodulators in COVID-19 management.',
            'Prospective cohort study examining long-term pulmonary complications and recovery patterns in post-COVID patients.',
            'Experimental and observational studies on mask efficacy, material effectiveness, and behavioral factors in transmission reduction.',
            'Comprehensive analysis of COVID-19 epidemiology, presentation, and outcomes in pediatric populations across different age groups.',
            'Assessment of telemedicine implementation, adoption barriers, and patient satisfaction during pandemic healthcare delivery shifts.'
        ],
        'publish_time': np.random.choice(dates, 15),
        'authors': [
            'Smith A, Johnson B, Chen C', 'Lee C, Wang D, Garcia E', 'Brown F, Davis G, Wilson H',
            'Taylor I, Martinez J, Anderson K', 'Thomas L, Clark M, Rodriguez N', 'Lewis O, Walker P',
            'Hall Q, Young R, Allen S', 'King T, Wright U, Scott V', 'Green W, Adams X, Baker Y',
            'Nelson Z, Carter AA, Mitchell BB', 'Perez CC, Roberts DD, Turner EE', 'Phillips FF, Campbell GG',
            'Evans HH, Parker II, Collins JJ', 'Edwards KK, Stewart LL, Sanchez MM', 'Morris NN, Rogers OO'
        ],
        'journal': np.random.choice(journals, 15),
        'has_pdf_parse': np.random.choice([True, False], 15, p=[0.7, 0.3]),
        'has_pmc_xml_parse': np.random.choice([True, False], 15, p=[0.6, 0.4]),
        'who_covidence_id': [f"COV_{i:06d}" for i in range(100001, 100016)],
        'source_x': np.random.choice(['PubMed', 'PMC', 'arXiv', 'WHO'], 15)
    }
    
    df = pd.DataFrame(sample_data)
    return df

def clean_data(df):
    """Clean and prepare the dataset"""
    # Convert publish_time to datetime
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
    
    # Handle missing values
    df['abstract'] = df['abstract'].fillna('No abstract available')
    df['authors'] = df['authors'].fillna('Unknown authors')
    df['journal'] = df['journal'].fillna('Unknown journal')
    
    # Create new columns for analysis
    df['year'] = df['publish_time'].dt.year
    df['month'] = df['publish_time'].dt.month
    df['abstract_word_count'] = df['abstract'].str.split().str.len()
    df['title_word_count'] = df['title'].str.split().str.len()
    
    return df

def generate_wordcloud(text_series):
    """Generate word cloud from text series"""
    text = ' '.join(text_series.dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                         colormap='viridis', max_words=100).generate(text)
    return wordcloud

def main():
    st.title("🔬 CORD-19 Research Dataset Explorer")
    st.markdown("### Comprehensive Analysis of COVID-19 Research Papers")
    
    # Load and clean data
    with st.spinner('Loading and processing data...'):
        df = load_sample_data()
        df = clean_data(df)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Analysis Section", 
                                   ["Data Overview", "Data Cleaning", "Basic Analysis", 
                                    "Visualizations", "Research Insights"])
    
    if app_mode == "Data Overview":
        show_data_overview(df)
    elif app_mode == "Data Cleaning":
        show_data_cleaning(df)
    elif app_mode == "Basic Analysis":
        show_basic_analysis(df)
    elif app_mode == "Visualizations":
        show_visualizations(df)
    elif app_mode == "Research Insights":
        show_research_insights(df)

def show_data_overview(df):
    st.header("📋 Dataset Overview & Basic Exploration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Papers", len(df))
    with col2:
        st.metric("Date Range", f"{df['publish_time'].min().strftime('%Y-%m')} to {df['publish_time'].max().strftime('%Y-%m')}")
    with col3:
        st.metric("Unique Journals", df['journal'].nunique())
    
    st.subheader("Data Structure")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Shape:**", df.shape)
        st.write("**Columns and Data Types:**")
        for col in df.columns:
            st.write(f"- {col}: {df[col].dtype}")
    
    with col2:
        st.write("**First 5 Rows:**")
        st.dataframe(df.head(), use_container_width=True)
    
    st.subheader("Missing Values Analysis")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing_data, 'Missing Percentage': missing_percent})
    st.dataframe(missing_df[missing_df['Missing Count'] > 0])

def show_data_cleaning(df):
    st.header("🧹 Data Cleaning & Preparation")
    
    st.subheader("Missing Values Handling")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Before Cleaning:**")
        missing_before = df.isnull().sum().sum()
        st.write(f"Total missing values: {missing_before}")
    
    with col2:
        st.write("**After Cleaning:**")
        # Show cleaning actions
        st.write("✅ publish_time converted to datetime")
        st.write("✅ abstract missing values filled")
        st.write("✅ authors missing values filled")
        st.write("✅ journal missing values filled")
        st.write("✅ New features created (year, month, word counts)")
    
    st.subheader("Data Quality Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Papers with PDF", df['has_pdf_parse'].sum())
    with col2:
        st.metric("Papers with PMC XML", df['has_pmc_xml_parse'].sum())
    with col3:
        st.metric("Avg Abstract Words", f"{df['abstract_word_count'].mean():.1f}")
    
    st.subheader("Cleaned Data Sample")
    st.dataframe(df[['title', 'journal', 'publish_time', 'year', 'abstract_word_count']].head(8))

def show_basic_analysis(df):
    st.header("📊 Basic Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Publications by Year")
        year_counts = df['year'].value_counts().sort_index()
        st.bar_chart(year_counts)
        
        st.subheader("Top Publishing Journals")
        top_journals = df['journal'].value_counts().head(5)
        st.dataframe(top_journals)
    
    with col2:
        st.subheader("Word Frequency in Titles")
        # Simple word frequency analysis
        all_titles = ' '.join(df['title'].dropna().str.lower())
        words = re.findall(r'\b\w+\b', all_titles)
        word_freq = Counter(words)
        
        # Remove common stop words
        stop_words = {'the', 'and', 'of', 'in', 'to', 'a', 'for', 'on', 'with', 'as', 'by', 'an', 'at'}
        filtered_words = {word: count for word, count in word_freq.items() 
                         if word not in stop_words and len(word) > 3}
        
        top_words = pd.Series(filtered_words).sort_values(ascending=False).head(10)
        st.bar_chart(top_words)
    
    st.subheader("Descriptive Statistics")
    numerical_cols = ['abstract_word_count', 'title_word_count']
    st.dataframe(df[numerical_cols].describe())

def show_visualizations(df):
    st.header("📈 Advanced Visualizations")
    
    # Interactive controls
    col1, col2 = st.columns(2)
    
    with col1:
        selected_year = st.selectbox("Select Year for Analysis", sorted(df['year'].unique()))
    with col2:
        chart_type = st.radio("Chart Style", ['Bar Chart', 'Line Chart', 'Area Chart'])
    
    st.subheader("Publication Trends Over Time")
    
    # Monthly publication trends
    monthly_data = df.groupby([df['publish_time'].dt.to_period('M')]).size()
    monthly_data.index = monthly_data.index.astype(str)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    if chart_type == 'Bar Chart':
        monthly_data.plot(kind='bar', ax=ax, color='steelblue', alpha=0.8)
    elif chart_type == 'Line Chart':
        monthly_data.plot(kind='line', ax=ax, marker='o', linewidth=2, markersize=4)
    else:
        monthly_data.plot(kind='area', ax=ax, alpha=0.6)
    
    ax.set_title('Monthly Publication Trends', fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Publications')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Journals")
        journal_counts = df['journal'].value_counts().head(8)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        journal_counts.plot(kind='barh', ax=ax, color='lightcoral')
        ax.set_title('Top Publishing Journals', fontweight='bold')
        ax.set_xlabel('Number of Papers')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Word Cloud of Research Titles")
        wordcloud = generate_wordcloud(df['title'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Common Words in Research Titles', fontweight='bold')
        st.pyplot(fig)
    
    st.subheader("Data Source Distribution")
    if 'source_x' in df.columns:
        source_counts = df['source_x'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%', startangle=90)
        ax.set_title('Paper Sources Distribution', fontweight='bold')
        st.pyplot(fig)

def show_research_insights(df):
    st.header("💡 Research Insights & Findings")
    
    st.subheader("Key Statistical Findings")
    
    insights = [
        f"📈 **Temporal Distribution**: Research publications span from {df['publish_time'].min().strftime('%B %Y')} to {df['publish_time'].max().strftime('%B %Y')}",
        f"📚 **Journal Diversity**: {df['journal'].nunique()} different journals represented in the dataset",
        f"🔬 **Research Depth**: Average abstract length of {df['abstract_word_count'].mean():.1f} words indicates comprehensive study descriptions",
        f"📄 **Data Accessibility**: {df['has_pdf_parse'].sum()} papers ({df['has_pdf_parse'].mean()*100:.1f}%) have available PDF content",
        f"👥 **Collaboration**: Multiple authors per paper indicate strong research collaboration patterns"
    ]
    
    for insight in insights:
        st.info(insight)
    
    st.subheader("Research Focus Areas")
    
    # Analyze common themes
    themes = {
        'Clinical Studies': df['title'].str.contains('clinical|patient|treatment', case=False).sum(),
        'Transmission Research': df['title'].str.contains('transmission|spread|infection', case=False).sum(),
        'Vaccine Development': df['title'].str.contains('vaccine|immunization', case=False).sum(),
        'Public Health': df['title'].str.contains('public health|policy|measure', case=False).sum(),
        'Economic Impact': df['title'].str.contains('economic|impact|GDP', case=False).sum()
    }
    
    theme_df = pd.DataFrame(list(themes.items()), columns=['Theme', 'Count'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    theme_df.sort_values('Count', ascending=True).plot(kind='barh', x='Theme', y='Count', ax=ax, color='teal')
    ax.set_title('Research Themes Distribution', fontweight='bold')
    ax.set_xlabel('Number of Papers')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("Interactive Data Exploration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        year_filter = st.multiselect(
            "Filter by Publication Year",
            options=sorted(df['year'].unique()),
            default=sorted(df['year'].unique())
        )
    
    with col2:
        journal_filter = st.multiselect(
            "Filter by Journal",
            options=df['journal'].unique(),
            default=df['journal'].unique()[:3] if len(df) > 3 else df['journal'].unique()
        )
    
    filtered_df = df[
        (df['year'].isin(year_filter)) & 
        (df['journal'].isin(journal_filter))
    ]
    
    st.write(f"**Filtered Results:** {len(filtered_df)} papers")
    st.dataframe(filtered_df[['title', 'journal', 'publish_time', 'authors']], use_container_width=True)

if __name__ == "__main__":
    main()