
import streamlit as st
import os
import pandas as pd
import plotly.express as px

from sklearn.feature_extraction.text import CountVectorizer

# Page configuration
st.set_page_config(
    page_title="Clinical Trial Insights",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Clinical Trial Insights & Reporting")

st.write(
    "Explore disease distribution, medical terms, "
    "clinical trial text patterns and model performance."
)

# Load cleaned clinical trial dataset
data_path = os.path.expanduser(
    '~/Documents/clinic/data/processed/clinical_trial_cleaned.csv'
)

df = pd.read_csv(data_path)

# Create application tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Disease Distribution",
    "Medical Terms",
    "Text Analysis",
    "Model Performance"
])

# Tab 1 - Disease Distribution
with tab1:
    st.header("Disease Category Distribution")

    disease_counts = df['source_condition_query'].value_counts().reset_index()
    disease_counts.columns = ['Disease Category', 'Number of Trials']

    fig = px.bar(
        disease_counts,
        x='Disease Category',
        y='Number of Trials',
        title='Clinical Trials by Disease Category',
        text='Number of Trials'
    )

    st.plotly_chart(fig, use_container_width=True)

# Tab 2 - Medical Terms
with tab2:
    st.header("Top 20 Frequently Occurring Medical Terms")

    vectorizer = CountVectorizer(
        stop_words='english'
    )

    word_matrix = vectorizer.fit_transform(
        df['brief_summary'].fillna('')
    )

    word_counts = word_matrix.sum(axis=0).A1

    word_df = pd.DataFrame({
        'Medical Term': vectorizer.get_feature_names_out(),
        'Frequency': word_counts
    })

    word_df = word_df.sort_values(
        by='Frequency',
        ascending=False
    ).head(20)

    fig_words = px.bar(
        word_df.sort_values('Frequency'),
        x='Frequency',
        y='Medical Term',
        orientation='h',
        title='Top 20 Frequently Occurring Terms'
    )

    st.plotly_chart(fig_words, use_container_width=True)

# Tab 3 - Text Analysis
with tab3:
    st.header("Clinical Trial Text Pattern Analysis")

    selected_disease = st.selectbox(
        "Select Disease Category",
        sorted(df['source_condition_query'].unique())
    )

    disease_text = df[
        df['source_condition_query'] == selected_disease
    ]['brief_summary'].fillna('')

    text_vectorizer = CountVectorizer(
        stop_words='english',
        max_features=15
    )

    text_matrix = text_vectorizer.fit_transform(disease_text)

    term_counts = text_matrix.sum(axis=0).A1

    term_df = pd.DataFrame({
        'Term': text_vectorizer.get_feature_names_out(),
        'Frequency': term_counts
    })

    term_df = term_df.sort_values(
        by='Frequency',
        ascending=False
    )

    fig_text = px.bar(
        term_df,
        x='Term',
        y='Frequency',
        title=f'Common Terms in {selected_disease} Trials'
    )

    st.plotly_chart(fig_text, use_container_width=True)

    st.write("Total clinical trials:", len(disease_text))

# Tab 4 - Model Performance
with tab4:
    st.header("Logistic Regression Model Performance")

    col1, col2 = st.columns(2)

    col1.metric("Training Accuracy", "96.80%")
    col2.metric("Testing Accuracy", "94.80%")

    performance = pd.DataFrame({
        'Disease Category': [
            'Anxiety',
            'Breast Cancer',
            'COPD',
            'COVID-19',
            'Glaucoma',
            'Rheumatoid Arthritis',
            'Sickle Cell Anemia',
            'Type 2 Diabetes'
        ],
        'Precision': [0.90, 0.95, 0.94, 0.98, 0.98, 0.96, 0.98, 0.95],
        'Recall':    [0.96, 0.97, 0.91, 0.93, 0.93, 0.88, 0.85, 0.97],
        'F1-Score':  [0.93, 0.96, 0.92, 0.96, 0.96, 0.92, 0.91, 0.96]
    })

    st.subheader("Classification Report")

    st.dataframe(performance, use_container_width=True)

    fig_performance = px.bar(
        performance,
        x='Disease Category',
        y=['Precision', 'Recall', 'F1-Score'],
        barmode='group',
        title='Model Performance by Disease Category'
    )

    fig_performance.update_layout(
        xaxis_tickangle=-45,
        yaxis_title="Score",
        yaxis_range=[0, 1]
    )

    st.plotly_chart(
        fig_performance,
        use_container_width=True
    )
