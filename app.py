import streamlit as st
import pandas as pd
from pytrends.request import TrendReq
import plotly.express as px
import time
from cachetools import LRUCache
from pytrends.exceptions import ResponseError
import calendar
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Initialize Google Trends API client and cache
try:
    pytrend = TrendReq(hl='en-US', tz=300, geo='GLOBAL')
except Exception as e:
    st.error(f"Error initializing Google Trends API: {e}")
    st.stop()

# Cache to store already fetched data
cache = LRUCache(maxsize=128)

# Function to validate the query input
def validate_query(query):
    if not query:
        st.warning("Please enter a query to analyze.")
        return False
    elif not query.replace(" ", "").isalnum():
        st.warning("Invalid query. Query should contain only letters and numbers.")
        return False
    return True

# Fetch data from Google Trends API with input validation and retry mechanism
def fetch_data(query, timeframe='today 1-m', geo='GLOBAL', category=0, language='', search_type=''):
    print("Query:", query)
    print("Timeframe:", timeframe)
    print("Geo:", geo)
    print("Category:", category)
    print("Language:", language)
    print("Search Type:", search_type)
    if not validate_query(query):
        return None

    retries = 3  # Maximum number of retries
    for attempt in range(1, retries + 1):
        try:
            # Fetch data from Google Trends API
            if query in cache:
                st.info("Fetching data from cache...")
                return cache[query]
            st.info("Fetching data from Google Trends API...")

            # Validate search_type input
            allowed_search_types = ['web', 'images', 'news', 'youtube', 'froogle']
            if search_type not in allowed_search_types:
                st.warning("Invalid value for Search Type. Please enter one of the following: 'web', 'images', 'news', 'youtube', or 'froogle'")
                return None

            # Map search_type to corresponding gprop value
            gprop_mapping = {'web': '', 'images': 'images', 'news': 'news', 'youtube': 'youtube', 'froogle': 'froogle'}

            pytrend.build_payload([query], timeframe=timeframe, geo=geo, cat=category, gprop=gprop_mapping[search_type])
            data = pytrend.interest_over_time()
            cache[query] = data  # Cache the fetched data
            return data
        except ResponseError as e:
            st.error(f"Error fetching data from Google Trends API: {e}")
            if attempt < retries:
                st.info(f"Retrying after a delay... (Attempt {attempt}/{retries})")
                time.sleep(5)  # Retry after 5 seconds
            else:
                st.error(f"Failed to fetch data after {retries} attempts.")
                return None

# Function to provide information about the available options
def info_button(parameter, info_text):
    with st.expander(f"ℹ️ Info about {parameter}"):
        st.write(info_text)

# Select parameters for fetching data
def select_parameters():
    st.write("Select parameters for fetching data:")
    timeframe = st.selectbox("Timeframe", ["today 1-m", "today 3-m", "today 12-m"])
    geo = st.text_input("Geographical Region", "GLOBAL")
    category_info = "Categories available: Arts & Entertainment, Autos & Vehicles, Beauty & Fitness, Books & Literature, Business & Industrial, Computers & Electronics, Finance, Food & Drink, Games, Health, Hobbies & Leisure, Home & Garden, Internet & Telecom, Jobs & Education, Law & Government, News, Online Communities, People & Society, Pets & Animals, Real Estate, Reference, Science, Shopping, Sports, Travel"
    category = st.text_input("Category", "0")
    info_button("Category", category_info)
    language_info = "You can enter the complete language name (e.g., English, Spanish, French) or use language codes (e.g., en, es, fr)"
    language = st.text_input("Language", "")
    
    # Info text for search_type
    search_type_info = "Options available: 'web', 'news', 'images', 'youtube', 'froogle'"
    
    # Provide dropdown options for search_type
    search_type_options = ['web', 'images', 'news', 'youtube', 'froogle']
    search_type = st.selectbox("Search Type", search_type_options)
    
    info_button("Search Type", search_type_info)
    return timeframe, geo, category, language, search_type

# Callback for analyse button click
def analyze_query(query_input, custom_parameters=False, timeframe=None, geo=None, category=None, language=None, search_type=None):
    if not query_input:
        st.warning("Please enter a query to analyze.")
    else:
        if custom_parameters:
            # Fetch data and proceed with further actions using custom parameters
            data = fetch_data(query_input, timeframe, geo, category, language, search_type)
        else:
            # Use default parameters
            data = fetch_data(query_input)

        if data is not None:
            # Analyze the query and proceed with further actions
            st.success("Query analysis initiated. Please wait...")

            # Display the main analysis results (e.g., interest by region, interest over time)
            st.write("Main Analysis:")
            # Call functions to display main analysis results

            # Additional analysis
            st.write("Additional Analysis:")
            st.write("Choose additional analysis options:")
            options = {
                "Pattern Analysis": plot_pattern_analysis,
                "Correlation Analysis": plot_correlation_analysis,
                "Geographical Analysis": plot_geographical_analysis,
                "Long-term Trends": plot_long_term_trends,
                "Seasonal Trends": plot_seasonal_trends
            }
            selected_options = st.multiselect("Select analysis options", list(options.keys()))
            for option in selected_options:
                options[option](data, query_input)

            # Fetch related queries data
            related_queries_data = fetch_related_queries(query_input)  # Function to fetch related queries data

            # Display related queries data
            st.write("Related Queries:")
            st.table(related_queries_data)  # Display related queries data in a table

            # Copy option
            if st.button("Copy Related Queries"):
                related_queries_text = '\n'.join(related_queries_data)  # Convert related queries data to text
                st.write("Related Queries copied to clipboard:", related_queries_text)  # Display copied text


# Callback for interest by region
def show_interest_by_region(analyse, query_input):
    if analyse:
        st.write("Interest By Region")
        region_df = fetch_data(query_input)
        if region_df is not None:
            # Plotting with Plotly
            fig = px.bar(region_df, x=region_df.index, y=query_input, labels={'index': 'Region'})
            fig.update_layout(title="Interest By Region", xaxis_title="Region", yaxis_title="Interest")
            st.plotly_chart(fig)

# Callback for interest over time
def show_interest_over_time(analyse, query_input):
    if analyse:
        st.write("Interest Over Time")
        time_df = fetch_data(query_input)
        if time_df is not None:
            fig = px.line(time_df, x=time_df.index, y=query_input, labels={'index': 'Date', query_input: 'Interest'})
            fig.update_layout(title="Interest Over Time", xaxis_title="Date", yaxis_title="Interest")
            st.plotly_chart(fig)

# Display trending searches and top charts regardless of API status
def show_trending_searches(query_input):
    st.write("Trending Searches")
    trending_today = fetch_data(query_input)
    if trending_today is not None:
        st.dataframe(trending_today.head(20))

# Initialize Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Function to preprocess text
def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word.lower() for word in tokens]
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Streamlit app layout
def text_analysis_ui():
    st.title('Text Preprocessing and Sentiment Analysis')

    # Text input
    text = st.text_area('Enter text for analysis')

    # Perform preprocessing and sentiment analysis when button is clicked
    if st.button('Analyze Text'):
        tokens = preprocess_text(text)
        st.write('Preprocessed Tokens:', tokens)

        sentiment_score = sia.polarity_scores(text)
        st.write(f'Sentiment Score: {sentiment_score}')
        if sentiment_score['compound'] >= 0.05:
            st.success('Positive Sentiment')
        elif sentiment_score['compound'] <= -0.05:
            st.error('Negative Sentiment')
        else:
            st.info('Neutral Sentiment')

        word_freq = nltk.FreqDist(tokens)
        df_word_freq = pd.DataFrame(list(word_freq.items()), columns=['Word', 'Frequency'])
        df_word_freq = df_word_freq.sort_values(by='Frequency', ascending=False)

        st.write('Top 10 Most Frequent Words:')
        st.dataframe(df_word_freq.head(10))

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_word_freq.head(10), x='Frequency', y='Word')
        plt.title('Top 10 Most Frequent Words')
        plt.xlabel('Frequency')
        plt.ylabel('Word')
        st.pyplot(plt.gcf())

# Seasonal Trends Function
def plot_seasonal_trends(data, query):
    # Code for seasonal trends analysis...
    pass

# Long-term Trends Function
def plot_long_term_trends(data, query):
    # Code for long-term trends analysis...
    pass

# Geographical Analysis Function
def plot_geographical_analysis(data, query):
    # Code for geographical analysis...
    pass

# Correlation Analysis Function
def plot_correlation_analysis(data, query):
    # Code for correlation analysis...
    pass

# Pattern Analysis Function
def plot_pattern_analysis(data, query):
    # Code for pattern analysis...
    pass

# Token Segmentation Function
def plot_token_segmentation(data, query, queries=None):
    # Code for token segmentation analysis...
    pass

# Register the synonyms extension for spaCy tokens if it doesn't exist
if "synonyms" not in dir(spacy.tokens.Token):
    def get_synonyms(token):
        synonyms = []  
        for syn in wordnet.synsets(token.text):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        return synonyms

    spacy.tokens.Token.set_extension("synonyms", getter=get_synonyms, force=True)

# Load the English language model provided by spaCy
nlp = spacy.load("en_core_web_sm")

# Function to extract semantic relationships
def extract_semantic_relationships(text):
    doc = nlp(text)
    relationships = []
    for token in doc:
        if token._.synonyms:  
            synonyms = [syn for syn in token._.synonyms]
        else:
            synonyms = []  
        relationships.append({"Token": token.text, "Synonyms": synonyms})
    return relationships

# Streamlit app layout for semantic analysis
def semantic_analysis_ui():
    st.title('Semantic Analysis')
    text_semantic = st.text_area('Enter text for semantic analysis')

    # Perform semantic analysis when button is clicked
    if st.button('Analyze Semantics'):
        semantic_relationships = extract_semantic_relationships(text_semantic)
        st.write("Semantic Relationships:")
        st.write(pd.DataFrame(semantic_relationships))

# Main function to run the Streamlit app
def main():
    css = """
    <style>
    /* Add your CSS styling here */
    body {
        background-color: #f0f2f6; /* Light grey background */
        color: #333; /* Dark grey text */
    }

    h1 {
        color: #007bff; /* Blue heading */
    }

    .custom-container {
        margin: 2rem;
    }

    button {
        background-color: #007bff; /* Blue button background */
        color: white; /* White button text */
        border-radius: 5px;
        padding: 10px 20px;
        cursor: pointer;
        border: none;
    }

    button:hover {
        background-color: #0056b3; /* Darker blue on hover */
    }

    a {
        color: #007bff; /* Blue hyperlink text */
        text-decoration: underline; /* Underline hyperlink */
    }

    a:hover {
        color: #0056b3; /* Darker blue on hover */
    }
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)

    with st.container():
        st.title("Google Search Query Analysis")

    query_input = st.text_input("Query")
    custom_parameters = st.checkbox("Custom Parameters")
    analyse = st.button("Analyse")

    # Initialize default values for parameters
    timeframe = None
    geo = None
    category = None
    language = None
    search_type = None

    if custom_parameters:
        timeframe, geo, category, language, search_type = select_parameters()

    # Pass default values for parameters to analyze_query function
    analyze_query(query_input, custom_parameters, timeframe, geo, category, language, search_type)
    show_interest_by_region(analyse, query_input)
    show_interest_over_time(analyse, query_input)
    show_trending_searches(query_input)
    text_analysis_ui()
    semantic_analysis_ui()

# Run the Streamlit app
if __name__ == "__main__":
    main()
