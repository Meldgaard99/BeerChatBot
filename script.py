import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os



class BeerChatbot:
    def __init__(self, file_path):
        """Initialize the chatbot with data and preprocessing."""
        self.data = pd.read_csv(file_path)
        self.conversation = []
        self.sentiment_scores = []
        self.user_inputs = []
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stopwords = set(stopwords.words('english'))
        self.stopwords.update(['beer', 'want', 'like',])
        self.vectorizer = TfidfVectorizer()
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        # Preprocess descriptions and add features
        self.data['processed_desc'] = self.data['Description'].apply(self.preprocess_text)
        self.data['entities'] = self.data['Description'].apply(self.extract_entities)
        self.train_ml_models()

    def preprocess_text(self, text):
        """Preprocesses text by removing stopwords, converting to lowercase,
               and applying stemming and lemmatization to normalize input."""
        if not isinstance(text, str):  return ''
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = self.tokenizer.tokenize(text)
        tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stopwords]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)


    def extract_entities(self, text):
        """Extracts beer-related entities (types) mentioned in the text
               based on a predefined list and dynamically added types from the dataset."""
        # Predefined styles list with additional styles
        if not hasattr(self, 'beer_styles'):
            self.beer_styles = {'ipa', 'lager', 'stout', 'porter', 'ale', 'pilsner', 'saison',
                                'amber', 'bock', 'brown ale', 'dubbel', 'tripel', 'quad', 'gose',
                                'wheat', 'hefeweizen', 'kolsch', 'pale ale', 'blonde ale',
                                'cream ale', 'barleywine', 'scotch ale', 'red ale', 'golden ale',
                                'rye', 'imperial stout', 'black ipa', 'session ipa', 'hazy ipa',
                                'fruit beer', 'sour', 'wild ale', 'milk stout', 'belgian strong ale'}
            # Add dynamic styles from dataset
            dynamic_styles = set(self.data['Style'].dropna().str.lower().unique())
            self.beer_styles.update(dynamic_styles)

        text = text.lower()
        entities = [style for style in self.beer_styles if style in text]
        return ', '.join(entities)
    def analyze_sentiment(self, text):
        """Analyze sentiment with TextBlob."""
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity
        self.sentiment_scores.append(sentiment_score)

        if 'bad' in text.lower() or 'worst' in text.lower():
            return "I see you're looking for beers that aren't rated highly. Here's what I found: "
        elif sentiment_score > 0.5:
            return "I'm glad you're so enthusiastic about beer! "
        elif sentiment_score < -0.3:
            return "I notice you seem a bit frustrated. Let me try to find better recommendations. "
        return ""

    def display_sentiment_summary(self):
        """Display summary of sentiment analysis from the conversation."""
        if not self.sentiment_scores:
            print("\nNo conversation data to analyze.")
            return

        avg_sentiment = sum(self.sentiment_scores) / len(self.sentiment_scores)
        print("\nSentiment Analysis Summary:")
        print(f"Average sentiment score: {avg_sentiment:.2f}")
        print(f"Most positive moment: {max(self.sentiment_scores):.2f}")
        print(f"Most negative moment: {min(self.sentiment_scores):.2f}")

        # Plot sentiment progression
        plt.figure(figsize=(10, 5))
        plt.plot(self.sentiment_scores, marker='o')
        plt.title('Conversation Sentiment Progression')
        plt.xlabel('Message Number')
        plt.ylabel('Sentiment Score')
        plt.grid(True)
        plt.show()

    def generate_wordcloud(self):
        """Generate and display wordcloud from conversation."""
        if not self.user_inputs:
            print("\nNo conversation data for wordcloud.")
            return

        # Combine all user inputs
        text = ' '.join(self.user_inputs)


        wordcloud = WordCloud(width=800, height=400,
                              background_color='white',
                              stopwords=self.stopwords,
                              min_font_size=10).generate(text)

        # Showing cloudwords
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('WordCloud of Conversation Topics')
        plt.show()

    def elbow_analysis(self, max_clusters=10):
        """Perform elbow analysis"""
        X = self.vectorizer.fit_transform(self.data['processed_desc'])
        wcss = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)

        # Calculate the dynamic optimal_k
        x_points = range(1, max_clusters + 1)
        y_points = np.array(wcss)
        slopes = np.diff(y_points) / np.diff(x_points)  # Calculate slopes between points
        second_derivative = np.diff(slopes)
        optimal_k = np.argmin(second_derivative) + 2

        return optimal_k

    def train_ml_models(self):
        """Trains machine learning models:
            1. K-Means for clustering beer descriptions into groups.
            2. Random Forest for classification and prediction of clusters."""
        X = self.vectorizer.fit_transform(self.data['processed_desc'])
        # Perform elbow analysis and get dynamic k
        optimal_k = self.elbow_analysis(max_clusters=10)
        print(f"Optimal k dynamically calculated: {optimal_k}")

        # Train K-Means
        self.kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        self.data['cluster'] = self.kmeans.fit_predict(X)


        # Train Random Forest for classification
        X_train, X_test, y_train, y_test = train_test_split(X, self.data['cluster'], test_size=0.2, random_state=42)
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(X_train, y_train)

        # Evaluate performance
        y_pred = self.rf_model.predict(X_test)
        print("Model Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    def recommend_beers(self, user_input):
        """Recommends beers based on user input by:
            - Predicting clusters using the Random Forest model.
            - Filtering and ranking beers based on similarity and input preferences."""
        processed_input = self.preprocess_text(user_input)
        input_vector = self.vectorizer.transform([processed_input])

        # Predict cluster
        cluster = self.rf_model.predict(input_vector)[0]
        cluster_data = self.data[self.data['cluster'] == cluster]

        # Filter for negative sentiment (bad beers)
        if 'bad' in user_input.lower() or 'worst' in user_input.lower():
            cluster_data = self.data[self.data['review_overall'] < 2.5]

        # Sort recommendations by similarity
        similarities = cluster_data['processed_desc'].apply(lambda x: self.similarity(processed_input, x))
        recommendations = cluster_data.loc[similarities.nlargest(3).index]

        # Generate explanations
        explanations = []
        for _, beer in recommendations.iterrows():
            reason = []
            if 'bitter' in user_input.lower() and beer['Bitter'] > 50:
                reason.append("high bitterness level")
            if 'sweet' in user_input.lower() and beer['Sweet'] > 50:
                reason.append("sweet flavor profile")
            if 'low alcohol' in user_input.lower() and beer['ABV'] < 5.0:
                reason.append("low alcohol content")
            if not reason:
                reason.append("similar description and style")

            explanation = (f"- {beer['Name']} ({beer['Style']}) - ABV: {beer['ABV']}% | "
                           f"IBU: {beer['Min IBU']}-{beer['Max IBU']} | Overall Rating: {beer['review_overall']:.2f} "
                           f"- Recommended because of: {', '.join(reason)}")
            explanations.append(explanation)

        return explanations

    def similarity(self, text1, text2):
        """Simple similarity metric using TF-IDF."""
        vec1 = self.vectorizer.transform([text1])
        vec2 = self.vectorizer.transform([text2])
        return (vec1 * vec2.T).toarray()[0][0]

    def chat(self):
        """Chatbot loop."""
        print("Welcome to the Beer Recommendation Chatbot!")
        print("Tell me what kind of beer you're looking for (or type 'exit' to quit)")

        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                break

            self.user_inputs.append(user_input)  # Store user input for wordcloud
            sentiment_response = self.analyze_sentiment(user_input)
            if sentiment_response:
                print(sentiment_response)

            recommendations = self.recommend_beers(user_input)
            for rec in recommendations:
                print(rec)

        # Display analysis after chat ends
        self.display_sentiment_summary()
        self.generate_wordcloud()



#Link til datasættet hvis der er problemer - https://www.kaggle.com/datasets/ruthgn/beer-profile-and-ratings-data-set?select=beer_profile_and_ratings.csv
if __name__ == "__main__":
    # Find dynamisk sti til data-mappen
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    file_path = os.path.join(data_dir, "beer_profile_and_ratings.csv")

    # Tjek om filen findes
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        exit(1)

    chatbot = BeerChatbot(file_path)
    chatbot.chat()