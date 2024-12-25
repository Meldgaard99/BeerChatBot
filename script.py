import json
import re
from collections import Counter
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud


class BeerChatbot:
    def __init__(self, file_path):
        """Initialize the chatbot with data and necessary preprocessing"""
        self.data = pd.read_csv(file_path)
        self.conversation = []
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stopwords = set(stopwords.words('english'))

        # Add beer-specific stopwords
        self.stopwords.update(['beer', 'ale', 'want', 'like', 'good'])

    def preprocess_text(self, text):
        """Enhanced text preprocessing with better handling of beer-related terms"""
        if not isinstance(text, str):
            return []

        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = self.tokenizer.tokenize(text)
        return [word for word in tokens if word not in self.stopwords]

    def find_beer(self, preferences):
        """Improved beer matching algorithm with weighted scoring"""
        matches = []

        # Convert preferences to set for faster lookup
        pref_set = set(preferences)

        for index, beer in self.data.iterrows():
            description = set(self.preprocess_text(beer['Description']))
            style = set(self.preprocess_text(beer['Style']))

            # Weighted scoring system
            score = (
                    len(description & pref_set) * 2 +  # Description matches count more
                    len(style & pref_set) * 3 +  # Style matches count even more
                    self._match_abv(beer['ABV'], preferences) +
                    self._match_ibu(beer['Min IBU'], beer['Max IBU'], preferences)
            )

            if score > 0:
                matches.append({
                    'beer': beer,
                    'score': score,
                    'matched_terms': list(description & pref_set) + list(style & pref_set)
                })

        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:3]

    def _match_abv(self, abv, preferences):
        """Helper method to match ABV preferences"""
        if any(term in preferences for term in ['strong', 'heavy']):
            return 2 if abv > 7.0 else 0
        if any(term in preferences for term in ['light', 'weak']):
            return 2 if abv < 5.0 else 0
        return 0

    def _match_ibu(self, min_ibu, max_ibu, preferences):
        """Helper method to match bitterness preferences"""
        if any(term in preferences for term in ['bitter', 'hoppy']):
            return 2 if max_ibu > 50 else 0
        if any(term in preferences for term in ['smooth', 'mild']):
            return 2 if max_ibu < 30 else 0
        return 0

    def format_beer_recommendation(self, match):
        """Format beer recommendation with detailed information"""
        beer = match['beer']
        return (
            f"- {beer['Name']} ({beer['Style']})\n"
            f"  ABV: {beer['ABV']}% | IBU: {beer['Min IBU']}-{beer['Max IBU']}\n"
            f"  Matched terms: {', '.join(match['matched_terms'])}\n"
            f"  Description: {beer['Description'][:100]}..."
        )

    def chat(self):
        """Main chatbot interaction loop with improved user experience"""
        print("Welcome to the Beer Recommendation Chatbot!")
        print("Tell me what kind of beer you're looking for (or type 'exit' to quit)")
        print("Examples: 'I want a fruity and smooth beer' or 'Looking for something strong and bitter'")

        while True:
            user_input = input("\nYou: ").strip()

            if not user_input:
                print("Please tell me what kind of beer you're looking for.")
                continue

            if user_input.lower() in ['exit', 'quit']:
                self._show_summary()
                break

            self.conversation.append(user_input)
            preferences = self.preprocess_text(user_input)

            if not preferences:
                print("I couldn't understand your preferences. Please try describing the beer you want.")
                continue

            recommendations = self.find_beer(preferences)

            if recommendations:
                print("\nBased on your preferences, here are some beers you might enjoy:")
                for match in recommendations:
                    print("\n" + self.format_beer_recommendation(match))
            else:
                print(
                    "I couldn't find any beers matching your preferences. Try describing the taste, strength, or style you're looking for.")

    def _show_summary(self):
        """Show conversation summary with improved visualization"""
        print("\nThanks for chatting! Here's a summary of our conversation:")
        words = []
        for message in self.conversation:
            words.extend(self.preprocess_text(message))

        if words:
            counter = Counter(words)
            print("\nMost common preferences:")
            for word, count in counter.most_common(5):
                print(f"- {word}: {count} times")

            self._generate_wordcloud(' '.join(self.conversation))

    def _generate_wordcloud(self, text):
        """Generate and display word cloud with better styling"""
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=50
        ).generate(text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title("Your Beer Preferences Visualization")
        plt.show()


if __name__ == "__main__":
    file_path = '/Users/rasmusjensen/PycharmProjects/BeerChatBot/data/beer_profile_and_ratings.csv'  # Update with your file path
    chatbot = BeerChatbot(file_path)
    chatbot.chat()