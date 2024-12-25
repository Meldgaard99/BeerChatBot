import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.decomposition import PCA
import spacy

# Indlæs datasæt
beer_data = pd.read_csv('/Users/rasmusjensen/PycharmProjects/BeerChatBot/data/beer_profile_and_ratings.csv')
# 1. Fjern irrelevante kolonner
beer_data = beer_data.drop(['Beer Name (Full)', 'Description'], axis=1)

# 2. Håndtering af manglende værdier
# Tjek for manglende data
print(beer_data.isnull().sum())

# Erstat kun manglende værdier i numeriske kolonner med median
numeric_cols = beer_data.select_dtypes(include=['float64', 'int64']).columns
beer_data[numeric_cols] = beer_data[numeric_cols].fillna(beer_data[numeric_cols].median())

# 3. Kod kategoriske variabler (Label Encoding for 'Style')
le = LabelEncoder()
beer_data['Style'] = le.fit_transform(beer_data['Style'])

# 4. Skaler numeriske data
scaler = StandardScaler()
beer_data[numeric_cols] = scaler.fit_transform(beer_data[numeric_cols])

# 5. Gem det rensede datasæt
beer_data.to_csv('/Users/rasmusjensen/PycharmProjects/BeerChatBot/data/beer_profile_and_ratings.csv', index=False)
print("Data renset og gemt!")

# -------------------------------------------
# 6. Anbefalingssystem (K-means Clustering)
features = beer_data[numeric_cols]
kmeans = KMeans(n_clusters=5, random_state=42)
beer_data['Cluster'] = kmeans.fit_predict(features)
print("Clustering færdig!")

# -------------------------------------------
# 7. Klassifikation af Øl-stil (Random Forest)
X = beer_data[numeric_cols]
y = beer_data['Style']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print("Klassifikation rapport:")
print(classification_report(y_test, predictions))

# -------------------------------------------
# 8. Regression for ABV (Random Forest Regressor)
X = beer_data.drop(['ABV'], axis=1)
y = beer_data['ABV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
print("Regression MSE:", mean_squared_error(y_test, predictions))

# -------------------------------------------
# 9. Sentimentanalyse (NLP med spaCy)
nlp = spacy.load('en_core_web_sm')
beer_descriptions = pd.read_csv('/mnt/data/beer_profile_and_ratings.csv')['Description'].dropna()
sentiments = []
for desc in beer_descriptions:
    doc = nlp(desc)
    sentiment_score = sum([token.sentiment for token in doc])
    sentiments.append(sentiment_score)
print("Sentimentanalyse færdig!")

# -------------------------------------------
# 10. Chatbot-funktion
print("Velkommen til Øl-chatbotten! Skriv 'exit' for at afslutte.")
while True:
    user_input = input("Spørg mig om øl: ").lower()
    if user_input == 'exit':
        print("Farvel!")
        break
    elif 'anbefal' in user_input:
        cluster = int(input("Indtast cluster (0-4): "))
        anbefalinger = beer_data[beer_data['Cluster'] == cluster]['Brewery'].head(5)
        print("Anbefalede bryggerier:", anbefalinger.tolist())
    elif 'stil' in user_input:
        print("Forudsigelse af øl-stil er ikke tilgængelig direkte gennem chatten endnu.")
    elif 'alkohol' in user_input:
        values = [float(x) for x in input("Indtast egenskaber adskilt med komma: ").split(',')]
        values = scaler.transform([values])
        abv_pred = regressor.predict(values)
        print(f"Forudset alkoholprocent: {abv_pred[0]:.2f}%")
    elif 'sentiment' in user_input:
        desc = input("Indtast en anmeldelse: ")
        doc = nlp(desc)
        sentiment_score = sum([token.sentiment for token in doc])
        print(f"Sentimentscore: {sentiment_score}")
    else:
        print("Jeg forstår ikke spørgsmålet. Prøv igen.")

print("Alle analyser er færdige!")
