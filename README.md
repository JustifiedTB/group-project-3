# group-project-3
OSU Bootcamp: Final Group Project

# Project Overview:
A terpene recommendation system that is designed to help users find a suitable terpene strain that fits their needs. 
- Users can search for a specific type of terpene or write a description of what they are looking for.
- The system will provide a list of five terpenes that best match the user's needs.
- The results include: terpene's name, a description of the terpene, a compound score that assesses the description, and a list of effects with a score that ranges from 0 to 1 based on how powerful the effects are in each terpene. 

# Data cleanup

# VADER:
- Vader was used to create a sentiment analysis based on the description column from the dataframe. 

- def analyze_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']
- terpene_data['sentiment_scores'] = terpene_data['description'].apply(analyze_sentiment)

the result was a compound score which is the overall sentiment of the description (positive and negative)

- a new compound score column was created and added to the dataframe

# Sentence transformer
- A Sentence Transformer model was used to generate embeddings for the description column
- final_data['embeddings'] = final_data['description'].apply(
    lambda x: sentence_model.encode(x, convert_to_tensor=True).cpu().numpy() if pd.notnull(x) else None
)
- the same model was used to generate embeddings for the user's input later in the recommendation function
sentence_model.encode([user_input], convert_to_tensor=True).cpu().numpy()
- the purpose of this model was to generate embeddings for both descripton and user input to match what the user is asking for.

# Langchain




















# Citations
## Leafly 
https://github.com/gugarosa/leaflyer

## Lab results 
https://github.com/MaxValue/Terpene-Profile-Parser-for-Cannabis-Strains
