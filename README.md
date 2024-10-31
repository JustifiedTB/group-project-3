# group-project-3
OSU Bootcamp: Final Group Project

# Project Overview:
A cannabis recommendation system that is designed to help users find the right profile of terpenes and cannabinoids to fit their needs. 
- Users can search for a specific type of terpene or write a description of what they are looking for: a strain or terpene they already enjoy, a particular aroma or flavor they like, an illness they are suffering, etc.
- The system will provide a list of five strains that best match the user's needs.
- The results include: strain's name, a description of the strain, a compound score that assesses the description's sentiment, and a list of effects with a score that ranges from 0 to 1 based on how powerful the effects are in each strain.

# Data Acquisition and Cleaning
We worked with two datasets in this project. The first was a Github compilation of cannabis product testing results from three different lab sources. The second was a Kaggle dataset sourced from Leafly and contained cannabis strain data including user feedback scores showing the percentage applicability of the strain across various effects and illnesses that can be ameliorated. The goal was to join these two datasets and develop a neural network model to predict the effects or illnesses scores based on the strain's tested values. This model would then be integrated into a language model based on the strain descriptions to provide further context based on empirical test data. The further hope would be that these user effects could be predicted in new strains based on initial test findings.



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
