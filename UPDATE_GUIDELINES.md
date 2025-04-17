# These are the update guidelines for integrating XGBoost and RandomForest into the system
1. Create a FeatureEngineering class specifically for your Model (refer to beechunker/ml/feature_engineering.py)
2. Create a class similar to BeeChunkerSOM (beechunker/ml/som.py) for your model - for example BeeChunkerRandomForest, BeeChunkerXGBoost
3. Create joblib files for your model (including any preprocessing models you are using) and store it in the models/ directory
4. Inform Jayesh

## DEADLINE - 12:00PM 04/19/2024