Welcome to ICT!

This is a simple crypto analysis and forecasting app 

To Run the app open terminal and type this command (streamlit run [location of Main.py file].Main.py)



Info:
The structure is like this:
Crypto/
│
├── Read_me.txt
├── Main.py
├── pages/
│   ├── Correlation.py   correlation page for streamlit 
│   ├── EDA.py   Exploratory Data Analysis page for streamlit 
│   ├── Models_Extra.py   modern models page for streamlit
│   ├── Models.py   main 4 models page for streamlit 
│   ├── df_clusters.csv   clusters data frame saved from AE2 
│   ├── models_extra/  the saved pre-trained models for N-BEATS and Transformer 
│   ├── models/  the saved models for LSTM and models.py (functions for running the models)
│
├── additional files/         
│   ├── Jupyter notebooks containing the code used for training and testing models 
