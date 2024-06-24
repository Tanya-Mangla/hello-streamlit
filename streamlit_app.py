import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Ensure you have the NLTK data
nltk.download('vader_lexicon')

# Function for authenticating user
def authenticate(username, password):
    return username in st.session_state['users'] and st.session_state['users'][username] == password

# Function for sign-up (save user credentials)
def signup(username, password):
    st.session_state['users'][username] = password

# Initialize session state for users if not already done
if 'users' not in st.session_state:
    st.session_state['users'] = {}

# Initialize session state for login status
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# Login/Sign-Up form
if not st.session_state['logged_in']:
    st.title("Welcome to the Data Analysis App")
    choice = st.selectbox("Login or Sign Up", ["Login", "Sign Up"])

    if choice == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate(username, password):
                st.success("Logged In Successfully")
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
            else:
                st.error("Invalid Username or Password")

    elif choice == "Sign Up":
        username = st.text_input("New Username")
        password = st.text_input("New Password", type="password")
        if st.button("Sign Up"):
            if username in st.session_state['users']:
                st.error("Username already exists")
            else:
                signup(username, password)
                st.success("User Registered Successfully")

if st.session_state['logged_in']:
    sia = SentimentIntensityAnalyzer()

    def analyze_text(text, analysis_type):
        scores = sia.polarity_scores(text)
        if analysis_type == 'Sentiment Analysis':
            if scores['compound'] >= 0.05:
                sentiment = 'Positive'
            elif scores['compound'] <= -0.05:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'
            return sentiment
        elif analysis_type == 'Emotion Analysis':
            if scores['compound'] >= 0.6:
                emotion = 'Very Good'
            elif 0.2 <= scores['compound'] < 0.6:
                emotion = 'Good'
            elif -0.2 <= scores['compound'] < 0.2:
                emotion = 'Neutral'
            elif -0.6 <= scores['compound'] < -0.2:
                emotion = 'Bad'
            else:
                emotion = 'Very Bad'
            return emotion

    def display_dashboard(data):
        st.subheader('Analysis Results')
        st.write(data)

        st.subheader('Statistics')
        st.write(data.describe())

        st.subheader('Charts')
        fig, ax = plt.subplots()
        data['analysis'].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)

        st.subheader('Download Results')
        # Ensure all data is converted to string before saving
        data = data.astype(str)
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(label='Download CSV', data=csv, file_name='results.csv', mime='text/csv')

    def main():
        st.title('Text Analysis App')

        # About Section
        st.sidebar.subheader('About')
        st.sidebar.info(
            "This is a text analysis app that allows you to perform sentiment analysis "
            "and emotion analysis on text data. You can choose the analysis type and "
            "upload text either manually or via a CSV file. After analysis, the results "
            "will be displayed along with statistical information and visualizations where efforts made by Tanya Mangla"
        )
        st.sidebar.subheader('How to Use')
        st.sidebar.markdown(
            """
            - Choose the analysis type from the sidebar options.
            - Select either 'Text Input' to manually enter text or 'CSV File' to upload a CSV file.
            - For 'Text Input', type or paste the text into the text area and click 'Analyze'.
            - For 'CSV File', upload a CSV file containing a 'text' column for analysis.
            - After analysis, explore the results, statistics, and charts.
            - You can also download the results as a CSV file.

                         Made By: TANYA MANGLA
            """
        )

        st.sidebar.title('Options')
        analysis_type = st.sidebar.selectbox('Choose Analysis Type', ['Sentiment Analysis', 'Emotion Analysis'])
        upload_type = st.sidebar.selectbox('Upload Type', ['Text Input', 'CSV File'])

        if upload_type == 'Text Input':
            user_input = st.text_area('Enter text for analysis')
            if st.button('Analyze'):
                if user_input:
                    result = analyze_text(user_input, analysis_type)
                    result_df = pd.DataFrame([{'text': user_input, 'analysis': result}])
                    display_dashboard(result_df)
                else:
                    st.error('Please enter text to analyze')
        elif upload_type == 'CSV File':
            uploaded_file = st.file_uploader('Upload a CSV file', type='csv')
            if uploaded_file:
                data = pd.read_csv(uploaded_file)
                if 'text' in data.columns:
                    # Fill missing values in 'text' column with empty strings
                    data['text'] = data['text'].fillna('')
                    data['analysis'] = data['text'].apply(lambda x: analyze_text(x, analysis_type))
                    display_dashboard(data)

                    # Show Dataset
                    if st.checkbox("Preview Dataset"):
                        if st.button("Head"):
                            st.write(data.head())
                        if st.button("Tail"):
                            st.write(data.tail())
                        if st.button("Information"):
                            st.write(data.info())
                        if st.button("Shape"):
                            st.write(data.shape)
                        if st.button("Describe"):
                            st.write(data.describe())

                    # Check DataType of Each Column
                    if st.checkbox("DataType of Each Column"):
                        st.text("DataTypes")
                        st.write(data.dtypes)

                    # Find Shape of Our Dataset (Number of Rows And Number of Columns)
                    data_shape = st.radio("What Dimension Do You Want To Check?", ('Rows', 'Columns'))
                    if data_shape == 'Rows':
                        st.text("Number of Rows")
                        st.write(data.shape[0])
                    elif data_shape == 'Columns':
                        st.text("Number of Columns")
                        st.write(data.shape[1])

                    # Find Null Values in The Dataset and Handle Them
                    if data.isnull().values.any():
                        st.warning("This Dataset Contains Some Null Values")
                        handle_null = st.selectbox("Do You Want to Remove or Fill Null Values?", ("Select One", "Remove", "Fill"))
                        if handle_null == "Remove":
                            data = data.dropna()
                            st.text("Null Values are Removed")
                        elif handle_null == "Fill":
                            fill_value = st.text_input("Enter the value to replace null values:")
                            if st.button("Fill Null Values"):
                                data = data.fillna(fill_value)
                                st.text("Null Values are Filled")
                    else:
                        st.success("No Missing Values")
                else:
                    st.error('CSV file must contain a "text" column')

    main()
