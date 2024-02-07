"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')


# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Home", "Recommender System","Solution Overview", "Project Summary", "Explore The Data", "About Us", "The Team", "Contact Us"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    st.sidebar.image("Pictures//Phantom.png")
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    st.sidebar.write('Welcome to our Streamlit App, proudly developed my Metaphase Analytics Consortium. Please feel free to browse the contents of our app through the tab above. For more information, please find our contacts below.')
    st.sidebar.subheader('Company Details:')
    st.sidebar.write('**Email:** *phantomai@gmail.com*')
    st.sidebar.write('**Telephone Number:** *011 367 8801*')
    st.sidebar.write('**Web Address:** *www.phantomai.co.za*')
    
    if page_selection == "Home":
        st.title("Tesla Cinema Experience: AI-Enhanced Streaming Platform for Autonomous Driving")
        st.title("Welcome!")
        st.image('Pictures/Tesla logo.webp')


    if page_selection == "Explore The Data":
        st.title('Exploratory Data Analysis')
        st.image("Pictures//EDA Cover.webp")
        st.header("Data Description")
        st.subheader("Data Overview")
        st.write("This dataset consists of several million 5-star ratings obtained from users of the online MovieLens movie recommendation service. The MovieLens dataset has long been used by industry and academic researchers to improve the performance of explicitly-based recommender systems, and now you get to as well!")
        st.write("For this Predict, we'll be using a special version of the MovieLens dataset which has enriched with additional data, and resampled for fair evaluation purposes.")
        st.subheader("Source")
        st.write("The data for the MovieLens dataset is maintained by the GroupLens research group in the Department of Computer Science and Engineering at the University of Minnesota. Additional movie content data was legally scraped from IMDB.")
        st.subheader("Supplied Files")
        st.write("- genome_scores.csv - a score mapping the strength between movies and tag-related properties")
        st.write("- genome_tags.csv - user assigned tags for genome-related scores")
        st.write("- imdb_data.csv - Additional movie metadata scraped from IMDB using the links.csv file.")
        st.write("- links.csv - File providing a mapping between a MovieLens ID and associated IMDB and TMDB IDs.")
        st.write("- sample_submission.csv - Sample of the submission format for the hackathon.")
        st.write("- tags.csv - User assigned for the movies within the dataset.")
        st.write("- test.csv - The test split of the dataset. Contains user and movie IDs with no rating data.")
        st.write("- train.csv - The training split of the dataset. Contains user and movie IDs with associated rating data.")
        st.header("Data Visualisation")
        st.subheader("Movie Ratings")
        st.image("Pictures//Movie Rating.png")
        st.write("From the above statistic and graphic, we can see that high percentage movie viewers tend to rate movies high. The mean average rating for all the movies is 3.5 while the modal rating(most frequently occuring) rating is 4.0.")
        st.image("Pictures//Movie Rating Distribution.png")
        st.write("Remember , the Movies data set contains 62423 unique movies . The analysis above shows that 14210 (23%) of the movies were not rated , 12537 (20%) was rated just once , 26165 (42%) was rated between 2 to 50 times , 2223 (4%) movies was rated 51 to 100 times , 5171 (8%) was rated 101 to 1000 times and 2117 (4%) movies was rated more than 1000 times. Therefore a large chunk of movies was not rated at all by the users.This would cause some problem for our collaborative based recommendation system")
        st.image("Pictures//Movies Rated.png")
        st.write("The visual on shows the top 10 most rated movies, with shawshank redemption as the most rated movie followed by forest gump. Although this most a the most rated, this does not neccessarily means they are the best movies.")
        st.subheader("Genres")
        st.image("Pictures//Genres.png")
        st.write("Drama is the most commonly occuring genre with almost half of the movies identifying itself as a drama film. Comedy comes in at a distant second with 25% of the movies having adequate doses of humor. Other major genres in the top 10 are Thriller, Romance,Action, Horror, Documentary and Crime.")
        st.image("Pictures//Tags.png")
        st.write("Sci-fi is the most popular tag while classic has the least number of counts.")
        st.subheader("Movie Run-time")
        st.image("Pictures//Runtime.png")
        st.write("The average length of a movie is about 1 hour 40 minutes long. The longest movie on record in this dataset is a staggering 877 minutes(approximately 15 hours long)")
        st.subheader("Production Year")
        st.image("Pictures//Production Year.png")



    if page_selection == "About Us":
        st.title("Welcome to Phantom AI")
        st.subheader("Slogan")
        st.image("Pictures//Cover Phantom.png")
        st.header("Who We Are")
        st.subheader("The Company")
        st.write("We were first established in the year 1923, in the capital of South Africa. Our founders began the company in the city of Johannesburg, and since then, it has grown and soared in the industries it operates in. By the year 2012, Phantom AI was the leading technology company on the African Continent, and is in the top 10 global companies in terms of revenue and the quality of services it offers.")
        st.subheader("Mission Statement")
        st.write("Enhancing convenience and enriching experiences.ing convenience and enriching experiences.")
        st.subheader("Vision")
        st.write("A data-driven culture integrated advanced analytics and aims to drive transformative outcomes.")
        st.subheader("Values")
        st.write("- Innovation")
        st.write("- Sustainability")
        st.write("- Ethical use of technology")
        st.subheader("Our Services")
        st.write("- Artificial Intelligence and Machine Learning")
        st.write("- Data and Marketing Analyitics")
        st.write("- Consulting Solutions")


    if page_selection == "The Team":
        st.title("Our Project Team")
        st.image("Pictures//Team.jpg")
        st.header("Introducing The Team")
        st.write("The project team is composed of 6 members, who have been working day in day out to provide the solutions that Tesla requires. Their efforts have resulted in the success of achieving deliverables that build-up into solving the problem as a whole. They provided excellent expertise and knowledge, and used their skills to develop reliable and valuable recommender systems.")
        st.header("The Team Members")
        st.subheader("Boitemogelo Tagane: CEO  ")
        st.image("Pictures//Temo.jpg")
        st.write("Boitemogelo is the CEO of Phantom AI, and has been with the company for over 10 years. He rose through the ranks of the organisation and has led the company to great success. He is a graduate at the University of Cape Town, having a Masters Degree in Computer Science specialising in Machine Learning")
        st.subheader("Kedijang Setsome: Project Manager ")
        st.image("Pictures//Kedijang.jpg")
        st.write("Kedijang is a Project Manager at Phantom AI and has been with the company for close to a decade. She is very competent and reliable, and is the best Project Manager the company has had. She graduated from the University of North West with a Masters Degree in Information Technology. She has gone through various roles in the company, ad was promoted to Project Manager a couple years ago")
        st.subheader("Anthonia Omonayin: Developer  ")
        st.write("Tonia is a Senior Developer at Phantom AI. She is one of the best in her roles, and has been with the company for almost a decade. She is a PHD holder in Computer Science, having graduated from the University of Lagos. Her technical knowledge and experience has led to great advancements and achievements within the organisation.")
        st.subheader("Edidiong Udofia: Developer ")
        st.image("Pictures//Edicool.jpg")
        st.write("Edidiong is also Senior Developer at Phantom AI. He has a vast amount of technical knowledge that has been very resourceful for the past 8 years that he has been with the company. He is a Masters Degree holder in Computer Science from Federal University of Technology. He also has Machine Learning and Data Engineering certificates from several institutions. His skills have been a driving success for the company.")
        st.subheader("Priscilla Vhafuniwa: Data Scientist ")
        st.image("Pictures//Pricilla.jpg")
        st.write("Pricilla has been with the organisation over 5 years and has good contributions in her role with the company. She is a graduate from University of Pretoria with an Honours Degree in Information Systems. She is also a graduate as a Data Scientist from Explore AI Academy, and that has been her occupation at Phantom AI since her graduation. ")
        st.subheader("Winnie Mmari: Data Scientist ")
        st.image("Pictures//Winnie.jpg")
        st.write("Winnie is a Data Scientist at Phantom AI. She is a Graduate at the University of Technology, with a Masters Degree in Information Technology, as well as being a graduate at Explore AI as a Data Scientist. WIth over 5 years working at the company, Winnie has developed to be one of the most competent employees in the company, and has been of great value to achieving the company’s KPIs")
    

    if page_selection == "Contact Us":
        st.title("Contact US")
        st.image("Pictures//Contact Us.jpg")
        st.markdown('    ')
        st.subheader('Our Contact Details:')
        st.markdown(' - **Tel**: 012 367 8801')
        st.write("- **Email**: phantomai@gmail.com")
        st.write("- **Website**: www.phantomai.co.za")
        st.markdown('- **Twitter**: @PhantomAI')
        st.markdown('- **Address**: 11 Adriana Cres, Rooihuiskraal, Centurion, 0154')

    if page_selection == "Project Summary":
        st.title("Movie Recommender Engine")
        st.subheader("Project Overview")
        st.image('Pictures//Tesla.webp')
        st.markdown("As Phantom AI, we were approached by Tesla to work on a project that will assist them with one of their biggest innovations, and enabling to penetrate the film and entertainment industry. Tesla is one of the biggest companies in the world and was founded by the richest man alive. It's aim is to create highly innovative, sophisticated vehicles that are intelligent, safe, and have a less negative effect on the environment, as their vehicles release less carboon footprint compared to gasoline cars. However, their biggest innovation is the development of self driven cars. This innovation has given them the opportunity to penetrate certain market segments, allowing them to increase their reach and have a wider market. In a nutshell, they want to bring the cinema... to your vehicles. They want to create a software that can be installed in their vehicles that will serve as a streaming platform for movies and series. This is convenient for the driver when he/she switches to auto-pilot so that the car drives itself, and the driver can relax and catch-up on his/her favourite shows.")


    
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.image('Pictures/Solution.jpg')
        st.header("The Solution")
        st.write("The team from Phantom AI followed thorough and meticulous processes to ensure they got the desired result. They collaborated on Github, and used resources such as Google Colab, VS Code, Kaggle, Anaconda and the Jupyter Notebook Environment. With these tools they have been able to establish elaborative and clear Exploratory Data Analysis that is easily comprehensive and illustrates the data perfectly. Furthermore they have developed a function that has the capacity to recommend similar movies based on the movie title that was inputted.")
        st.write("Lastly, the team developed machine learning models that have the capacity to recommend movies that are similar to each other based on the title/s that were put in. These models have reliable algorithms, but other models performed much better than others. We have concluded that the most reliable and best performing model was the SVD model, and that is the model which we integrated to make recommendations for the user on the app.")

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
