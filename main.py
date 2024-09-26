import streamlit as st
import pandas as pd #install package pip install -U scikit-learn
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

Test 
# Set the page configuration
st.set_page_config(
    page_title="My Streamlit App",
    layout="wide",
    page_icon=":guardsman:"
)

# Read the datasets into a pandas DataFrame
df_raw = pd.read_csv("kaggle_survey_2020_responses.csv", low_memory=False)
df = pd.read_csv("df_newcolumnnames.csv")
df_preprocessed = pd.read_csv("preprocessed_df.csv")

# Important variables
q7 = ['q7_most_freq_used_prgmg_language_Python',
      'q7_most_freq_used_prgmg_language_R',
      'q7_most_freq_used_prgmg_language_SQL',
      'q7_most_freq_used_prgmg_language_C',
      'q7_most_freq_used_prgmg_language_C++',
      'q7_most_freq_used_prgmg_language_Java',
      'q7_most_freq_used_prgmg_language_Javascript',
      'q7_most_freq_used_prgmg_language_Julia',
      'q7_most_freq_used_prgmg_language_Swift',
      'q7_most_freq_used_prgmg_language_Bash',
      'q7_most_freq_used_prgmg_language_MATLAB',
      'q7_most_freq_used_prgmg_language_None',
      'q7_most_freq_used_prgmg_language_Other', ]


####### Setting up the data lists #######

####### Creating fuction which counts the amount of values for q7 by professional_role #######
def counter(dataframe):
    counter_array = []
    for i in range(len(q7)):
        counter_array.append(dataframe[q7[i]].value_counts()[0])
    return counter_array

# Student Data
q1_program_student = ['Python', 'R', 'SQL', 'C', 'C++', 'Java', 'Javascript', 'Julia', 'Swift', 'Bash', 'MATLAB',
                          'None', 'Other']
q2_role_student = ['Student', 'Student', 'Student', 'Student', 'Student', 'Student', 'Student', 'Student',
                       'Student', 'Student', 'Student', 'Student', 'Student']
q3_student_counter = counter(df[df["q5_professional_role"] == "Student"])

# Data Engineer
q1_program_DataEngineer = ['Python', 'R', 'SQL', 'C', 'C++', 'Java', 'Javascript', 'Julia', 'Swift', 'Bash',
                               'MATLAB', 'None', 'Other']
q2_role_DataEngineer = ['Data Engineer', 'Data Engineer', 'Data Engineer', 'Data Engineer', 'Data Engineer',
                            'Data Engineer', 'Data Engineer', 'Data Engineer', 'Data Engineer', 'Data Engineer',
                            'Data Engineer', 'Data Engineer', 'Data Engineer']
q3_DataEngineer_counter = counter(df[df["q5_professional_role"] == 'Data Engineer'])

# Software Engineer
q1_program_SoftwareEngineer = ['Python', 'R', 'SQL', 'C', 'C++', 'Java', 'Javascript', 'Julia', 'Swift', 'Bash',
                                   'MATLAB', 'None', 'Other']
q2_role_SoftwareEngineer = ['Software Engineer', 'Software Engineer', 'Software Engineer', 'Software Engineer',
                                'Software Engineer', 'Software Engineer', 'Software Engineer', 'Software Engineer',
                                'Software Engineer', 'Software Engineer', 'Software Engineer', 'Software Engineer',
                                'Software Engineer']
q3_SoftwareEngineer_counter = counter(df[df["q5_professional_role"] == 'Software Engineer'])

# Data Scientist
q1_program_DataScientist = ['Python', 'R', 'SQL', 'C', 'C++', 'Java', 'Javascript', 'Julia', 'Swift', 'Bash',
                                'MATLAB', 'None', 'Other']
q2_role_DataScientist = ['Data Scientist', 'Data Scientist', 'Data Scientist', 'Data Scientist', 'Data Scientist',
                             'Data Scientist', 'Data Scientist', 'Data Scientist', 'Data Scientist', 'Data Scientist',
                             'Data Scientist', 'Data Scientist', 'Data Scientist']
q3_DataScientist_counter = counter(df[df["q5_professional_role"] == 'Data Scientist'])

# Data Analyst
q1_program_DataAnalyst = ['Python', 'R', 'SQL', 'C', 'C++', 'Java', 'Javascript', 'Julia', 'Swift', 'Bash',
                              'MATLAB', 'None', 'Other']
q2_role_DataAnalyst = ['Data Analyst', 'Data Analyst', 'Data Analyst', 'Data Analyst', 'Data Analyst',
                           'Data Analyst', 'Data Analyst', 'Data Analyst', 'Data Analyst', 'Data Analyst',
                           'Data Analyst', 'Data Analyst', 'Data Analyst']
q3_DataAnalyst_counter = counter(df[df["q5_professional_role"] == 'Data Analyst'])

# Research Scientist
q1_program_ResearchScientist = ['Python', 'R', 'SQL', 'C', 'C++', 'Java', 'Javascript', 'Julia', 'Swift', 'Bash',
                                    'MATLAB', 'None', 'Other']
q2_role_ResearchScientist = ['Research Scientist', 'Research Scientist', 'Research Scientist', 'Research Scientist',
                                 'Research Scientist', 'Research Scientist', 'Research Scientist', 'Research Scientist',
                                 'Research Scientist', 'Research Scientist', 'Research Scientist', 'Research Scientist',
                                 'Research Scientist']
q3_ResearchScientist_counter = counter(df[df["q5_professional_role"] == 'Research Scientist'])

# Other
q1_program_Other = ['Python', 'R', 'SQL', 'C', 'C++', 'Java', 'Javascript', 'Julia', 'Swift', 'Bash', 'MATLAB',
                        'None', 'Other']
q2_role_Other = ['Other', 'Other', 'Other', 'Other', 'Other', 'Other', 'Other', 'Other', 'Other', 'Other', 'Other',
                     'Other', 'Other']
q3_Other_counter = counter(df[df["q5_professional_role"] == 'Other'])

# Currently not employed
q1_program_NotEmployed = ['Python', 'R', 'SQL', 'C', 'C++', 'Java', 'Javascript', 'Julia', 'Swift', 'Bash',
                              'MATLAB', 'None', 'Other']
q2_role_NotEmployed = ['Currently not employed', 'Currently not employed', 'Currently not employed',
                           'Currently not employed', 'Currently not employed', 'Currently not employed',
                           'Currently not employed', 'Currently not employed', 'Currently not employed',
                           'Currently not employed', 'Currently not employed', 'Currently not employed',
                           'Currently not employed']
q3_NotEmployed_counter = counter(df[df["q5_professional_role"] == 'Currently not employed'])

# Statistician
q1_program_Statistician = ['Python', 'R', 'SQL', 'C', 'C++', 'Java', 'Javascript', 'Julia', 'Swift', 'Bash',
                               'MATLAB', 'None', 'Other']
q2_role_Statistician = ['Statistician', 'Statistician', 'Statistician', 'Statistician', 'Statistician',
                            'Statistician', 'Statistician', 'Statistician', 'Statistician', 'Statistician',
                            'Statistician', 'Statistician', 'Statistician']
q3_Statistician_counter = counter(df[df["q5_professional_role"] == 'Statistician'])

# Product/Project Manager
q1_program_ProjectManager = ['Python', 'R', 'SQL', 'C', 'C++', 'Java', 'Javascript', 'Julia', 'Swift', 'Bash',
                                 'MATLAB', 'None', 'Other']
q2_role_ProjectManager = ['Project Manager', 'Project Manager', 'Project Manager', 'Project Manager',
                              'Project Manager', 'Project Manager', 'Project Manager', 'Project Manager',
                              'Project Manager', 'Project Manager', 'Project Manager', 'Project Manager',
                              'Project Manager']
q3_ProjectManager_counter = counter(df[df["q5_professional_role"] == 'Product/Project Manager'])

# Machine Learning Engineer
q1_program_MLEngineer = ['Python', 'R', 'SQL', 'C', 'C++', 'Java', 'Javascript', 'Julia', 'Swift', 'Bash', 'MATLAB',
                             'None', 'Other']
q2_role_MLEngineer = ['ML Engineer', 'ML Engineer', 'ML Engineer', 'ML Engineer', 'ML Engineer', 'ML Engineer',
                          'ML Engineer', 'ML Engineer', 'ML Engineer', 'ML Engineer', 'ML Engineer', 'ML Engineer',
                          'ML Engineer']
q3_MLEngineer_counter = counter(df[df["q5_professional_role"] == 'Machine Learning Engineer'])

# Business Analyst
q1_program_BusinessAnalyst = ['Python', 'R', 'SQL', 'C', 'C++', 'Java', 'Javascript', 'Julia', 'Swift', 'Bash',
                                  'MATLAB', 'None', 'Other']
q2_role_BusinessAnalyst = ['Business Analyst', 'Business Analyst', 'Business Analyst', 'Business Analyst',
                               'Business Analyst', 'Business Analyst', 'Business Analyst', 'Business Analyst',
                               'Business Analyst', 'Business Analyst', 'Business Analyst', 'Business Analyst',
                               'Business Analyst']
q3_BusinessAnalyst_counter = counter(df[df["q5_professional_role"] == 'Business Analyst'])

# Merging all lists together
q1_program = q1_program_student + q1_program_DataEngineer + q1_program_SoftwareEngineer + q1_program_DataScientist + q1_program_DataAnalyst + q1_program_ResearchScientist + q1_program_Other + q1_program_NotEmployed + q1_program_Statistician + q1_program_ProjectManager + q1_program_MLEngineer + q1_program_BusinessAnalyst
q2_role = q2_role_student + q2_role_DataEngineer + q2_role_SoftwareEngineer + q2_role_DataScientist + q2_role_DataAnalyst + q2_role_ResearchScientist + q2_role_Other + q2_role_NotEmployed + q2_role_Statistician + q2_role_ProjectManager + q2_role_MLEngineer + q2_role_BusinessAnalyst
q3_count = q3_student_counter + q3_DataEngineer_counter + q3_SoftwareEngineer_counter + q3_DataScientist_counter + q3_DataAnalyst_counter + q3_ResearchScientist_counter + q3_Other_counter + q3_NotEmployed_counter + q3_Statistician_counter + q3_ProjectManager_counter + q3_MLEngineer_counter + q3_BusinessAnalyst_counter

####### initialize data of lists #######
data = {'q1_program': q1_program,
            'q2_role': q2_role,
            'q3_count': q3_count,
            }

####### Create DataFrame #######
df_test = pd.DataFrame(data)


def project_overview():
        st.title("Project Overview")
        st.write("---------------------------")
        st.header("Project Introduction")
        st.write(
            'Our project is aimed at developing a comprehensive understanding of the technical roles that have emerged in the data industry. We want to establish the range of skills that are required for each role, including data science, data analytics, data engineering, and more. Our approach is to use data analytics techniques to analyze the tasks carried out and the tools used by each position. Ultimately, the outcome of the study will be a role recommender system for students interested in pursuing a career in data-related fields.')

        st.header("Data Source and Description")
        st.write(
            "We will be using a dataset that contains the responses of 20,036 participants to a poll organized by Kaggle.com in 2020. The poll aimed to present a complete view of the state of data science and machine learning in the industry. The dataset focuses on respondents who are currently employed in data-related roles.")
        st.write("Here are some additional details about the dataset:")
        st.write(
            "- The survey consisted of multiple-choice questions and captured information on job roles, tools used, salaries, and job satisfaction.")
        st.write(
            "- The dataset is available in CSV format and can be found on the Kaggle website at https://www.kaggle.com/c/kaggle-survey-2020/overview.")
        st.write(
            "We are excited about the potential of this project to shed light on the current state of the data industry and to provide guidance to students who are interested in pursuing a career in data-related fields. Stay tuned for more updates!")

def data_screening():
    st.title("🔍 Data Screening")
    st.write("---------------------------")
    st.write("Here is a preview of the raw dataset:")

    code_snippet_data_preview= '''
#Read the Kaggle 2020 survey dataset into a pandas DataFrame
df = pd.read_csv("kaggle_survey_2020_responses.csv",low_memory=False)

#Display the first few rows of the DataFrame
df.head()
    '''

    with st.expander("Click here to show the code"):
        st.code(code_snippet_data_preview, language='python')

    st.dataframe(df_raw.head())
    st.header("🔄 Renaming and Reformatting Columns")

    st.markdown(
        "In order to streamline our data analysis process, we have taken the necessary step of simplifying the column names by making them more concise and informative. 💡💡")

    st.write(
        "This will enable us to easily identify and group together columns with similar characteristics, and allow us to apply helper functions that make data processing much more efficient. By doing this, we are setting ourselves up for a more thorough and insightful analysis of the data, and ultimately making it easier to draw meaningful conclusions from our findings.")
    code_snippet_renaming_columns = '''
    # Rename the columns of the DataFrame based on the first row
    df.columns = df.iloc[0]

    # Drop the first row of the DataFrame, which contains the original column names
    df.drop(index=0,inplace=True)

    # Renaming columns in DataFrame using a dictionary of old and new column names

    rename_columns = {
    'Duration (in seconds)':'duration_seconds',
    'What is your age (# years)?':'q1_age',
    'What is your gender? - Selected Choice':'q2_gender',
    'In which country do you currently reside?':'q3_country',
    'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?':'q4_nxt2year_highestEdu',
    'Select the title most similar to your current role (or most recent title if retired): - Selected Choice':'q5_professional_role',
    'For how many years have you been writing code and/or programming?':'q6_coding_experience',
    'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Python':'q7_most_freq_used_prgmg_language_Python',
    'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - R':'q7_most_freq_used_prgmg_language_R',
    'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - SQL':'q7_most_freq_used_prgmg_language_SQL',
     'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - C':'q7_most_freq_used_prgmg_language_C',
     'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - C++': 'q7_most_freq_used_prgmg_language_C++',
     'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Java':'q7_most_freq_used_prgmg_language_Java',
     'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Javascript':'q7_most_freq_used_prgmg_language_Javascript',
     'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Julia': 'q7_most_freq_used_prgmg_language_Julia',
     'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Swift': 'q7_most_freq_used_prgmg_language_Swift',
     'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Bash': 'q7_most_freq_used_prgmg_language_Bash',
     'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - MATLAB': 'q7_most_freq_used_prgmg_language_MATLAB',
     'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - None': 'q7_most_freq_used_prgmg_language_None',
     'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Other': 'q7_most_freq_used_prgmg_language_Other',
     'What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice':'q8_recommended_prgmg_language'
     
     # Inserting our indivually columns to our df
    df.rename(columns=rename_columns,inplace=True)
    
    df.to_csv('df_newcolumnnames.csv',index=False) '''
    # Display the code snippet using the code block in Streamlit
    with st.expander("Click here to show the code"):
        st.code(code_snippet_renaming_columns, language='python')
    st.dataframe(df.head())

    st.header("💾 Checking data types")
    st.write(
        'During the initial data type analysis, it was observed that all columns in the DataFrame, except for the first one which measures the time needed to complete the survey, have the object data type. However, since the time column will not be utilized in the analysis, it will be dropped in the preprocessing step.')
    code_snippet_checking_datatypes = '''
# Get the number of columns and rows
num_columns, num_rows = df.shape

# Print the dimensions of the dataframe
print(f"Our DataFrame has {num_rows} rows x {num_columns} cols")

# Check the data types of each column
print("Checking datatypes")
print(df.dtypes)'''
    with st.expander("Click here to show the code"):
        st.code(code_snippet_checking_datatypes, language='python')
    # Display the data types of each column
    st.write(df.dtypes)
    st.write('Furthermore, the age and salary columns were found to contain strings representing ranges, which can make it challenging to analyze and build models. Therefore, in the preprocessing step, these string values will be replaced with their respective mean values as integers. This will enable more efficient work with these columns during the analysis, allowing for a more accurate understanding of the distribution of age and salary within the dataset. These insights can help to inform decision-making.')

    st.header("💭 Management of missing data")

    # Calculate the total number of values and missing values in the DataFrame
    total_values = df.size
    missing_values = df.isna().sum().sum()

    # Calculate the total ratio of missing values in percentage
    missing_value_ratio = (missing_values / total_values) * 100

    st.write("Our dataset is riddled with a missing value ratio of  {:.2f}%".format(missing_value_ratio),"which is a considerable number! Let's delve deeper into the dataset and investigate each column's missing value percentage.")

    code_snippet_missing_values = '''
# count the total number of missing values
missing_values = df.isnull().sum().sum()

# count the total number of values in the dataframe
total_values = df.size

# calculate the total ratio of missing values in percentage
missing_value_ratio = (missing_values / total_values) * 100
print("Missing value ratio: {:.2f}%".format(missing_value_ratio))

#Display a list which entails column name and associated percentage of NaN's
percent_missing = round(df.isnull().sum() * 100 / len(df),2)

#Display the DataFrame which entails column name and associated percentage of NaN's as seperate columns in df
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})
    '''
    with st.expander("Click here to show the code"):
        st.code(code_snippet_missing_values, language='python')

    # Calculate the total number of values and missing values in the DataFrame
    total_values = df.size
    missing_values = df.isna().sum().sum()

    # Calculate the total ratio of missing values in percentage
    missing_value_ratio = (missing_values / total_values) * 100

    # Calculate percentage of missing values for each column
    percent_missing = round(df.isnull().sum() * 100 / len(df), 2)

    # Create a DataFrame to show column names and associated percentage of NaN's
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                     'percent_missing': percent_missing})

    # Display the DataFrame
    st.write(missing_value_df)
    st.write('Our dataset contains multiple choice questions, resulting in high levels of missing data in some columns, often exceeding 90%. This is a common occurrence in surveys and questionnaires. To preserve data integrity, we opt not to drop NaN values, as Python can handle them during data exploration and visualization. Dropping NaN values could lead to significant data loss and skewed results if NaN values are imputed incorrectly. It is recommended to address NaN values during the preprocessing phase using appropriate imputation techniques, ensuring accurate and reliable analysis.')

def data_exploration_and_visualization_1():
    st.title("📈 Data Exploration & Visualization (1)")
    st.write("---------------------------")
    st.header("👤 Demographic Analysis")
    st.subheader('1. Most common nationalities')
    st.write('We begin by exploring the countries that our survey participants are hailing from. Not only does this provide us with a **snapshot of the data job landscape in those nations**, but it also gives us a glimpse into the level of demand that exists in each region. As we take a quick peek at the map, we gain a clearer understanding of the survey **participants origins**.')

    code_snippet_nationalities = '''
    q3_df = df['q3_country'].value_counts()
fig = go.Figure()
fig.add_trace(go.Bar(x = q3_df.index, y = q3_df.values))
fig.update_layout(width=1000, height=700,barmode="group", plot_bgcolor = "white",title="Most Common Nationalities",xaxis_title="countries", yaxis_title="Number of respondents",legend_title_text="Professional role")
fig.show()

# Group the data by country and count the number of respondents
q3_df = df.groupby('q3_country').size().reset_index(name='count')

# Create the map figure using Plotly Express
fig = px.choropleth(q3_df, locations='q3_country', locationmode='country names', color='count',
                    projection='natural earth', title='Number of Respondents by Country',
                    color_continuous_scale='dense', range_color=(0, q3_df['count'].max()))

# Hide the legend and set the figure size
fig.update_layout(showlegend=False, width=1000, height=700)

# Remove the frame around the map
fig.update_geos(bgcolor='rgba(0,0,0,0)', showcountries=True, countrycolor='lightgray')

# Show the figure
fig.show()
    '''

    with st.expander("Click here to show the code"):
        st.code(code_snippet_nationalities, language='python')
    # Group the data by country and count the number of respondents
    q3_df = df.groupby('q3_country').size().reset_index(name='count')

    # Create the map figure using Plotly Express
    fig = px.choropleth(q3_df, locations='q3_country', locationmode='country names', color='count',
                        projection='natural earth',
                        color_continuous_scale='dense', range_color=(0, q3_df['count'].max()))

    # Hide the legend and set the figure size
    fig.update_layout(showlegend=False, width=1000, height=700)

    # Remove the frame around the map
    fig.update_geos(bgcolor='rgba(0,0,0,0)', showcountries=True, countrycolor='lightgray')

    # Show the figure
    st.plotly_chart(fig)

    st.write('📊🔍  Let us dive in and take a closer look at the numbers!')

    q3_df = df['q3_country'].value_counts()
    fig = px.bar(x=q3_df.index, y=q3_df.values, width=800, height=600,
                 labels={"y": "Number of Respondents"})

    st.plotly_chart(fig)


    st.write('Time to wrap up our discoveries ')
    st.write('- 🇮🇳 First off, it is worth noting that **India is leading** the pack with the highest number of survey participants at a staggering 29.2%. 🇺🇸 The United States is in second place with 11.2%, followed by a range of other countries.')
    st.write('- 🌍 In terms of African representation, 🇳🇬 Nigeria is taking the lead as the most represented country.')
    st.write('- 🇪🇺🇬🇧 Finally, it is interesting to note that the majority of participants from Europe come from the UK.')

    st.subheader('2. Gender distribution & age ranges participation')
    st.write("Now that we have gained an insight into the countries of origin of the survey participants, let's delve deeper into their **demographic information**. Our attention now turns to exploring the **gender and age distribution** to gain further insights and valuable information that will aid us in drawing conclusions. 🕵️‍♀️🔎")

    code_snippet_gender_age = '''
q1_q2_df = df.loc[:, ["q1_age","q2_gender"]].replace(
    {'Prefer not to say':'Divers', 'Nonbinary':"Divers", "Prefer to self-describe": "Divers"}
)
q1_q2_df['q2_gender'].value_counts()


q1_q2_df = q1_q2_df.groupby(["q1_age","q2_gender"]).size().reset_index().rename(columns = {0:"Count"})

fig = go.Figure()
for gender, group in q1_q2_df.groupby("q2_gender"):
   fig.add_trace(go.Bar(x = group['q1_age'], y = group['Count'], name = gender))
fig.update_layout(barmode="group", plot_bgcolor = "white",title="Gender distribution & age ranges participation",xaxis_title="Age (years)", yaxis_title="count",legend_title_text="Gender")
fig.show()
    '''
    with st.expander("Click here to show the code"):
        st.code(code_snippet_gender_age, language='python')
    # Replace values
    q1_q2_df = df.loc[:, ["q1_age", "q2_gender"]].replace({'Prefer not to say': 'Divers',
                                                           'Nonbinary': "Divers",
                                                           "Prefer to self-describe": "Divers"})
    # Count gender values
    gender_counts = q1_q2_df['q2_gender'].value_counts()

    # Group data by age and gender
    q1_q2_df = q1_q2_df.groupby(["q1_age", "q2_gender"]).size().reset_index().rename(columns={0: "Count"})

    # Create a bar plot
    fig = go.Figure()
    for gender, group in q1_q2_df.groupby("q2_gender"):
        fig.add_trace(go.Bar(x=group['q1_age'], y=group['Count'], name=gender))

    fig.update_layout(barmode="group",
                      title="Gender distribution & age ranges participation",
                      xaxis_title="Age (years)", yaxis_title="Count", legend_title_text="Gender")

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    st.write("Here's a summary of the demographics:")
    st.write(
        "- 👨‍💻 The **most active age group is 25-29 years old**, with a combined contribution of over half of the total submissions from the 18-21, 22-24, and 25-29 age groups.")
    st.write("- 👴 Surprisingly, even individuals over 60 years old made 653 submissions.")
    st.write(
        "- 🚻 However, the **gender and diversity distribution on Kaggle is unbalanced**, with **women being largely underrepresented**, which points to a **gender imbalance** in the broader data industry.")

    st.subheader('3. Education Level Distribution')
    st.write('To make a splash in the data job world, having the right educational background can be the key to success. So, let us explore the distribution of educational degrees as reported by the survey respondents.')

    code_snippet_education = '''
    race = go.Pie(labels=['Bachelor’s degree', 'Master’s degree', 'Some college/university without bachelor’s degree',
                      'Doctoral degree', 'I prefer not to answer', "Professional degree", 'No formal education past high school'], 
              values=df['q4_nxt2year_highestEdu'].value_counts().values, 
              hoverinfo='percent+value+label', 
              textinfo='percent',
              textposition='inside',
              hole=0.6,
              showlegend=True,
              marker=dict(colors=plt.cm.viridis_r(np.linspace(0, 1, 28)),
                          line=dict(color='#000000',
                                    width=2),
                         )
                 )

fig = go.Figure(data=[race])
fig.update_layout(title="Education Level Distribution",legend_title_text="Gender", width=800, height=600)
fig.show()
    '''
    with st.expander("Click here to show the code"):
        st.code(code_snippet_education, language='python')
    # Get data for the pie chart
    edu_counts = df['q4_nxt2year_highestEdu'].value_counts().values
    edu_labels = ['Bachelor’s degree', 'Master’s degree', 'Some college/university without bachelor’s degree',
              'Doctoral degree', 'I prefer not to answer', "Professional degree", 'No formal education past high school']

    # Create pie chart trace
    race = go.Pie(labels=edu_labels,
              values=edu_counts,
              hoverinfo='percent+value+label',
              textinfo='percent',
              textposition='inside',
              hole=0.6,
              showlegend=True,
              marker=dict(colors=plt.cm.viridis_r(np.linspace(0, 1, 28)),
                          line=dict(color='#000000', width=2)
                         )
             )

    # Create figure and add pie chart trace
    fig = go.Figure(data=[race])

    # Update layout of figure
    fig.update_layout(title="Education Level Distribution",
                  legend_title_text="Gender",
                  width=650,
                  height=500)

    # Display figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    st.write('Let us bring together our key takeaways and present them in a brief summary.')

    st.write("- 🎓 It looks like graduate degrees are the way to go in the data job world!")
    st.write(
        "- 📚 According to the survey results, participants were asked about their highest level of formal education attained or planned to attain within the next 2 years.")
    st.write(
        "- 💼 Over 40% of students are currently pursuing their Bachelor's degree, while more than 35% are planning to attain their Master's degree.")
    st.write(
        "- 👨‍🎓 Although only 5.5% of participants are aiming for a Ph.D., a vast majority of 80% have an academic background.")

def data_exploration_and_visualization_2():

    st.title("📈 Data Exploration & Visualization (2)")
    st.write("---------------------------")
    st.header("🔗 Correlation Analysis")
    st.write('As we delve deeper into our analysis, our focus turns to exploring the interplay between professional roles (target variable) and explanatory variables. Leveraging survey data and statistical correlation analyses, we aim to shed light on the impact of factors such as coding proficiency, experience, and more on specific data roles. Does the programming language R, for instance, hold a significant correlation with scientific analyst positions? Do certain programming languages dominate across all data jobs, or are some more prevalent than others? These queries will be elucidated in the forthcoming sections.')
    st.subheader('1. What are the most frequently used programming languages?')
    st.write('Let us commence by examining the most dominant programming languages, regardless of their affiliation with any particular job role.')

    code_snippet_programminglanguages = '''
    ######################## FIRST PLOT: Most frequent used programming languages #####################
fig = px.histogram(df,q7, title="Most frequent used programming languages")
fig.update_layout(showlegend=False,xaxis_title="programming languages", yaxis_title="count",plot_bgcolor = "white")
fig.show()



######################## SECOND PLOT: Most frequent used programming languages by professional role ###################



####### Creating fuction which counts the amount of values for q7 by professional_role #######
def counter (dataframe):
    counter_array = []
    for i in range (len(q7)):
        counter_array.append(dataframe[q7[i]].value_counts()[0])
    return counter_array


####### Setting up the data lists #######

#Student Data
q1_program_student = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_student = ['Student','Student','Student','Student','Student','Student','Student','Student','Student','Student','Student','Student','Student']
q3_student_counter = counter(df[df["q5_professional_role"]=="Student"])


#Data Engineer
q1_program_DataEngineer = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_DataEngineer = ['Data Engineer','Data Engineer','Data Engineer','Data Engineer','Data Engineer','Data Engineer','Data Engineer','Data Engineer','Data Engineer','Data Engineer','Data Engineer','Data Engineer','Data Engineer']
q3_DataEngineer_counter = counter(df[df["q5_professional_role"]=='Data Engineer'])

#Software Engineer
q1_program_SoftwareEngineer = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_SoftwareEngineer = ['Software Engineer','Software Engineer','Software Engineer','Software Engineer','Software Engineer','Software Engineer','Software Engineer','Software Engineer','Software Engineer','Software Engineer','Software Engineer','Software Engineer','Software Engineer']
q3_SoftwareEngineer_counter = counter(df[df["q5_professional_role"]=='Software Engineer'])

#Data Scientist
q1_program_DataScientist = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_DataScientist = ['Data Scientist','Data Scientist','Data Scientist','Data Scientist','Data Scientist','Data Scientist','Data Scientist','Data Scientist','Data Scientist','Data Scientist','Data Scientist','Data Scientist','Data Scientist']
q3_DataScientist_counter = counter(df[df["q5_professional_role"]=='Data Scientist'])

#Data Analyst
q1_program_DataAnalyst = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_DataAnalyst = ['Data Analyst','Data Analyst','Data Analyst','Data Analyst','Data Analyst','Data Analyst','Data Analyst','Data Analyst','Data Analyst','Data Analyst','Data Analyst','Data Analyst','Data Analyst']
q3_DataAnalyst_counter = counter(df[df["q5_professional_role"]=='Data Analyst'])

#Research Scientist
q1_program_ResearchScientist = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_ResearchScientist = ['Research Scientist','Research Scientist','Research Scientist','Research Scientist','Research Scientist','Research Scientist','Research Scientist','Research Scientist','Research Scientist','Research Scientist','Research Scientist','Research Scientist','Research Scientist']
q3_ResearchScientist_counter = counter(df[df["q5_professional_role"]=='Research Scientist'])

#Other
q1_program_Other = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_Other = ['Other','Other','Other','Other','Other','Other','Other','Other','Other','Other','Other','Other','Other']
q3_Other_counter = counter(df[df["q5_professional_role"]=='Other'])


# Currently not employed
q1_program_NotEmployed = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_NotEmployed = ['Currently not employed','Currently not employed','Currently not employed','Currently not employed','Currently not employed','Currently not employed','Currently not employed','Currently not employed','Currently not employed','Currently not employed','Currently not employed','Currently not employed','Currently not employed']
q3_NotEmployed_counter = counter(df[df["q5_professional_role"]=='Currently not employed'])

# Statistician
q1_program_Statistician = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_Statistician = ['Statistician','Statistician','Statistician','Statistician','Statistician','Statistician','Statistician','Statistician','Statistician','Statistician','Statistician','Statistician','Statistician']
q3_Statistician_counter = counter(df[df["q5_professional_role"]=='Statistician'])


# Product/Project Manager
q1_program_ProjectManager = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_ProjectManager = ['Project Manager','Project Manager','Project Manager','Project Manager','Project Manager','Project Manager','Project Manager','Project Manager','Project Manager','Project Manager','Project Manager','Project Manager','Project Manager']
q3_ProjectManager_counter = counter(df[df["q5_professional_role"]=='Product/Project Manager'])


# Machine Learning Engineer
q1_program_MLEngineer = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_MLEngineer = ['ML Engineer','ML Engineer','ML Engineer','ML Engineer','ML Engineer','ML Engineer','ML Engineer','ML Engineer','ML Engineer','ML Engineer','ML Engineer','ML Engineer','ML Engineer']
q3_MLEngineer_counter = counter(df[df["q5_professional_role"]=='Machine Learning Engineer'])


# Business Analyst
q1_program_BusinessAnalyst = ['Python','R','SQL','C', 'C++','Java','Javascript','Julia','Swift','Bash','MATLAB','None','Other']
q2_role_BusinessAnalyst = ['Business Analyst','Business Analyst','Business Analyst','Business Analyst','Business Analyst','Business Analyst','Business Analyst','Business Analyst','Business Analyst','Business Analyst','Business Analyst','Business Analyst','Business Analyst']
q3_BusinessAnalyst_counter = counter(df[df["q5_professional_role"]=='Business Analyst'])



#Merging all lists together
q1_program = q1_program_student+q1_program_DataEngineer+q1_program_SoftwareEngineer+q1_program_DataScientist+q1_program_DataAnalyst+q1_program_ResearchScientist+q1_program_Other+q1_program_NotEmployed+q1_program_Statistician+q1_program_ProjectManager+q1_program_MLEngineer+q1_program_BusinessAnalyst
q2_role=q2_role_student+q2_role_DataEngineer+q2_role_SoftwareEngineer+q2_role_DataScientist+q2_role_DataAnalyst+q2_role_ResearchScientist+q2_role_Other+q2_role_NotEmployed+q2_role_Statistician+q2_role_ProjectManager+q2_role_MLEngineer+q2_role_BusinessAnalyst
q3_count=q3_student_counter+q3_DataEngineer_counter+q3_SoftwareEngineer_counter+q3_DataScientist_counter+q3_DataAnalyst_counter+q3_ResearchScientist_counter+q3_Other_counter+q3_NotEmployed_counter+q3_Statistician_counter+q3_ProjectManager_counter+q3_MLEngineer_counter+q3_BusinessAnalyst_counter

####### initialize data of lists #######
data = {'q1_program':q1_program,
        'q2_role':q2_role,
        'q3_count':q3_count,
       }
 
####### Create DataFrame #######
df_test = pd.DataFrame(data)


####### Plot second figure #######
fig = go.Figure()
for gender, group in df_test.groupby("q2_role"):
   fig.add_trace(go.Bar(x = group['q1_program'], y = group['q3_count'], name = gender))
fig.update_layout(barmode="group", plot_bgcolor = "white",title="Most frequent used programming languages by professional role",xaxis_title="programming languages", yaxis_title="count",legend_title_text="Professional role",height=500)
fig.show()
    '''

    with st.expander("Click here to show the code"):
        st.code(code_snippet_programminglanguages, language='python')

    fig = px.histogram(df, q7, title="Most frequently used programming languages")
    fig.update_layout(showlegend=False, xaxis_title="Programming languages", yaxis_title="Count")

    st.plotly_chart(fig)

    st.write("🐍 Python is the most widely used programming language by a large margin, with SQL coming in second place. R is in third place. Now, let's take a closer look at how each job role uses these languages in detail.")

    ######################## SECOND PLOT: Most frequent used programming languages by professional role ###################


    ####### Plot second figure #######
    fig = go.Figure()
    for gender, group in df_test.groupby("q2_role"):
        fig.add_trace(go.Bar(x=group['q1_program'], y=group['q3_count'], name=gender))
    fig.update_layout(barmode="group",
                      title="Most frequent used programming languages by professional role",
                      xaxis_title="programming languages", yaxis_title="count", legend_title_text="Professional role", width=900)
    st.plotly_chart(fig)
    st.write('**Lets draw this to a conclusion:**')
    st.write(
        "- 🌟 **Python and SQL are the rockstars** of the job market, with their popularity spanning across all job roles and even among the student community.")
    st.write(
        "- 📊 **R is the go-to language for Data Scientists**, Data Analysts, and students, but seems to have fallen out of favor among ML Engineers and Data Engineers.")
    st.write(
        "- 💻 **Bash, the power tool for the command line**, seems to have a higher fanbase among MLOps Engineers. Interestingly, some other languages also seem to have an edge over R in terms of popularity among Data Scientists and Data Analysts.")

    st.subheader('2. Relationship between coding experience and Data Job Roles')
    st.write('🤔 Is coding experience a key factor in determining specific data-driven job roles?')
    st.write('🔍 The **V-Cramer Test result (0,2)** has revealed a moderate correlation between coding experience and data job roles. This raises the question of how programming proficiency and job requirements are intricately linked.')
    st.write('**Pair counting** will **identify relationships and patterns** between the two variables:')
    code_snippet_paircount = '''
    #V Cramers Test
    def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))
    cramers_v(df['q6_coding_experience'],df['q5_professional_role'])
    
    #Heatmap
    def heatmap_paircount_between_cat_variables(dataframe, column1_title, column2_title, figsize=(10, 8)):
    sns.set(font_scale=1.2)
    fig, ax = plt.subplots(figsize=figsize)
    ct_counts = dataframe.groupby([column1_title, column2_title]).size()
    ct_counts = ct_counts.reset_index(name='count')
    ct_counts = ct_counts.pivot(index=column2_title, columns=column1_title, values='count')
    heatmap_paircount = sns.heatmap(ct_counts, cmap='YlGnBu', annot=True, fmt='d')
    ax.set_title('Pairplot Heatmap of Counts between Two Variables')
    ax.set_xlabel(column1_title)
    ax.set_ylabel(column2_title)
    return heatmap_paircount
heatmap_paircount_between_cat_variables(df,'q6_coding_experience','q5_professional_role');
    '''
    with st.expander("Click here to show the code"):
        st.code(code_snippet_paircount, language='python')


    def heatmap_paircount_between_cat_variables(dataframe, column1_title, column2_title, figsize=(10, 8)):
        ct_counts = dataframe.groupby([column1_title, column2_title]).size().reset_index(name='count')
        ct_counts_pivot = ct_counts.pivot(index=column2_title, columns=column1_title, values='count')
        heatmap_data = go.Heatmap(z=ct_counts_pivot.values,
                                  x=ct_counts_pivot.columns.tolist(),
                                  y=ct_counts_pivot.index.tolist(),
                                  colorscale='YlGnBu',
                                  zmin=0,
                                  zmax=ct_counts['count'].max(),
                                  reversescale=False,
                                  showscale=True,
                                  hovertemplate='%{z}',
                                  text=ct_counts_pivot.values,
                                  )
        heatmap_layout = go.Layout(xaxis=dict(title=column1_title),
                                   yaxis=dict(title=column2_title),
                                   height=figsize[1] * 60,
                                   width=figsize[0] * 60
                                   )
        heatmap_fig = go.Figure(data=[heatmap_data], layout=heatmap_layout)
        heatmap_fig.update_traces(texttemplate='%{text:.0f}', textfont=dict(size=12))
        return heatmap_fig

    # Display the heatmap
    heatmap_fig = heatmap_paircount_between_cat_variables(df, 'q6_coding_experience', 'q5_professional_role')
    st.plotly_chart(heatmap_fig, use_container_width=True)
    st.write('**Conclusion:**')
    st.write('- Like the V Cramer Test already suggested, we can not see strong relationships between the two variables.')
    st.write('- But as expected, **students usually have little programming experience**.')
    st.write('- Interestingly, **data scientists are more experienced in coding** and bring a coding experience of 3-5 years or 5-10 years here.')


    st.subheader('3. Relationship between job role and salary')
    st.write('💰Let us dive into the topic of money! It is fascinating to see what annual salaries the survey data reveals. That is why we can use a correlation heatmap to explore which **job roles are associated with different annual compensations**.')

    code_snippet_correlation= '''
    def heatmap_between_cat_variables(dataframe,column1_title,column2_title):
    df_dummies1 = pd.get_dummies(dataframe[column1_title])
    df_dummies2 = pd.get_dummies(dataframe[column2_title])
    df_new = pd.concat([dataframe.drop([column2_title], axis=1), df_dummies1,df_dummies2], axis=1)
    df_new = df_new.iloc[:,1:]
    plt.figure(figsize = (18,6))
    heatmap = sns.heatmap(df_new.corr(method = 'spearman'), vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', linewidths=3, linecolor='black',annot= True);
    return heatmap
    '''

    with st.expander("Click here to show the code"):
        st.code(code_snippet_correlation, language='python')

    def heatmap_between_cat_variables(df, column1_title, column2_title):
        # Create dummy variables for the two columns
        df_dummies1 = pd.get_dummies(df[column1_title])
        df_dummies2 = pd.get_dummies(df[column2_title])

        # Combine the dummy variables and remove the original columns
        df_new = pd.concat([df.drop([column1_title, column2_title], axis=1), df_dummies1, df_dummies2], axis=1)
        df_new = df_new.iloc[:, 1:]

        # Create a correlation matrix using the spearman method
        corr_matrix = df_new.corr(method='spearman')

        # Create a heatmap using the correlation matrix
        fig = px.imshow(corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        color_continuous_scale='YlGnBu',
                        zmin=-1,
                        zmax=1)

        # Add annotations to the heatmap
        annotations = []
        for i, row in enumerate(corr_matrix.values):
            for j, value in enumerate(row):
                annotations.append(dict(
                    x=corr_matrix.columns[j],
                    y=corr_matrix.columns[i],
                    text=f"{value:.2f}",
                    showarrow=False,
                    font=dict(color='black', size=6)
                ))
        fig.update_layout(width=1000, height=800,
            xaxis_title=column1_title,
            yaxis_title=column2_title,
            annotations=annotations
        )
        return fig

    heatmap_fig = heatmap_between_cat_variables(df, 'q24_yearly_compensation_$USD', 'q5_professional_role')
    st.plotly_chart(heatmap_fig)
    st.write('Based on the correlations displayed in the heatmap, **no strong links** can be identified. Nonetheless, it can be inferred that students typically fall in the lower income bracket, if they have any income at all. 💰 On the other hand, data scientists, following the saying **"with experience comes money,"** tend to have salaries in the higher income ranges.')

def preprocessing():
    st.title("🛠️ Data Preprocessing")
    st.write("---------------------------")
    st.subheader('Step 1: Yearly compensation: Replacing string values by its corresponding int value (mean) ')
    st.write('Before building our model, let us get our data ready with some **preprocessing steps**! We start by creating a **copy of our original dataframe**, then focus on the "q24_yearly_compensation_$USD" column, which contains valuable info on yearly compensation. We spruce it up by **converting string values to their integer counterparts**, taking the mean of string ranges, and filling in missing values with the mean for corresponding professional roles.')
    # Define the code snippet as a string
    code_snippet_1 = '''
# Define function in order to convert string values to their integer counterparts
df_preprocessed = df.copy()
def preprocess_compensation(column):
    def get_mean_compensation(compensation):
        if pd.isnull(compensation):
            return np.nan
        if compensation.startswith('$'):
            compensation = compensation[1:]
        if '>' in compensation:
            split_compensation = compensation.split(' ')
            if len(split_compensation) < 3:
                return np.nan
            compensation = split_compensation[2].replace(',', '')
            return int(float(compensation))
        lower, upper = map(lambda x: float(x.replace(',', '')), compensation.split('-'))
        mean = (lower + upper) / 2
        return int(round(mean))
    return column.apply(get_mean_compensation)
    
#Call function and replace compensation columns data
df_preprocessed['q24_annual_compensation'] = preprocess_compensation(df_preprocessed['q24_yearly_compensation_$USD'])
    
#Group by professional role and find the mean of the yearly compensation column
mean_compensation_by_role = df_preprocessed.groupby('q5_professional_role')['q24_annual_compensation'].mean()

# Replace missing values in yearly compensation column with the mean compensation for the corresponding professional role
for role in mean_compensation_by_role.index:
    mean_compensation = mean_compensation_by_role[role]
    df_preprocessed.loc[(df_preprocessed['q24_annual_compensation'].isna()) & (df_preprocessed['q5_professional_role'] == role), 'q24_annual_compensation'] = mean_compensation
    '''

    # Display the code snippet using the code block in Streamlit
    with st.expander("Click here to show the code"):
        st.code(code_snippet_1, language='python')


    st.subheader('Step 2: Dropping rows of non-data job related roles ')
    st.write('💰 After sprucing up the yearly compensation column, our next step is to 🚫 **filter out non-data job roles related values by dropping them**. That wayy, we ensure that our **model will be only trained on data-related roles**. This helps in **reducing noise and bias** that might be introduced by considering non-data job roles.')
    code_snippet_2 = '''
# Create a list of values to drop from the 'q5_professional_role' column
to_drop = ['Student','Other','Currently not employed',np.nan,]

# Use the 'isin' method to select rows where the 'q5_professional_role' column does not contain any values in 'to_drop'
df_preprocessed = df_preprocessed[~df_preprocessed['q5_professional_role'].isin(to_drop)]
        '''

    # Display the code snippet using the code block in Streamlit
    with st.expander("Click here to show the code"):
        st.code(code_snippet_2, language='python')

    st.subheader('Step 3: q6_coding_experience: Replacing string values by its corresponding int value (mean) ')
    st.write('👋 In Step 3, we will **preprocess the "q6_coding_experience"** column of our dataset by **replacing string values with corresponding integer values**. We use the "preprocess_coding_experience" function to handle a range of values and null values, and drop the rows containing null values. The resulting "q6_coding_experience" column is clean and numeric, ready to be used in our machine learning models.')
    code_snippet_3 = '''
# Define a function to preprocess the 'q6_coding_experience' column
def preprocess_coding_experience(value):
    # Return the value if it is null
    if pd.isnull(value):
        return value
    # If the value contains '<', assign a value of 1 to represent less than a year of coding experience
    if '<' in value:
        return 1
    # If the value contains '+', assign a value of 25 to represent 25 or more years of coding experience
    if '+' in value:
        return 25
    # If the value contains 'I have never', assign a value of 0 to represent no coding experience
    if 'I have never' in value:
        return 0
    # Otherwise, split the value by space, extract the first two integers separated by a hyphen, and calculate the average
    start, end = map(int, value.split(' ')[0].split('-')[:2])
    return (start + end) / 2

# Apply the preprocess_coding_experience function to the 'q6_coding_experience' column of the df_preprocessed dataframe
df_preprocessed['q6_coding_experience_years'] = df_preprocessed['q6_coding_experience'].apply(preprocess_coding_experience)

# Check how many null values are in the 'q6_coding_experience' column before dropping them
print(df_preprocessed['q6_coding_experience'].isna().sum())

# Drop rows with null values in the 'q6_coding_experience' column
df_preprocessed.dropna(subset=['q6_coding_experience'], inplace=True)
            '''

    # Display the code snippet using the code block in Streamlit
    with st.expander("Click here to show the code"):
        st.code(code_snippet_3, language='python')

    st.subheader('Step 4: Standard scaling numerical variables')
    st.write(
        '🔢 In Step 4, we will **standard scale our numerical variables** using the StandardScaler() function. We select the numerical columns and apply the scaler to bring all features to a similar scale. This **improves the performance of machine learning models**. After this step, our numerical variables will be ready for training.')
    code_snippet_4 = '''
# Select numerical columns
numerical_cols = df_preprocessed.select_dtypes(include=['float']).columns

# Standardize the numerical columns
scaler = StandardScaler()
df_preprocessed[numerical_cols] = scaler.fit_transform(df_preprocessed[numerical_cols])
                '''

    # Display the code snippet using the code block in Streamlit
    with st.expander("Click here to show the code"):
        st.code(code_snippet_4, language='python')

    st.subheader('Step 5: Preprocessing categorical columns')
    st.write(
        '📏 To **preprocess categorical columns** we can use the pandas " get_dummies" function. This handy function **creates new one-hot encoded columns for each unique category** in the input column. This helps to **transform categorical data into a more numerical format** that can be easily understood by machine learning models. So, we can say that get_dummies function plays an important role in preparing our data for further analysis.')
    code_snippet_5 = '''
# Select categorical columns
cat_cols = df_preprocessed.select_dtypes(include='object').columns

# One-hot encode the categorical columns using pd.get_dummies() function
# prefix: the string to use as prefix for the new columns 
# prefix_sep: the separator to use between prefix and the original column name
# drop_first: to drop the first category to avoid multicollinearity 
df_preprocessed = pd.get_dummies(df_preprocessed, 
                                 columns=cat_cols, 
                                 prefix=cat_cols, 
                                 prefix_sep='_', 
                                 drop_first=True)
                '''

    # Display the code snippet using the code block in Streamlit
    with st.expander("Click here to show the code"):
        st.code(code_snippet_5, language='python')

    st.subheader('Step 6: Dropping redundant columns')
    st.write('💥 In Step 6, we will **remove redundant columns** from our preprocessed dataframe using the drop() function. Here, we will be dropping the "duration_seconds" column to **reduce the dimensionality of the dataset** and remove any unimportant or redundant features. Removing such features can improve the accuracy of the model. After this step, our dataframe will be ready for training machine learning models.')
    code_snippet_6 = '''
df_preprocessed = df_preprocessed.drop(['duration_seconds'], axis=1)
                '''

    # Display the code snippet using the code block in Streamlit
    with st.expander("Click here to show the code"):
        st.code(code_snippet_6, language='python')

    st.subheader('✅ Overview: Preprocessed Dataframe')
    st.write('👀 Take a look at the preprocessed dataframe:')
    st.dataframe(df_preprocessed)

def model_building():
    st.title("🛠️ Model Building and Evaluation")
    st.write('🔍 In the process of building a machine learning based job role recommender, three different models - **Logistic Regression**, **Random Forest**, and **Gradient Boosting Classifier** - will be tested to determine which one will perform the best in predicting job roles based on preprocessed data.')
    st.write("---------------------------")
    st.header('📈 Model 1: Logistic Regression')
    # Split the dataset into training and testing sets
    X = df_preprocessed.drop(columns=['q5_professional_role_DBA/Database Engineer',
                                      'q5_professional_role_Data Analyst',
                                      'q5_professional_role_Data Scientist',
                                      'q5_professional_role_Machine Learning Engineer',
                                      'q5_professional_role_Product/Project Manager',
                                      'q5_professional_role_Research Scientist',
                                      'q5_professional_role_Software Engineer',
                                      'q5_professional_role_Statistician', 'q5_professional_role_Data Engineer'])
    y = df_preprocessed.loc[:, ['q5_professional_role_Data Analyst',
                                'q5_professional_role_Data Scientist', 'q5_professional_role_Data Engineer']]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert y_train and y_test into 1D arrays of binary labels
    y_train = np.argmax(y_train.values, axis=1)
    y_test = np.argmax(y_test.values, axis=1)

    # Hyperparameter tuning using grid search
    parameters = {'C': [0.1, 1, 10, 100], 'penalty': ['l2']}
    clf = GridSearchCV(LogisticRegression(max_iter=10000), parameters, cv=5)
    clf.fit(X_train, y_train)
    best_params = clf.best_params_

    # Train the logistic regression model with best hyperparameters
    model = LogisticRegression(C=best_params['C'], penalty=best_params['penalty'], max_iter=10000)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write('In this code, the Logistic Regression model is built using preprocessed data to predict job roles. First, the data is split into training and testing sets. Then, hyperparameter tuning is performed using GridSearchCV to find the best hyperparameters. ')

    code_snippet_logisticregression= '''
    # Split the dataset into training and testing sets
X = df_preprocessed.drop(columns=['q5_professional_role_DBA/Database Engineer',
       'q5_professional_role_Data Analyst',
       'q5_professional_role_Data Scientist',
       'q5_professional_role_Machine Learning Engineer',
       'q5_professional_role_Product/Project Manager',
       'q5_professional_role_Research Scientist',
       'q5_professional_role_Software Engineer',
       'q5_professional_role_Statistician','q5_professional_role_Data Engineer'])
y = df_preprocessed.loc[:, ['q5_professional_role_Data Analyst',
       'q5_professional_role_Data Scientist','q5_professional_role_Data Engineer']]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert y_train and y_test into 1D arrays of binary labels
y_train = np.argmax(y_train.values, axis=1)
y_test = np.argmax(y_test.values, axis=1)

# Hyperparameter tuning using grid search
parameters = {'C': [0.1, 1, 10, 100], 'penalty': ['l2']}
clf = GridSearchCV(LogisticRegression(max_iter=10000), parameters, cv=5)
clf.fit(X_train, y_train)
best_params = clf.best_params_

# Train the logistic regression model with best hyperparameters
model = LogisticRegression(C=best_params['C'], penalty=best_params['penalty'], max_iter=10000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Perform cross-validation
cv_scores = cross_val_score(model, X, y.values.argmax(axis=1), cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))

# Analyze errors by comparing y_test and y_pred
errors = y_test != y_pred
error_indices = np.where(errors)[0]
print(f"Number of errors: {len(error_indices)}")

# Create a SHAP explainer and compute SHAP values
explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer.shap_values(X_test)

# Visualize the SHAP values
shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, plot_type='bar')
    '''
    with st.expander("Click here to show the code"):
        st.code(code_snippet_logisticregression, language='python')


    st.write('After training and evaluating the logistic regression model on the testing set, the accuracy score achieved is',accuracy , 'which indicates that the model is able to correctly predict job roles with a 70.37% accuracy.')
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y.values.argmax(axis=1), cv=5)
    st.write("The cross_val_score function is used to perform 5-fold cross-validation on the Logistic Regression model. The scores obtained for each fold are:", cv_scores)
    st.write("...and the mean cross-validation score is:", np.mean(cv_scores),'which is a reasonably good performance')

    # Analyze errors by comparing y_test and y_pred
    errors = y_test != y_pred
    error_indices = np.where(errors)[0]
    st.write('However, the errors variable indicates that there are', len(error_indices), 'errors in the predictions made by the model, which suggests that there is still some room for improvement.')
    st.subheader('Feature Selection for Improved Recommendation Model Performance')
    # Create a SHAP explainer and compute SHAP values
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)

    st.write('To develop our recommendation model, we do not require respondents to answer all survey questions. Instead, we will focus only on the questions that have the most significant average impact on output of our model. To identify these key questions, we will use an algorithm called SHAP to calculate how much a single feature affected the prediction. This approach will help us reduce the number of questions needed to make accurate recommendations, while still maintaining the model accuracy.')
    # visualize the SHAP values
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, plot_type='bar',alpha=0.7)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xticklabels(ax.get_xticks(), fontsize=8)
    ax.patch.set_alpha(0)
    with st.expander("Check out the **SHAP summary plot** by clicking here 👈"):
        st.pyplot(fig, bbox_inches='tight', pad_inches=0, transparent=True)

    st.header('🌲 Model 2: Random Forest')
    st.write(' The advantage of using a random forest model is that it can handle large datasets with many features and can avoid overfitting by averaging the predictions of multiple decision trees. In our example, we will be using feature selection with a random forest and hyperparameter tuning with grid search to optimize the performance of our model. ')

    code_snippet_randomforest = '''
    # Feature selection using random forest
selector = SelectFromModel(estimator=RandomForestClassifier(n_estimators=100))
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Hyperparameter tuning using grid search
parameters = {'n_estimators': [50, 100, 150], 'max_depth': [10, 20, 30, None]}
clf = GridSearchCV(RandomForestClassifier(), parameters, cv=5)
clf.fit(X_train_selected, y_train)
best_params = clf.best_params_

# Train the random forest model with best hyperparameters
model = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'])
model.fit(X_train_selected, y_train)

# Evaluate the model
y_pred = model.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Perform cross-validation
cv_scores = cross_val_score(model, selector.transform(X), y.values.argmax(axis=1), cv=5)
    '''
    with st.expander("Click here to show the code"):
        st.code(code_snippet_randomforest, language='python')
    # Feature selection using random forest
    selector = SelectFromModel(estimator=RandomForestClassifier(n_estimators=100))
    selector.fit(X_train, y_train)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    # Hyperparameter tuning using grid search
    parameters = {'n_estimators': [50, 100, 150], 'max_depth': [10, 20, 30, None]}
    clf = GridSearchCV(RandomForestClassifier(), parameters, cv=5)
    clf.fit(X_train_selected, y_train)
    best_params = clf.best_params_

    # Train the random forest model with best hyperparameters
    model = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'])
    model.fit(X_train_selected, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)

    # Perform cross-validation
    cv_scores = cross_val_score(model, selector.transform(X), y.values.argmax(axis=1), cv=5)

    # Display the results using Streamlit
    st.write("Accuracy:", accuracy)
    st.write("Cross-validation scores:", cv_scores)
    st.write("Mean cross-validation score:", np.mean(cv_scores))

    st.header('🚀 Model 3: Gradient Boost Classifier')
    st.write('Gradient boosting is particularly useful when working with complex datasets, where traditional machine learning algorithms might not be effective. The goal is to find the best combination of hyperparameters for the gradient boosting classifier, which will help us achieve high accuracy on the test data.')
    code_snippet_gradientboostclassifier = '''
    # Define the hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 1.0]
}

# Perform grid search using 5-fold cross validation
grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid=param_grid, cv=5, n_jobs=-1)

# Fit the grid search to the training data
grid_search.fit(X_train_selected, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Train the Gradient Boosting Classifier with the best hyperparameters
model = GradientBoostingClassifier(**best_params)
model.fit(X_train_selected, y_train)

# Evaluate the model
y_pred = model.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Perform cross-validation
cv_scores = cross_val_score(model, selector.transform(X), y.values.argmax(axis=1), cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))

# Analyze errors by comparing y_test and y_pred
errors = y_test != y_pred
error_indices = np.where(errors)[0]
error_samples = X_test.iloc[error_indices]
print("Number of errors:", len(error_samples))
    '''

    with st.expander("Click here to show the code"):
        st.code(code_snippet_gradientboostclassifier, language='python')

    # Define the hyperparameters to tune
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 1.0]
    }

    # Perform grid search using 5-fold cross validation
    grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid=param_grid, cv=5, n_jobs=-1)

    # Fit the grid search to the training data
    grid_search.fit(X_train_selected, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # Train the Gradient Boosting Classifier with the best hyperparameters
    model = GradientBoostingClassifier(**best_params)
    model.fit(X_train_selected, y_train)

    # Evaluate the model
    y_pred1 = model.predict(X_test_selected)
    accuracy1 = accuracy_score(y_test, y_pred1)
    st.write("Accuracy:", accuracy1)

    # Perform cross-validation
    cv_scores1 = cross_val_score(model, selector.transform(X), y.values.argmax(axis=1), cv=5)
    st.write("Cross-validation scores:", cv_scores1)
    st.write("Mean cross-validation score:", np.mean(cv_scores1))

    # Analyze errors by comparing y_test and y_pred
    errors1 = y_test != y_pred1
    error_indices = np.where(errors1)[0]
    error_samples = X_test.iloc[error_indices]
    st.write("Number of errors:", len(error_samples))

    st.header('Conclusions')

    st.write("We tested three different models: Logistic Regression, Random Forest and Gradient Boosting Classifier.")

    st.write(
        "- Logistic Regression is a linear classifier that uses a logistic function to model the probability of the output belonging to a certain class. Here, we used a grid search to tune the hyperparameters of the model, specifically the regularization strength C and the penalty type L2.")

    st.write(
        "- Next, we used Random Forest, an ensemble method that uses multiple decision trees to make predictions. We also performed feature selection using a Random Forest model to select the most important features for the classification task. We then used a grid search to tune the hyperparameters of the model, including the number of estimators and the maximum depth of the trees. This resulted in a slight improvement in accuracy compared to Logistic Regression.")

    st.write(
        "- Finally, we used Gradient Boosting Classifier, another ensemble method that sequentially adds models to correct errors made by previous models. We used the same features selected by the Random Forest model and did not perform any additional hyperparameter tuning. This resulted in the **highest accuracy of the three models tested**.")

    st.write(
        "- Overall, we can see that the performance of the models improved as we incorporated more advanced techniques such as feature selection and hyperparameter tuning. The **Gradient Boosting Classifier performed the best**, likely due to its ability to correct errors made by previous models and its use of a strong regularization term. In any case, an accuracy of around 0.79 is not bad, and the model seems to be performing reasonably well on the given dataset.")


# Create a dictionary of page names and corresponding functions
pages = {
    "Project Overview": project_overview,
    "Data Screening": data_screening,
    "Data Exploration and Data Viz' (1): Demographic Analysis": data_exploration_and_visualization_1,
    "Data Exploration and Data Viz' (2): Correlation Analysis": data_exploration_and_visualization_2,
    "Data Preprocessing": preprocessing,
    "Model Building and Evaluation": model_building,
}

# Create the Streamlit app
def main():
    st.sidebar.title("Navigation")
    page_names = list(pages.keys())
    selected_page = st.sidebar.radio("Go to", page_names)
    pages[selected_page]()

if __name__ == "__main__":
    main()

