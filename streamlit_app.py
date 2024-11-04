import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
import langchain


### Load your API Key
my_secret_key = st.secrets['MyOpenAIKey']

### Create the LLM API object
llm = OpenAI(openai_api_key=my_secret_key)

# Define templates for different response types
trip_template = """
You are an expert in analyzing travel experiences.
From the following text, determine whether the trip experience was "positive" or "negative".

Text:
{trip_experience}
"""

# Create the decision-making chain
trip_experience_chain = (
    PromptTemplate.from_template(trip_template)
    | llm
    | StrOutputParser()
)

# individual response templates for each scenario
airline_issue_prompt = PromptTemplate.from_template(
    """You are a customer service representative for an airline.
    The customer had a negative experience with the airline due to an issue caused by the airline (e.g., lost luggage).

    Your response should follow these guidelines:
    1. Express sympathy and acknowledge the inconvenience.
    2. Inform the customer that the customer service team will contact them soon to provide compensation.
    3. Address the customer directly in a professional tone.

Trip Experience: {trip_experience}

"""    
) | llm

external_issue_prompt = PromptTemplate.from_template(
    """You are a customer service representative for an airline.
    The customer had a negative experience due to external factors beyond the airline's control (e.g., weather-related delays).

    Your response should follow these guidelines:
    1. Express sympathy for the inconvenience caused by external circumstances.
    2. Explain that the airline is not liable in such cases and will not provide compensation.
    3. Address the customer directly in a professional tone.

Trip Experience: {trip_experience}

"""
) | llm


positive_feedback_prompt = PromptTemplate.from_template(
    """You are a customer service representative for an airline.
    The customer had a positive experience, and you would like to acknowledge their feedback.

    Your response should follow these guidelines:
    1. Thank the customer for their feedback and for choosing to fly with the airline.
    2. Convey appreciation and a professional, positive tone.
    3. Address the customer directly.

Trip Experience: {trip_experience}
""" 
) | llm

# Routing/Branching chain
branch = RunnableBranch(

    (lambda x: "negative" in x["trip_experience"].lower(), "airline" in x["trip_experience"].lower(), "lost luggage" in x["trip_experience"].lower(), airline_issue_prompt),
    (lambda x: "negative" in x["trip_experience"].lower(), "weather" in x["trip_experience"].lower(), external_issue_prompt),
    positive_feedback_prompt
)

# Put all the chains together
full_chain = {"trip_experience": trip_experience_chain, "trip_experience": lambda x: x["request"]} | branch

# Display Side
st.title("Flight Experience Feedback Form")
st.header("Share with us your experience of the latest trip.")

# Textbox for user input
trip_experience = st.text_area("Please describe your trip:")

if trip_experience:
    # Run the full chain to get a categorized response
    result = full_chain.invoke({"request": trip_experience})
    st.write(result)
