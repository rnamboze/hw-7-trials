import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.llms import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
import langchain
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

### Load your API Key
my_secret_key = st.secrets['MyOpenAIKey']

### Create the LLM API object
llm = OpenAI(openai_api_key=my_secret_key)

# Define the main analysis prompt
trip_analysis_template = """
Analyze the following trip experience:
1. Summarize the experience briefly.
2. Indicate if the experience was positive or negative.
3. If negative, specify if the dissatisfaction was due to the airline (e.g., lost luggage) or beyond the airline’s control (e.g., weather delay).

Trip Experience: {trip_experience}
"""
trip_analysis_prompt = PromptTemplate(input_variables=["trip_experience"], template=trip_analysis_template)
trip_analysis_chain = LLMChain(llm=llm, prompt=trip_analysis_prompt)

# Define response templates
airline_issue_template = "We're sorry to hear about your negative experience caused by the airline. Our customer service team will reach out soon to resolve the issue or provide compensation. We apologize for any inconvenience caused."
external_issue_template = "We’re sorry to hear about your negative experience due to factors beyond the airline's control, such as weather-related delays. Unfortunately, the airline is not liable in such cases. We appreciate your understanding."
positive_feedback_template = "Thank you for your positive feedback and for choosing to fly with us! We’re glad you had a good experience."

# Define response chains (no LLM needed since responses are fixed)
def airline_issue_response(_):
    return airline_issue_template

def external_issue_response(_):
    return external_issue_template

def positive_feedback_response(_):
    return positive_feedback_template

# Define the branching logic
def is_airline_issue(summary):
    return "negative" in summary.lower() and ("airline" in summary.lower() or "lost luggage" in summary.lower())

def is_external_issue(summary):
    return "negative" in summary.lower() and ("weather" in summary.lower() or "beyond control" in summary.lower() or "delay" in summary.lower())

branch = RunnableBranch(
    (is_airline_issue, airline_issue_response),
    (is_external_issue, external_issue_response),
    positive_feedback_response  # Default response
)

# Combine the chains
full_chain = trip_analysis_chain | branch

# Streamlit Interface
st.title("Trip Experience Summarizer")
st.header("Share with us your experience of the latest trip.")

# Textbox for user input
trip_experience = st.text_area("Please describe your trip:")

if trip_experience:
    # Run the full chain to get a categorized response
    analysis_result = trip_analysis_chain.run(trip_experience=trip_experience)
    response = branch.invoke(analysis_result)
    
    # Display the summary and the response
    st.subheader("Summary of Your Trip Experience:")
    st.write(analysis_result)
    st.subheader("Our Response:")
    st.write(response)
