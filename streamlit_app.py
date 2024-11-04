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

### Define templates for different response types
trip_template = """
Analyze the following trip experience:
1. Summarize the experience briefly.
2. Indicate if the experience was positive or negative.
3. If negative, specify if the dissatisfaction was due to the airline (e.g., lost luggage) or beyond the airline’s control (e.g., weather delay).

Trip Experience: {trip_experience}
"""

airline_issue_template = """
You had a negative experience caused by the airline. Our customer service team will reach out soon to resolve the issue or provide compensation. We apologize for any inconvenience caused.
"""

external_issue_template = """
We’re sorry to hear about your negative experience due to factors beyond the airline's control, such as weather-related delays. Unfortunately, the airline is not liable in such cases. We appreciate your understanding.
"""

positive_feedback_template = """
Thank you for your positive feedback and for choosing to fly with us! We’re glad you had a good experience.
"""

# Create the main prompt chain for analyzing the trip experience
trip_prompt = PromptTemplate(input_variables=["trip_experience"], template=trip_template)
trip_chain = LLMChain(llm=llm, prompt=trip_prompt)

# Define individual response templates for each scenario
airline_issue_prompt = PromptTemplate.from_template(airline_issue_template) | llm
external_issue_prompt = PromptTemplate.from_template(external_issue_template) | llm
positive_feedback_prompt = PromptTemplate.from_template(positive_feedback_template) | llm

# Define the branching logic
branch = RunnableBranch(
    # Route based on detected keywords in the response
    (lambda x: "negative" in x["trip_summary"].lower() and "airline" in x["trip_summary"].lower(), airline_issue_prompt),
    (lambda x: "negative" in x["trip_summary"].lower() and ("weather" in x["trip_summary"].lower() or "beyond control" in x["trip_summary"].lower()), external_issue_prompt),
    positive_feedback_prompt
)

# Combine chains
full_chain = {"trip_summary": trip_chain, "trip_experience": lambda x: x["trip_experience"]} | branch

# Streamlit Interface
st.title("Trip Experience Summarizer")
st.header("Share with us your experience of the latest trip.")

# Textbox for user input
trip_experience = st.text_area("Please describe your trip:")

if trip_experience:
    # Run the full chain to get a categorized response
    result = full_chain.invoke({"trip_experience": trip_experience})
    st.write(result)
