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

# Define the LangChain Prompt and Chain
prompt_template = PromptTemplate(input_variables=["trip_experience"], 
                                 template="Summarize the following trip experience briefly:\n\n{trip_experience}")
chain = LLMChain(llm=llm, prompt=prompt_template)

# Streamlit App
st.title("Trip Experience Summarizer")
st.header("Share with us your experience of the latest trip.")

# Textbox for user input
trip_experience = st.text_area("Describe your trip:")

# Button to process input
if st.button("Submit"):
    if trip_experience:
        # Run the input through the LangChain
        summary = chain.run(trip_experience=trip_experience)
        
        # Display the result
        st.subheader("Summary of Your Trip Experience:")
        st.write(summary)
    else:
        st.warning("Please describe your trip before submitting!")
