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

#### Define the Prompt and Chain
trip_template = """Summarize the following trip experience briefly:{trip_experience}"""
prompt_template = PromptTemplate(input_variables=["trip_experience"], 
                                 template=trip_template)
chain = LLMChain(llm=llm, prompt=prompt_template)

### Title of App
st.title("Trip Experience Summarizer")
st.header("Share with us your experience of the latest trip.")

### Textbox for user input
trip_experience = st.text_area("Please describe your trip:")

if trip_experience:
        response = chain.run(trip_experience)
        st.write(response)

        if "negative" in response.lower() and ("airline" in response.lower() or "luggage" in response.lower() or "delay" in response.lower()):
          st.warning("We're sorry to hear about your negative experience. Our customer service team will contact you soon to resolve the issue or provide compensation.")

