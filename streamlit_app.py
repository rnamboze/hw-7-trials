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
llm = OpenAI(openai_api_key="YOUR_OPENAI_API_KEY")


st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
