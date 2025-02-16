import streamlit as st
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from typing import TypedDict, List

# Define the state structure for the agent
class State(TypedDict):
    text: str
    classification: str
    entities: List[str]
    summary: str

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Classification function
def classification_node(state: State):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Classify the following text into one of the categories: News, Blog, Research, or Other.\n\nText: {text}\n\nCategory:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    classification = llm.invoke([message]).content.strip()
    return {"classification": classification}

# Entity extraction function
def entity_extraction_node(state: State):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Extract all the entities (Person, Organization, Location) from the following text. Provide the result as a comma-separated list.\n\nText: {text}\n\nEntities:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    entities = llm.invoke([message]).content.strip().split(", ")
    return {"entities": entities}

# Summarization function
def summarization_node(state: State):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text in one short sentence.\n\nText: {text}\n\nSummary:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    summary = llm.invoke([message]).content.strip()
    return {"summary": summary}

# Build the agent workflow
workflow = StateGraph(State)
workflow.add_node("classification_node", classification_node)
workflow.add_node("entity_extraction", entity_extraction_node)
workflow.add_node("summarization", summarization_node)
workflow.set_entry_point("classification_node")
workflow.add_edge("classification_node", "entity_extraction")
workflow.add_edge("entity_extraction", "summarization")
workflow.add_edge("summarization", END)
app = workflow.compile()

# Streamlit app interface
st.title("AI Text Analysis Agent")
text_input = st.text_area("Enter text for analysis:")

if st.button("Analyze"):
    if text_input:
        state_input = {"text": text_input}
        result = app.invoke(state_input)

        st.subheader("Results")
        st.write("**Classification:**", result["classification"])
        st.write("**Entities:**", ", ".join(result["entities"]))
        st.write("**Summary:**", result["summary"])
    else:
        st.warning("Please enter some text to analyze.")
