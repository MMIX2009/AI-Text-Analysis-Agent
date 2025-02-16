## What is it?
A text analysis system built using LangGraph and LangChain. It performs three core tasks on a given text:

Classification: It classifies the text into categories like News, Blog, Research, or Other.
Entity Extraction: It extracts key entities (e.g., persons, organizations, locations) as a comma-separated list.
Summarization: It generates a concise summary of the text in one short sentence.

## How It Works:
The agent uses a graph-based workflow where each task is a node in the graph. The flow starts with classification, proceeds to entity extraction, and then ends with summarization.
It uses the ChatOpenAI model (gpt-4o-mini) to handle these tasks.
The model runs with a temperature of 0 to ensure deterministic and consistent outputs.

## Example Output:
When given text about OpenAI announcing GPT-4, the agent outputs:

Classification: News
Entities: OpenAI, GPT-4, GPT-3
Summary: OpenAI's upcoming GPT-4 model is a multimodal AI that aims for human-level performance, improved safety, and greater efficiency compared to GPT-3.

## Key Concepts:
State Management: Maintains context across tasks.
Decision-Making Framework: Decides the next steps based on intermediate results.
Tool Use: Selects the right tools for the right tasks based on the context.
The document emphasizes how agents like this can be extended to analyze various text types, from medical research papers to legal documents.
