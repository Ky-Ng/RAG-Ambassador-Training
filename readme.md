# RAG Ambassador Training

___
## Introduction
As a USC Dornsife Student Ambassador, there are so many wonderful text-based training resources; however, there is one big problem:

1) Handbooks are not interactive
2) And therefore cannot simulate ambassador interactions for critical training events like “Tough Questions” or Diversity Equity and Inclusion (DEI) training

Note: Tough Questions is a simulated prospective student to ambassador conversation where ambassadors practice answering sensitive or personal questions to their experience at USC
___
## Retrieval Augmentation Generation (RAG) Solution:
1. Collect Training Resources (handbooks, presentation slide decks, and personal notes)
2. Ingest resources into a Vectorized Database (Pinecone) for semantic retrieval
3. Use entries from training resources as few-shot prompts for Large Language Model (Cohere `Command` and `Cohere Embeddings`)
___
## Project Phases
1) Jupyter Notebook prototype of Documents as Files and generating responses with LLM RAG
2) Deploy Notebook prototype as a Backend API using Serverless Architecture:
	1) POST request to add Documents to Pinecone Database
	2) GET request to receive LLM response with Few Shot Prompts
3) Deploy a frontend that calls the Serverless Endpoints and visualizes RAG
	1) Document Parsing on Client Side and sending sentences as text string over HTTPs
	2) UI for visualizing few-shot prompt sentences and which sentence was used to answer Yes No Questions (More elaboration is needed for this in the future)
___
## Documentation
Below is a high level overview of the project. 

For more detailed documentation of the engineering process behind building this, check out [bit.ly/rag-docs](https://bit.ly/rag-docs)  which is a constantly updated Engineering Notebook with nightly builds (just like the Cohere `Command` Model!)
___
## Technologies Used
### Cohere `Command`
- Cohere's Command model (similar to an OpenAI GPT) is a general purpose language model
	- For RAG Ambassador Training, we use `Command` to answer questions a Dornsife Ambassador may have given a `context`
### Pinecone
- Pinecone is a Vector Database which similar to your normal relational database (RDB) can hold text sentences but in a special way
	- Pinecone stores sentences as `embeddings` rather than plain strings so that they can later be retrieved by `semantic search`
	- Semantic Search refers to the process of retrieving sentences by a similarity score (such as the angle between high-dimensional embedding vector)
- Serverless Architecture
	- Rather than paying for "always on" Pinecone Pods, this project uses Pinecone Serverless Architecture where you are only payed per usage
		- Note: Pinecone serverless is only offered in `aws us-west-2` 
## Cohere Embeddings
- Pinecone doesn't accept any ol' plain sentences, it needs a numerical representation of the sentence called an embedding
- In this project, we use Cohere embeddings for semantically comparing and storing sentences
### LangChain
But wait, we have a problem! 
1) Cohere Embeddings can turn Natural Language documents into numerical representations
2) Pinecone can store those embeddings
3) Cohere `Command`can respond with those sentences(also known as `context` or `documents`)

But how can we string it all together?
- Using LangChain and the `LangChain Expression Language (LCEL)` we can easily build up these NLP pipeline components and string them together modularly using `|` (read as "pipe") operators
___
## Magic `|`'s
1) Here we create individual chain components:
1) Cohere `Command` model for Question and Answering
2) Pinecone Vector Store Retreiver using Semantic Search
3) Templated prompt with slotted in `question` and `context`

```python
llm = Cohere(model="command")

retriever = vectorstore_a.as_retriever()

prompt_str = """
Question: {question}

Please answer the question above using only the context provided.
Context: {context}

Answer:
"""
```

2) Then we string these modular components together using `LangChain Expression Language (LCEL)`
```python
prompt =  PromptTemplate.from_template(prompt_str)

retrieve_question_and_context = RunnableParallel({"question": RunnablePassthrough(), "context": retriever})

chain = retrieve_question_and_context | prompt | llm | StrOutputParser()
```

3) We can now easily invoke calls to the pipeline using the `.invoke()` method

```python
out = chain.invoke("Why is USC good?")
print(out)
```
___
## Results

1) `out = chain.invoke("Why is USC good?")`
```
USC is known for its intense and difficult courses in linguistics and CS, but also for its incredibly warm summer. It is a highly acclaimed institution for higher education, world-renowned as the best school in the world. 
```

2) `out = chain.invoke("Why is USC bad?")`
```
Some reasons why some people think USC is bad are because of the difficulty of its linguistics and CS classes, and how Complex Analysis is complex and that accounting is not accounted for.
```

3) `out = chain.invoke("Tell me about USC's Dance Program?")`
```
USC's dance program is the best in the world and is housed in the equally acclaimed Kauffman school of dance. 

An unnamed source remarks that it is better than the Juliard [sic] school. 

It is part of the College of Letters of Arts and Sciences called Dornsife Viterbi which highlights the liberal arts and sciences component of the University of Southern California.
```
___
## What's Next
- Now that we've finished Phase 1 (Jupyter Notebook Prototyping), it's now time to move to Phase 2 to deploy the notebook as a `Docker Container` available as an `AWS` backend endpoint