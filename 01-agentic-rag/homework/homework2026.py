

import os
import json
from openai import OpenAI
import minsearch
from gitsource import GithubRepositoryDataReader, chunk_documents
from dotenv import load_dotenv          # <-- ADD THIS

load_dotenv()                            # <-- ADD THIS (must be before client creation)
# ==========================================
# Setup DeepSeek Client
# ==========================================
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

MODEL_NAME = "deepseek-v4-flash"

# ==========================================
# Preparation: Fetching the documents
# ==========================================
print("Fetching documents from GitHub (this may take a minute)...")
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

files = reader.read()
documents = []
for file in files:
    doc = file.parse()
    documents.append(doc)

# ==========================================
# Q1. How many lesson pages
# ==========================================
print(f"\n--- Q1 ---")
print(f"Number of lesson pages: {len(documents)}")

# ==========================================
# Q2. Indexing and searching
# ==========================================
print(f"\n--- Q2 ---")
index = minsearch.Index(
    text_fields=["content"],
    keyword_fields=["filename"]
)
index.fit(documents)

query_q2 = "How does the agentic loop keep calling the model until it stops?"
search_results_q2 = index.search(query_q2)
print(f"First result filename: {search_results_q2[0]['filename']}")

# ==========================================
# Q3. RAG
# ==========================================
print(f"\n--- Q3 ---")

def build_context(search_results):
    context = ""
    for res in search_results:
        context += f"Filename: {res['filename']}\nContent:\n{res['content']}\n\n"
    return context

prompt_template = """
You are a course teaching assistant.
Answer the student's question using the context provided.
If the answer is not in the context, say so.

Context:
{context}

Question: {question}
"""

def rag(query, search_index, model=MODEL_NAME):
    search_results = search_index.search(query)
    context = build_context(search_results)
    prompt = prompt_template.format(context=context, question=query)
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    answer = response.choices[0].message.content
    usage = response.usage
    return answer, usage

query_q3 = "How does the agentic loop keep calling the model until it stops?"
answer_q3, usage_q3 = rag(query_q3, index)
print(f"Input tokens (full docs): {usage_q3.prompt_tokens}")

# ==========================================
# Q4. Chunking
# ==========================================
print(f"\n--- Q4 ---")
chunks = chunk_documents(documents, size=2000, step=1000)
print(f"Number of chunks: {len(chunks)}")

# ==========================================
# Q5. RAG with chunking
# ==========================================
print(f"\n--- Q5 ---")
chunk_index = minsearch.Index(
    text_fields=["content"],
    keyword_fields=["filename"]
)
chunk_index.fit(chunks)

answer_q5, usage_q5 = rag(query_q3, chunk_index)
print(f"Input tokens (chunked): {usage_q5.prompt_tokens}")
ratio = usage_q3.prompt_tokens / usage_q5.prompt_tokens
print(f"Ratio (full/chunked): {ratio:.2f}x fewer tokens")

# ==========================================
# Q6. Turning it into an agent
# ==========================================
print(f"\n--- Q6 ---")

def search_tool(query: str):
    results = chunk_index.search(query)
    return build_context(results)

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "search_tool",
            "description": "Search the course lessons for relevant information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant lesson content.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

def run_agent(query, model=MODEL_NAME):
    messages = [
        {
            "role": "system",
            "content": "You're a course teaching assistant. Answer the student's question using the search tool. Make multiple searches with different keywords before answering."
        },
        {"role": "user", "content": query}
    ]
    
    search_count = 0
    
    while True:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools_schema,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        messages.append(response_message)
        
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                if tool_call.function.name == "search_tool":
                    search_count += 1
                    args = json.loads(tool_call.function.arguments)
                    search_result = search_tool(args["query"])
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": search_result
                    })
        else:
            return response_message.content, search_count

query_q6 = "How does the agentic loop work, and how is it different from plain RAG?"
answer_q6, search_count = run_agent(query_q6)
print(f"Agent called search {search_count} times.")
print(f"Final Answer snippet: {answer_q6[:150]}...")