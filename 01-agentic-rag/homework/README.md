# Agentic RAG Homework - Concept Guide

> A beginner-friendly explanation of every concept in this homework, from the ground up.

---

## Table of Contents

1. [What Problem Are We Solving?](#what-problem-are-we-solving)
2. [RAG: Retrieval-Augmented Generation](#rag-retrieval-augmented-generation)
3. [Document Indexing & Search](#document-indexing--search)
4. [Token Counting](#token-counting)
5. [Chunking](#chunking)
6. [Agentic RAG & Tool Calling](#agentic-rag--tool-calling)
7. [The Agentic Loop](#the-agentic-loop)
8. [How It All Connects](#how-it-all-connects)

---

## What Problem Are We Solving?

LLMs (like ChatGPT, DeepSeek, etc.) are trained on a fixed dataset. They **don't know**:

- Your company's internal docs
- A specific course's lessons
- Anything that happened after their training cutoff

But we want them to answer questions **about** that private knowledge.

**Two bad approaches:**
- ❌ **Stuff everything into the prompt** → LLMs have a context limit (you can't fit a whole wiki in one message)
- ❌ **Fine-tune the model** → Expensive, slow, hard to update

**The good approach:** **RAG** — let the LLM search your documents first, then answer based on what it found.

---

## RAG: Retrieval-Augmented Generation

RAG is a 3-step pipeline:

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   RETRIEVE  │ ──▶ │   AUGMENT   │ ──▶ │   GENERATE   │
│             │     │              │     │              │
│ Search your │     │ Paste the    │     │ LLM reads    │
│ documents   │     │ results into │     │ the context  │
│ for relevant│     │ the prompt   │     │ and answers  │
│ passages    │     │ as context   │     │ the question │
└─────────────┘     └──────────────┘     └──────────────┘
```

### Step-by-step with an example:

**User asks:** *"How does the agentic loop work?"*

**1. Retrieve:** Search our lesson documents and find the top 5 most relevant passages.

**2. Augment:** Build a prompt like:
```
You are a teaching assistant.

Context:
[passage from lesson 14 about agentic loops]
[passage from lesson 03 about RAG]
[passage from lesson 14 about tool calling]
...

Question: How does the agentic loop work?
```

**3. Generate:** Send this prompt to the LLM. It reads the context and gives an informed answer.

> **Key insight:** The LLM never "memorized" your documents. It reads them at query time, just like a human would look something up before answering.

---

## Document Indexing & Search

Before you can search, you need to **index** your documents.

### What is an index?

Think of it like a book's index at the back. Instead of reading every page to find "agentic loop", you look it up in the index and jump straight to the relevant pages.

In our code, we use **minsearch**, a lightweight text search library:

```python
index = minsearch.Index(
    text_fields=["content"],      # Search inside the document text
    keyword_fields=["filename"]   # Filter by exact filename
)
index.fit(documents)              # Build the index from all documents
```

### How the search works

```python
results = index.search("How does the agentic loop keep calling the model?")
```

minsearch does **keyword-based search** — it looks at the words in your query and finds documents containing similar words. It's like a simplified version of what Google does.

Each result comes back with a relevance score. We take the top results and feed them into our RAG prompt.

---

## Token Counting

### What is a token?

LLMs don't read text character-by-character. They read **tokens** — small chunks of text.

```
"Hello world"     →  ["Hello", " world"]         → 2 tokens
"Agentic RAG"     →  ["Ag", "entic", " R", "AG"] → 4 tokens
"🚀"              →  ["🚀"]                       → 1 token
```

There's no perfect 1:1 rule, but roughly:
- **1 token ≈ 4 characters** in English
- **1 token ≈ ¾ of a word**
- **100 tokens ≈ 75 words**

Different models use different tokenizers, so the same text may have different token counts across providers (which is why your DeepSeek result differed from OpenAI).

### Why does token count matter?

Two reasons:

1. **Cost:** LLM APIs charge **per token**. More tokens = more money.
2. **Context limit:** Every model has a maximum context window (e.g., 128K tokens). Exceed it and you get an error.

### In our homework

- **Q3** measures how many tokens we send when using full documents as context (~10,552 tokens)
- **Q5** measures how many tokens we send after chunking (~4,890 tokens)
- **The difference is real money saved** at scale

---

## Chunking

### The problem

Lesson pages are long — some are thousands of characters. When we search and retrieve a whole page, we might pull in a lot of **irrelevant text** alongside the one paragraph that actually answers the question.

```
┌─────────────────────────────────────────┐
│ Lesson 14: Agentic Loop                  │
│                                          │
│ [Intro - 500 chars of intro stuff]       │  ← not relevant
│                                          │
│ [Background - 800 chars of history]      │  ← not relevant
│                                          │
│ [THE ANSWER - 200 chars you need]        │  ← ✅ This is what you want
│                                          │
│ [Summary - 600 chars of recap]           │  ← not relevant
│                                          │
│ [Exercises - 900 chars of homework]      │  ← not relevant
└─────────────────────────────────────────┘
         ↑ You retrieve ALL of this (~3000 chars)
```

This wastes tokens and can confuse the LLM with irrelevant information.

### The solution: Chunking

Split each document into smaller, **overlapping** pieces (chunks). Index the chunks instead of whole documents.

```python
chunks = chunk_documents(documents, size=2000, step=1000)
```

Here's what `size=2000, step=1000` means:

```
Document: "ABCDEFGHIJ" (each letter = 200 chars)

Chunk 1: [A B C D E F G H I J]  ← chars 0-1999
                   Chunk 2:       [E F G H I J K L M N]  ← chars 1000-2999
                                   Chunk 3:       [I J K L M N O P Q R]  ← chars 2000-3999
```

- **size=2000**: Each chunk is 2000 characters
- **step=1000**: Move forward 1000 chars for the next chunk
- **Overlap**: 2000 - 1000 = 1000 chars of overlap between consecutive chunks

### Why overlap?

Without overlap, a sentence split across a boundary would be cut in half:

```
NO OVERLAP:           WITH OVERLAP:
Chunk 1: "...the model"   Chunk 1: "...the model decides"
Chunk 2: "decides to..."  Chunk 2: "decides to stop when..."
                         ↑ "decides" appears in both!
```

The overlap ensures no important passage is lost at the boundaries.

### The benefit

```
Before chunking:  Retrieve whole page → 10,552 tokens sent to LLM
After chunking:  Retrieve one chunk  →  4,890 tokens sent to LLM
                                    →  ~2× fewer tokens = ~2× cheaper
```

---

## Agentic RAG & Tool Calling

### Plain RAG's limitation

In plain RAG, **we** (the developer) decide what to search:

```python
# We hardcode the search query = the user's exact question
results = index.search("How does the agentic loop work, and how is it different from plain RAG?")
```

But what if the user asks a complex, multi-part question? A single search might not cover all aspects. A human would naturally:
1. Search for "agentic loop"
2. Read the results
3. Realize they need more info, search for "plain RAG"
4. Compare and synthesize

**Plain RAG can't do this.** It searches once and answers.

### Agentic RAG: Let the LLM decide

Instead of us controlling the search, we give the LLM a **search tool** and let it decide:
- **What** to search for
- **When** to search
- **How many times** to search
- **When** it has enough info to answer

### How tool calling works

LLMs support a feature called **function calling** (or **tool calling**). You tell the model:

> "Here's a tool called `search` that takes a query string and returns relevant documents. You can use it if you need information."

The model can then respond in two ways:

**1. It wants to use a tool:**
```json
{
  "tool_calls": [{
    "function": {
      "name": "search",
      "arguments": "{\"query\": \"agentic loop mechanism\"}"
    }
  }]
}
```

**2. It's ready to answer:**
```json
{
  "content": "The agentic loop works by..."
}
```

### The flow

```
User: "How does the agentic loop work, and how is it different from plain RAG?"
  │
  ▼
LLM thinks: "I need info about two things. Let me search for the agentic loop first."
  │
  ▼
LLM calls: search("agentic loop mechanism")
  │
  ▼
We run the search, return results to the LLM
  │
  ▼
LLM thinks: "Good info. But I also need to compare with plain RAG. Let me search for that."
  │
  ▼
LLM calls: search("plain RAG vs agentic RAG")
  │
  ▼
We run the search, return results to the LLM
  │
  ▼
LLM thinks: "Now I have enough context to give a complete answer."
  │
  ▼
LLM responds with the final answer
```

In your run, the agent called search **4 times** before answering. It autonomously decided it needed multiple searches to cover the question fully.

---

## The Agentic Loop

The "agentic loop" is the **while-true cycle** that keeps running until the LLM decides to stop:

```python
while True:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools_schema,
        tool_choice="auto"
    )
    
    if response.has_tool_calls:
        # LLM wants to search → execute the tool, add result to messages, loop again
        for tool_call in response.tool_calls:
            result = execute_tool(tool_call)
            messages.append(tool_result_message(result))
    else:
        # LLM gave a final answer → break out of the loop
        return response.content
```

```
┌──────────────────────────┐
│   Send messages to LLM   │
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│   Did it call a tool?    │
└───────────┬──────────────┘
       YES /   \ NO
        /       \
       ▼         ▼
┌────────────┐  ┌──────────────┐
│ Execute    │  │ Return the   │
│ the tool,  │  │ final answer │
│ add result │  │              │
│ to messages│  └──────────────┘
└─────┬──────┘
      │
      └──────▶ Loop back to top
```

**"Keep calling the model until it stops"** means: the loop continues as long as the model returns tool calls. When the model returns a regular text response (no tool calls), the loop stops.

---

## How It All Connects

Here's the full journey from beginner to agent:

```
Level 1: Raw LLM
  "Answer my question" → LLM guesses from training data
  ❌ Can't answer questions about your private documents

Level 2: RAG (Q2, Q3)
  "Answer my question" → Search docs → Paste results into prompt → LLM answers
  ✅ Can answer from your documents
  ❌ Searches once with the exact user query, might miss things

Level 3: RAG + Chunking (Q4, Q5)
  Same as Level 2, but documents are split into smaller pieces
  ✅ More precise retrieval, fewer tokens, cheaper
  ❌ Still searches once

Level 4: Agentic RAG (Q6)
  "Answer my question" → LLM decides what to search → searches → reads results → 
  decides if it needs more info → searches again → ... → answers
  ✅ Autonomous, multi-step research
  ✅ Handles complex, multi-part questions
  ❌ More tokens overall (multiple searches), slower, more expensive per query
```

Each level adds capability but also adds complexity. The right choice depends on your use case:

| Approach | Best For | Cost | Latency |
|----------|----------|------|---------|
| Raw LLM | General knowledge | Low | Fast |
| Plain RAG | Simple factual questions | Medium | Medium |
| RAG + Chunking | Large document collections | Medium-Low | Medium |
| Agentic RAG | Complex, multi-faceted questions | Higher | Slower |

---

## Glossary

| Term | Plain English |
|------|--------------|
| **LLM** | Large Language Model — an AI that generates text (ChatGPT, DeepSeek, etc.) |
| **RAG** | A technique where you search documents first, then ask the LLM to answer based on what you found |
| **Index** | A searchable data structure built from your documents |
| **Token** | The unit LLMs use to read text (~4 characters or ¾ of a word) |
| **Chunking** | Splitting long documents into smaller, overlapping pieces |
| **Tool Calling** | A feature where the LLM can ask your code to run a function (like search) |
| **Agent** | An LLM that can use tools and make decisions about what to do next |
| **Agentic Loop** | The while-loop that keeps sending messages to the LLM until it stops requesting tools |
| **Context Window** | The maximum number of tokens an LLM can process in one request |
| **Prompt** | The text message you send to an LLM |

---

*This homework walks you through building each level, from basic search (Q2) to a fully autonomous agent (Q6).*