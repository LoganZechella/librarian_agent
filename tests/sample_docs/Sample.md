# Librarian Agent

An AI-powered knowledge base agent for searching, ingesting, and retrieving documents using both keyword and semantic search. Librarian supports local and S3 document storage, leverages MongoDB Atlas for fast vector and keyword search, and provides structured, citation-rich responses for research and knowledge management.

## Features

- **Simple lookup**: Find documents by keyword or phrase (e.g., “Find the PDF of Project X spec.”)
- **Contextual search**: Retrieve information by meaning, not just keywords (e.g., “Show me all design decisions about caching.”)
- **Browsing**: List documents by topic, tag, or type (e.g., “List all whitepapers on Topic Y.”)
- **Document ingestion**: Add new documents (PDF, Word, Markdown) from local disk or S3 to the knowledge base
- **Hybrid retrieval**: Combines MongoDB Atlas Vector Search (semantic) and text index (keyword)
- **Structured responses**: Outputs include Summary, Results (with citations), and Next Steps

## Core Use Cases

- Simple document lookup
- Contextual and semantic search
- Browsing by topic or tag
- Ingesting and updating documents in the KB

## Architecture

- **Language**: Python
- **Framework**: OpenAI Agents SDK
- **Database**: MongoDB Atlas (vector and text search)
- **Storage**: Local filesystem and/or AWS S3
- **Core Tools**:
  - `text_search`: Keyword search via MongoDB Atlas text index
  - `semantic_search`: Vector search via MongoDB Atlas
  - `read_document`: Load and extract text from PDF, Word, Markdown (local or S3)
  - `ingest_document`: Chunk, embed, and upsert documents into the KB

## Setup & Installation

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd librarian_agent
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set environment variables**
   - `MONGODB_ATLAS_URI`: MongoDB Atlas connection string
   - `MONGODB_DB`: (optional) Database name (default: `librarian_kb`)
   - `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`: (if using S3)
   - `OPENAI_API_KEY`: For embedding and agent reasoning

   You can use a `.env` file for convenience.

## Usage

### Example: Running the Agent

```python
from librarian.agent import librarian, Runner

user_query = "Show me all design decisions about caching."
result = Runner.run_sync(librarian, user_query)
print(result.final_output)
```

### Example Queries

- "Find the PDF of Project X spec."
- "List all whitepapers on Topic Y."
- "Add new doc to KB."

## Response Format

Responses are structured as JSON with the following sections:
- **Summary**: High-level answer or synthesis
- **Results**: List of relevant excerpts, each with citation (source, page, etc.)
- **Next Steps**: Suggested follow-ups or actions

Example citation: `[1] file.pdf – p.10`

## Extending

- Add new tools by defining a function and decorating with `@function_tool`
- Update the agent’s tool list in `librarian/agent.py`
- Adjust chunking, embedding, or retrieval logic in `librarian/tools.py`

## License

MIT License

## Credits

- Built with [OpenAI Agents SDK](https://github.com/openai/openai-agents)
- Uses [MongoDB Atlas](https://www.mongodb.com/atlas) for storage and search

---