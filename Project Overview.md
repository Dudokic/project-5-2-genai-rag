1. Dataset Selection
Collect relevant documents for sustainability reporting like ESRS, GRI, and other European reporting guidelines as your primary dataset.
For supplementary knowledge, you could scrape or access open scientific publications on environmental, social, and governance (ESG) topics or other sustainability studies.
2. Exploratory Data Analysis (EDA)
Analyze the structure and content of these documents. Look at the complexity of terms, patterns in regulatory language, and key sections that you may want your RAG system to prioritize.
Document sections on specific reporting standards, compliance requirements, and examples, which may serve as valuable chunks later.
3. Embedding and Storing Chunks
3.A Embedding: Use models like all-MiniLM-L6-v2 or similar to create embeddings for each document chunk. Sentence Transformers work well for technical language, and you could experiment with OpenAI embeddings if you need broader vocabulary handling.
3.B Connection to Vector DB: Store these embeddings in a database like ChromaDB, which will allow your system to search for and retrieve relevant content quickly.
Consider adding fields to your database for document type (e.g., regulation, guideline, scientific paper) to improve query relevance.
4. Connecting to LLM
Set up access to OpenAI or another LLM API, possibly integrating the retrieved document chunks to give targeted, well-informed responses.
LangChain and LlamaIndex can be helpful here to streamline this retrieval and generation process, especially when dealing with multi-part documents or multi-step reasoning.
5. Evaluation
Generate test queries based on common sustainability reporting questions, perhaps referencing real reporting standards for consistency.
For the bonus, you could use an LLM to evaluate the answers, helping you iterate on accuracy and relevance.
6. Deployment (Bonus)
If deployed, this could be a powerful tool for users needing sustainability reports, allowing for easy queries about compliance and sustainability metrics directly aligned with ESRS/GRI standards.