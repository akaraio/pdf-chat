# pdf-chat

The code creates a Streamlit application that allows users to upload a PDF document and interact with it using natural language. The application uses various libraries, including langchain, transformers, and PyPDFLoader, to load the PDF document, split it into chunks, add them to an in-memory vector store, and use a Hugging Face pipeline to generate answers to user queries based on the context provided by the relevant documents. The code also includes a chat interface where users can enter their questions and receive answers from the AI model.
