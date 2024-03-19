import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
import base64
import os
# from io import BytesIO, StringIO
# from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
import markdown
import pdfkit

from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

# vectore store FAISS
from langchain_community.vectorstores import FAISS

# Output Parser
from langchain.output_parsers.structured import ResponseSchema, StructuredOutputParser

os.environ["GOOGLE_API_KEY"] = st.secrets.google_creds.GOOGLE_API_KEY

st.set_page_config(page_title="Your Book Note Creator",
                   page_icon=":bookmark:")
st.title("AI NotesBot: Your Interactive Book Notes Creator :bookmark::link:")
llm = GoogleGenerativeAI(model = "gemini-pro", temperature = 0)

map_prompt = """
Use following pieces of context to answer the question/instruction at the end. Generate answer with pretty html text.

```{context}```
Question: {question}
"""
map_prompt_template = PromptTemplate(template=map_prompt, 
                                     input_variables=["context", "question"])

chain_type_kwargs = {"prompt": map_prompt_template}

# Preprocessing for Output
genre_schema = ResponseSchema(name = "book_genre",
                              description = "Genre of the book",
                              type = "string"
                             )
chapter_count_schema = ResponseSchema(name = "chapter_count",
                                      description = "Number of chapters in the book",
                                      type = "integer"
                                     )

response_schemas = [genre_schema, 
                    chapter_count_schema]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()

def remove_top_margin():
    st.markdown(
        """
            <style>
                .appview-container .main .block-container {{
                    padding-top: {padding_top}rem;
                    padding-bottom: {padding_bottom}rem;
                    }}

            </style>""".format(
            padding_top=1, padding_bottom=5
        ),
        unsafe_allow_html=True
    )
    
def initialize_sessions():
    #Initialise the key in session state
    if "clicked" not in st.session_state:
        st.session_state.clicked = {1:False}
        
def click_create(button):
    # st.session_state.page = 'recommendations'
    if "raw_text" in st.session_state:
        del st.session_state.raw_text
        
    if "summary" in st.session_state:
        del st.session_state.summary

    if "full_summary" in st.session_state:
        del st.session_state.full_summary

    # if "content" in st.session_state:
    #     del st.session_state.content
        
    # if "messages" in st.session_state:
    #     del st.session_state.messages
    
    st.session_state.clicked[button] = True
    
def change_upload_file():
    if "raw_text" in st.session_state:
        del st.session_state.raw_text
        
    if "summary" in st.session_state:
        del st.session_state.summary

    if "full_summary" in st.session_state:
        del st.session_state.full_summary    

def get_pdf_text(pdf_doc):
    text=""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text+= page.extract_text()
    
    text = text.replace('\t', ' ')
    return text

def get_text_docs(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "\t"], 
        chunk_size=10000, 
        chunk_overlap=3000)
    docs = text_splitter.create_documents([text])
    return docs

def get_vector_store(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(docs, embedding = embeddings)
    return vector_store

# # Function to convert Markdown to HTML
# def markdown_to_html(markdown_text):
#     return markdown.markdown(markdown_text)

# Function to convert Markdown to HTML
def markdown_to_html(markdown_text):
    return markdown.markdown(markdown_text)

# Function to generate PDF from HTML content
def generate_pdf(html_content, output_file, wkhtmltopdf_path):
    pdf_options = {'quiet': ''}
    config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
    # pdf_stream = BytesIO()
    pdfkit.from_string(html_content, output_file, 
                       options=pdf_options, 
                       configuration=config)

def main():
    try:
        remove_top_margin()
        initialize_sessions()
        side_con = st.sidebar.container()
        uploaded_file = side_con.file_uploader("Upload a book..", 
                                            type = ["pdf"],
                                            key = "upload_book",
                                            on_change = change_upload_file
                                            )
        if uploaded_file is not None:
            side_con.button("Summarize", on_click = click_create, args=[1])
            
        if st.session_state.clicked[1]:
            if "raw_text" not in st.session_state:
                st.session_state.raw_text = get_pdf_text(uploaded_file)
            
            if st.session_state.raw_text is not None:
                num_tokens = llm.get_num_tokens(st.session_state.raw_text)
                print (f"This book has {num_tokens} tokens in it")
                
                docs = get_text_docs(st.session_state.raw_text)
                num_documents = len(docs)
                print (f"Now our book is split up into {num_documents} documents")

                ## create and get vectorstore
                vectorstore = get_vector_store(docs)
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(),
                    chain_type_kwargs=chain_type_kwargs
                )
                
                summarize_instruction = """
                Retrieve all the contents of the book provided as context.
                Your goal is to give a summary of the book so that a reader will have a full understanding of what is mentioned.
                Start with mentioning the title of the book and make sure the summary is of atleast 3000 words.
                """            
                if "summary" not in st.session_state:
                    st.session_state.summary = \
                        qa_chain.invoke(summarize_instruction)
                        
                if st.session_state.summary is not None:
                    
                    # get the book genre & chapters count
                    book_details_instruction = f"""Provide the total_chapter count and the book genre in format mentioned below
                    {format_instructions}
                    """
                    book_details = qa_chain.invoke(book_details_instruction)
                    book_details_dict = output_parser.parse(book_details["result"])
                    
                    if "full_summary" not in st.session_state:
                        # Generate full summary document
                        st.session_state.full_summary = \
                            st.session_state.summary["result"]
                        
                        for i in range(1, book_details_dict["chapter_count"]+1):
                            summarize_instruction = f"""
                            Summarize Chapter {i} of the book in atleast 2000 words and in full detail so that a reader will have a full 
                            understanding of what is mentioned without missing any important concept.
                            """
                            chapter_summary = qa_chain.invoke(summarize_instruction)["result"]
                            #print(chapter_summary)
                            st.session_state.full_summary += \
                                f"\n\n{chapter_summary}"
                    
                    reference_instruction = """
                    List all the links, books, papers, videos, podcasts referenced in the book
                    """
                    references =  qa_chain.invoke(reference_instruction)["result"]
                    if references:
                        st.session_state.full_summary += \
                                f"\n\n{references}"               
                    
                    if st.session_state.full_summary is not None:
                        full_summary_text = st.session_state.full_summary
                        
                        # Display markdown text
                        st.markdown(full_summary_text)
                        
                        # Convert Markdown to HTML
                        html_content = markdown_to_html(full_summary_text)
                        
                        wkhtmltopdf_path = '/usr/bin/wkhtmltopdf'
                        output_file = "./output/book_summary.pdf"
                        generate_pdf(html_content,
                                     output_file,
                                     wkhtmltopdf_path)
                        # href = f'<a href="{output_file}" download>Click here to download Summary PDF</a>'
                        # print(href)
                        # st.markdown(href, unsafe_allow_html=True)
                        
                        # markdown_to_pdf(full_summary_text, "book_summary.pdf")
                        # st.markdown(f'<a href="book_summary.pdf" download>Download Summary Document</a>', unsafe_allow_html=True)                        
                        # b64 = base64.b64encode(st.session_state.full_summary.encode()).decode()
                        # st.markdown(f'<a href="data:file/pdf;base64;{b64}" download="book_summary.pdf">Download Book Summary</a>', unsafe_allow_html=True)
                        with open(output_file, "rb") as f:            
                            st.download_button(
                                label = "Download Summary Document",
                                data = f,
                                file_name = "summary_doc.pdf",
                                mime = "application/pdf",
                                key  = "download_summary"
                                )
    except Exception as e:
        st.error(f"Error:{str(e)}")


        
        

        
        ### Embed Documents

        
        #print(raw_text[:50])
        
# Call the main function
if __name__ == "__main__":
    main()        