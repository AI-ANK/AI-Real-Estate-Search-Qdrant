# Streamlit application for New York Housing Market Explorer

# Required imports
import streamlit as st
import pandas as pd
from llama_index import SimpleDirectoryReader, ServiceContext, StorageContext, VectorStoreIndex
from llama_index.llms import OpenAI
from llama_index.embeddings import FastEmbedEmbedding
from qdrant_client import QdrantClient
import json
import os
from sqlalchemy import create_engine
from llama_index import SQLDatabase, ServiceContext
from llama_index.indices.struct_store import NLSQLTableQueryEngine
from pathlib import Path
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.query_engine import (
    SQLAutoVectorQueryEngine,
    RetrieverQueryEngine,
)
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.indices.vector_store import VectorIndexAutoRetriever

from llama_index.indices.vector_store.retrievers import (
    VectorIndexAutoRetriever,
)
from llama_index.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.query_engine.retriever_query_engine import (
    RetrieverQueryEngine,
)

from sqlalchemy import text, event

st.set_page_config(layout="wide")
write_dir = Path("textdata")

# Initialize Qdrant client
client = QdrantClient(
    url=os.environ['QDRANT_URL'], 
    api_key=os.environ['QDRANT_API_KEY'],
)

# Initialize LLM and embedding model
llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")
embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
service_context = ServiceContext.from_defaults(chunk_size_limit=1024, llm=llm, embed_model=embed_model)

vector_store = QdrantVectorStore(client=client, collection_name="housing2")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

#Create vector indexes and store in Qdrant. To be run only once in the beginning
#from llama_index import VectorStoreIndex
#index = VectorStoreIndex.from_documents(documents, vector_store=vector_store, service_context=service_context, storage_context=storage_context)

# Load the vector index from Qdrant collection
index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context
)


# Function to extract and format text data from a dataframe row
def get_text_data(data):
    return f"""   
    BROKERTITLE: {data['BROKERTITLE']}
    TYPE: {data['TYPE']}
    PRICE: {data['PRICE']}
    BEDS: {data['BEDS']}
    BATH: {data['BATH']}
    PROPERTYSQFT: {data['PROPERTYSQFT']}
    ADDRESS: {data['ADDRESS']}
    STATE: {data['STATE']}
    MAIN_ADDRESS: {data['MAIN_ADDRESS']}
    ADMINISTRATIVE_AREA_LEVEL_2: {data['ADMINISTRATIVE_AREA_LEVEL_2']}
    LOCALITY: {data['LOCALITY']}
    SUBLOCALITY: {data['SUBLOCALITY']}
    STREET_NAME: {data['STREET_NAME']}
    LONG_NAME: {data['LONG_NAME']}
    FORMATTED_ADDRESS: {data['FORMATTED_ADDRESS']}
    LATITUDE: {data['LATITUDE']}
    LONGITUDE: {data['LONGITUDE']}
    """
def create_text_and_embeddings():
    # Write text data to 'textdata' folder and creating individual files
    if write_dir.exists():
        print(f"Directory exists: {write_dir}")
        [f.unlink() for f in write_dir.iterdir()]
    else:
        print(f"Creating directory: {write_dir}")
        write_dir.mkdir(exist_ok=True, parents=True)

    for index, row in df.iterrows():
        if "text" in row:
            file_path = write_dir / f"Property_{index}.txt"
            with file_path.open("w") as f:
                f.write(str(row["text"]))
        else:
            print(f"No 'text' column found at index {index}")

    print(f"Files created in {write_dir}")
#create_text_and_embeddings()   #execute only once in the beginning

@st.cache_data
def load_data():
    if write_dir.exists():
        reader = SimpleDirectoryReader(input_dir="textdata")
        documents = reader.load_data()
    return documents

documents = load_data()

# Streamlit UI setup
st.title('New York Housing Market Explorer')

# Load the dataset
df_file_path = 'NY-House-Dataset.csv'  # Path to the csv file
if os.path.exists(df_file_path):
    df = pd.read_csv(df_file_path)
    df["text"] = df.apply(get_text_data, axis=1)
    st.dataframe(df)  # Display df in the UI
else:
    st.error("Data file not found. Please check the path and ensure it's correct.")

# Input from user
user_query = st.text_input("Enter your query:", "Suggest 3 houses in Manhattan brokered by compass.")

# Define the options for the radio button
options = ['Simple: Qdrant Similarity Search + LLM Call (works well for filtering type of queries)', 'Advanced: Qdrant Similarity Search + Llamaindex Text-to-SQL']

# Create a radio button for the options
selection = st.radio("Choose an option:", options)

# Processing the query
if st.button("Submit Query"):
    # Execute different blocks of code based on the selection
    if selection == 'Simple: Qdrant Similarity Search + LLM Call (works well for filtering type of queries)':
        # Part 1, semantic search + LLM call
        # Generate query vector
        query_vector = embed_model.get_query_embedding(user_query)
        # Perform search with Qdrant
        response = client.search(collection_name="housing2", query_vector=query_vector, limit=10)
        # Processing and displaying the results
        text = ''
        properties_list = []  # List to store multiple property dictionaries
        for scored_point in response:
            # Access the payload, then parse the '_node_content' JSON string to get the 'text'
            node_content = json.loads(scored_point.payload['_node_content'])
            text += f"\n{node_content['text']}\n"    
            # Initialize a new dictionary for the current property
            property_dict = {}
            for line in node_content['text'].split('\n'):
                if line.strip():  # Ensure line is not empty
                    key, value = line.split(': ', 1)
                    property_dict[key.strip()] = value.strip()
            # Add the current property dictionary to the list
            properties_list.append(property_dict)

        # properties_list contains all the retrieved property dictionaries
        with st.status("Retrieving points/nodes based on user query", expanded = True) as status:
            for property_dict in properties_list:
                st.json(json.dumps(property_dict, indent=4))
                print(property_dict)
            status.update(label="Retrieved points/nodes based on user query", state="complete", expanded=False)
        
        with st.status("Simple Method: Generating response based on Similarity Search + LLM Call", expanded = True) as status:
            prompt_template = f"""
                Using the below context information respond to the user query.
                context: '{properties_list}'
                query: '{user_query}'
                Response structure should look like this:
                *Detailed Response*
                
                *Relevant Details in Table Format*
                
                Also, generate the latitude and longitude for all the properties included in the response in JSON object format. For example, if there are properties at 40.761255, -73.974483 and 40.7844489, -73.9807532, the JSON object should look like this limited with 3 backticks. JUST OUTPUT THE JSON, NO NEED TO INCLUDE ANY TITLE OR TEXT BEFORE IT:

                ```[
                    {{
                        "latitude": 40.761255,
                        "longitude": -73.974483
                    }},
                    {{
                        "latitude": 40.7844489,
                        "longitude": -73.9807532
                    }}
                ]```

                """
            llm_response = llm.complete(prompt_template)
            response_parts = llm_response.text.split('```')
            st.markdown(response_parts[0])

    elif selection == 'Advanced: Qdrant Similarity Search + Llamaindex Text-to-SQL':
        #Part 2, Semantic Search + Text-to-SQL
        with st.status("Advanced Method: Generating response based on Qdrant Similarity Search + Llamaindex Text-to-SQL", expanded = True):
            df2 = df.drop('text', axis=1)
            #Create a SQLite database and engine
            engine = create_engine("sqlite:///NY_House_Dataset.db?mode=ro", connect_args={"uri": True})

            @event.listens_for(engine, "before_cursor_execute")
            def receive_before_cursor_execute(conn, cursor, statement, params, context, executemany):
                if statement.strip().lower().startswith(("insert", "update", "delete")):
                    # Instead of raising an error, log or print a warning
                    print("Read-only mode: Modification attempt blocked.")
                    # Prevent execution by skipping the operation
                    raise Exception("Attempted to modify the database in read-only mode.")

            sql_database = SQLDatabase(engine)
            #Convert the DataFrame to a SQL table within the SQLite database
            df2.to_sql('housing_data_sql', con=engine, if_exists='replace', index=False)

            #Build sql query engine
            sql_query_engine = NLSQLTableQueryEngine(
                sql_database=sql_database
            )

            vector_store_info = VectorStoreInfo(
                content_info="Housing data details for NY",
                metadata_info = [
                    MetadataInfo(name="BROKERTITLE", type="str", description="Title of the broker"),
                    MetadataInfo(name="TYPE", type="str", description="Type of the house"),
                    MetadataInfo(name="PRICE", type="float", description="Price of the house"),
                    MetadataInfo(name="BEDS", type="int", description="Number of bedrooms"),
                    MetadataInfo(name="BATH", type="float", description="Number of bathrooms"),
                    MetadataInfo(name="PROPERTYSQFT", type="float", description="Square footage of the property"),
                    MetadataInfo(name="ADDRESS", type="str", description="Full address of the house"),
                    MetadataInfo(name="STATE", type="str", description="State of the house"),
                    MetadataInfo(name="MAIN_ADDRESS", type="str", description="Main address information"),
                    MetadataInfo(name="ADMINISTRATIVE_AREA_LEVEL_2", type="str", description="Administrative area level 2 information"),
                    MetadataInfo(name="LOCALITY", type="str", description="Locality information"),
                    MetadataInfo(name="SUBLOCALITY", type="str", description="Sublocality information"),
                    MetadataInfo(name="STREET_NAME", type="str", description="Street name"),
                    MetadataInfo(name="LONG_NAME", type="str", description="Long name of the house"),
                    MetadataInfo(name="FORMATTED_ADDRESS", type="str", description="Formatted address"),
                    MetadataInfo(name="LATITUDE", type="float", description="Latitude coordinate of the house"),
                    MetadataInfo(name="LONGITUDE", type="float", description="Longitude coordinate of the house"),
                ],
            )
            vector_auto_retriever = VectorIndexAutoRetriever(
                index, vector_store_info=vector_store_info
            )

            retriever_query_engine = RetrieverQueryEngine.from_args(
                vector_auto_retriever, service_context=service_context
            )

            sql_tool = QueryEngineTool.from_defaults(
                query_engine=sql_query_engine,
                description=(
                    "Useful for translating a natural language query into a SQL query over"
                    " a table 'houses', containing prices of New York houses, providing valuable insights into the real estate market in the region. It includes information such as broker titles, house types, prices, number of bedrooms and bathrooms, property square footage, addresses, state, administrative and local areas, street names, and geographical coordinates."
                
                ),
            )
            vector_tool = QueryEngineTool.from_defaults(
                query_engine=retriever_query_engine,
                description=(
                    f"Useful for answering questions about different housing listings in New York. Use this to refine your answers"
                ),
            )

            query_engine = SQLAutoVectorQueryEngine(
                sql_tool, vector_tool, service_context=service_context
            )
            response = query_engine.query(f"{user_query}+. Provide a detailed response and include lONG_NAME, name of broker, number of beds, number of baths, propertysqft and FORMATTED_ADDRESS. ALWAYS USE LIKE in WHERE CLAUSE. ALWAYS RESPOND IN WELL FORMATTED MARKDOWN")
            st.markdown(response.response)
