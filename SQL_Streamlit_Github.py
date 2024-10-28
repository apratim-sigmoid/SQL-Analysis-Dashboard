import streamlit as st
import pandas as pd
from PIL import Image
import os
import re
import json
import tempfile
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.document_loaders.csv_loader import CSVLoader
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from rank_bm25 import BM25Okapi
from pydantic import BaseModel, ConfigDict
from dotenv import load_dotenv

# Configuration class
class SecretConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    api_key: str

load_dotenv()

# Default configurations
DEFAULT_DATA_PATH = "Data.csv"

def load_default_data():
    """Load the default CSV file if it exists."""
    try:
        if os.path.exists(DEFAULT_DATA_PATH):
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(DEFAULT_DATA_PATH, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                st.error("❌ Could not read the file with any standard encoding.")
                return None
                
            return df
        else:
            st.error(f"❌ Default data file '{DEFAULT_DATA_PATH}' not found!")
            return None
    except Exception as e:
        st.error(f"❌ Error loading default data: {str(e)}")
        return None

def process_dataframe_to_store_in_db(df):
    """Replace whitespace in column names with underscore."""
    for col in df.columns:
        df.rename({col: re.sub(r"\s+", "_", col)}, axis=1, inplace=True)
    return True

def create_sql_db(db_name='SQL_DB', df=None):
    """Store dataframe into SQLDatabase object with proper connection handling."""
    if "DBs" not in os.listdir():
        os.makedirs("DBs")
    
    db_path = f"DBs/{db_name}.db"
    
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except PermissionError:
            import time
            timestamp = int(time.time())
            db_path = f"DBs/{db_name}_{timestamp}.db"
            
    engine = create_engine(
        f"sqlite:///{db_path}",
        poolclass=NullPool,
        connect_args={'timeout': 15}
    )
    
    with engine.connect() as connection:
        df.to_sql(db_name, connection, index=False, if_exists='replace')
    
    return SQLDatabase(engine=engine)

def extract_data_with_csvloader(df):
    """Convert dataframe to langchain document format."""
    with tempfile.NamedTemporaryFile(delete=False, mode="w+", suffix=".csv") as tmp_file:
        df.to_csv(tmp_file.name, index=False)
        loader = CSVLoader(tmp_file.name)
        csv_data = loader.load_and_split()
    return csv_data

def preprocess_text_for_bm25(text):
    """Preprocess text for BM25 search."""
    pattern = r'[^a-zA-Z0-9\s]'
    formatted_text = re.sub(pattern, ' ', text).lower()
    return formatted_text.split(' ')

def create_bm25_object(data):
    """Create BM25 object from document corpus."""
    corpus = [doc_.page_content.replace("\n", " ") for doc_ in data]
    tokenized_corpus = [preprocess_text_for_bm25(doc) for doc in corpus]
    return BM25Okapi(tokenized_corpus)

def fetch_query_content_from_table(bm25, query, table_data, k=5):
    """Fetch relevant context for a query."""
    tokenized_query = preprocess_text_for_bm25(query)
    doc_scores = bm25.get_scores(tokenized_query)
    results = [(i, score) for i, score in enumerate(doc_scores)]
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    context = ""
    for res_ in sorted_results[:k]:
        if res_[1] > 0:
            context += f"---\n{table_data[res_[0]].page_content.replace('{', '[').replace('}', ']')}\n"
    return context

def initialize_components(api_key, df):
    """Initialize the necessary components for the SQL analysis."""
    try:
        config = SecretConfig(api_key=api_key)
        process_dataframe_to_store_in_db(df)
        
        # Initialize components
        db = create_sql_db('sample_df', df=df)
        llm = ChatOpenAI(
            model="gpt-4-turbo",
            openai_api_key=config.api_key,
            temperature=0,
            max_tokens=500
        )
        
        sql_agent = create_sql_agent(
            llm,
            db=db,
            agent_type="zero-shot-react-description",
            max_iterations=10,
            max_execution_time=500,
            top_k=10,
            verbose=True
        )
        
        table_data = extract_data_with_csvloader(df)
        bm25_obj = create_bm25_object(table_data)
        
        return {
            'db': db,
            'sql_agent': sql_agent,
            'table_data': table_data,
            'bm25_obj': bm25_obj
        }
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None

def cleanup_db_connections(db):
    """Safely close database connections."""
    try:
        if hasattr(db, '_engine'):
            db._engine.dispose()
    except Exception as e:
        print(f"Warning: Error while closing database connections: {e}")

# Streamlit app configuration
st.set_page_config(
    page_title="SQL Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

logo = Image.open("perrigo-logo.png")
st.image(logo, width=100)

# Custom CSS
st.markdown("""
    <style>
    .stAlert {
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .st-emotion-cache-16idsys p {
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'current_data_source' not in st.session_state:
    st.session_state.current_data_source = None

def reset_app_state():
    """Reset the app state when data source changes"""
    st.session_state.initialized = False
    if 'df' in st.session_state:
        del st.session_state.df
    if 'db' in st.session_state:
        cleanup_db_connections(st.session_state.db)
        del st.session_state.db
    if 'sql_agent' in st.session_state:
        del st.session_state.sql_agent
    if 'table_data' in st.session_state:
        del st.session_state.table_data
    if 'bm25_obj' in st.session_state:
        del st.session_state.bm25_obj

def main():
    st.markdown('<h2>GenAI Answer Bot</h2>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        st.subheader("1. API Key")
        api_key = st.text_input("Enter OpenAI API Key:", type="password")
        if not api_key:
            st.warning("⚠️ Please enter your OpenAI API key")
        
        # Data source selection
        st.subheader("2. Data Source Selection")
        data_source = st.radio(
            "Choose Data Source:",
            ["Use Default Data", "Upload Custom File"]
        )
        
        # Reset state if data source changes
        if st.session_state.current_data_source != data_source:
            st.session_state.current_data_source = data_source
            reset_app_state()
        
        df = None
        if data_source == "Use Default Data":
            df = load_default_data()
            if df is not None:
                st.success("Default data loaded successfully!")
        else:
            uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx'])
            if uploaded_file:
                try:
                    df = pd.read_excel(uploaded_file, sheet_name="Aggregated Data")
                    st.success("✅ Custom file loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error loading custom file: {str(e)}")
    
    # Main content area
    if not api_key:
        st.info("Please enter your OpenAI API key in the sidebar to get started.")
        return
    
    # Check data availability and initialization
    if not st.session_state.initialized:
        if data_source == "Use Default Data" and df is None:
            st.error("❌ Default data file not found. Please check if 'Data.csv' exists in the directory.")
            return
        elif data_source == "Upload Custom File":
            if df is None:
                st.info("📁 Please upload your Excel file in the sidebar.")
                return
        
        # Initialize components if we have valid data
        if df is not None:
            with st.spinner("Initializing components..."):
                try:
                    components = initialize_components(api_key, df)
                    if components:
                        st.session_state.df = df
                        st.session_state.db = components['db']
                        st.session_state.sql_agent = components['sql_agent']
                        st.session_state.table_data = components['table_data']
                        st.session_state.bm25_obj = components['bm25_obj']
                        st.session_state.initialized = True
                except Exception as e:
                    st.error(f"❌ Error during initialization: {str(e)}")
                    return
    
    # Only show the main interface if initialized
    if st.session_state.initialized:
        # Display sample data
        with st.expander("📊 View Sample Data"):
            st.dataframe(st.session_state.df.head(), use_container_width=True)
        
        # Query interface
        st.subheader("💬 Ask Questions About Your Data")
        
        # Sample queries
        st.markdown("#### Sample Queries")
        sample_queries = [
            "What is the cost per pallet for alloga uk in 2024?",
            "How has the cost per pallet evolved over the last 3 months?",
            "Find the top 5 NAMEs by total pallets shipped",
            "What is the trend in total shipping costs over time?",
            "Calculate the average Distance covered per order for each NAME",
            "Which regions have the highest shipping costs?",
            "Compare the cost efficiency between different product types"
        ]
        
        selected_query = st.selectbox(
            "Select a sample query or write your own below:",
            [""] + sample_queries,
            key="query_select"
        )
        
        query = st.text_area(
            "Enter your query:",
            value=selected_query,
            placeholder="Type your query here...",
            height=100,
            key="query_input"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            submit_button = st.button("🔍 Submit Query")
        
        if submit_button and query:
            with st.spinner("🔄 Analyzing your query..."):
                try:
                    context = fetch_query_content_from_table(
                        bm25=st.session_state.bm25_obj,
                        query=query,
                        table_data=st.session_state.table_data
                    )
                    
                    query_template = """
                    User Query: `{__query__}`

                    The following context contains sample rows from the data to help you construct the correct SQL query. 
                    Use this context to identify the appropriate columns and create an SQL query that accurately retrieves the required information.
                    `{__context__}`
                    """
                    
                    updated_query = query_template.format(__query__=query, __context__=context)
                    
                    result = st.session_state.sql_agent.invoke({"input": updated_query})
                    
                    st.session_state.chat_history.append({
                        "query": query,
                        "response": result['output']
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("📜 Analysis History")
            for idx, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(
                    f"Query {len(st.session_state.chat_history) - idx}: {chat['query'][:50]}...",
                    expanded=(idx == 0)
                ):
                    st.markdown("**🔍 Query:**")
                    st.write(chat['query'])
                    st.markdown("**💡 Response:**")
                    st.write(chat['response'])

if __name__ == "__main__":
    try:
        main()
    finally:
        if 'db' in st.session_state:
            cleanup_db_connections(st.session_state.db)
