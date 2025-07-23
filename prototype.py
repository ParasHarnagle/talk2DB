import asyncio
import collections
import json
import os
import re
from statistics import mean, variance
import time
from typing import Dict, List, Optional, Tuple
import boto3
import pymysql
import requests
import yaml
import io
from prophet import Prophet
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse,JSONResponse
from agent_squad.utils.tool import AgentTools, AgentTool
from agent_squad.retrievers import AmazonKnowledgeBasesRetriever, AmazonKnowledgeBasesRetrieverOptions
from pydantic import BaseModel
from agent_squad.orchestrator import AgentSquad, AgentSquadConfig
from agent_squad.agents import (
    AgentStreamResponse,
     BedrockLLMAgent, 
     BedrockLLMAgentOptions,
     AgentCallbacks,
     SupervisorAgent,
    SupervisorAgentOptions
)

from nixtla import NixtlaClient
from mao_usage_tracking import (
    find_matching_time,
    collect_token_usage_around_time
)
from botocore.config import Config
import json
import pandas as pd
import numpy as np
import traceback
from botocore.exceptions import ClientError
import pandas as pd
from agent_squad.storage import InMemoryChatStorage
from agent_squad.classifiers import BedrockClassifier, BedrockClassifierOptions
import tiktoken

from decimal import Decimal
from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError


GLOBAL_ENGINE = None
GLOBAL_DB_CREDS = None

#from dotenv import load_dotenv
#load_dotenv()
s3 = boto3.client('s3')

BUCKET = os.environ.get("bucket")
if BUCKET is None:
    raise ValueError("Environment variable 'bucket' is not set.")

def load_json_from_s3(bucket: str, key: str) -> dict:
    response = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(response['Body'].read())

def load_text_from_s3(bucket: str, key: str) -> str:
    response = s3.get_object(Bucket=bucket, Key=key)
    return response['Body'].read().decode('utf-8')



dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
CHAT_HISTORY_TABLE = dynamodb.Table("mao_chat_sessions")

def save_chat_history_dynamo(session_id: str, client_type: str, history: list):
    print("Saving Chat session ...")
    CHAT_HISTORY_TABLE.put_item(
        Item={
            "session_id": session_id,
            "client_type": client_type,
            "chat_history": history  # store only last 5
        }
    )

def load_chat_history_dynamo(session_id: str, client_type: str) -> list:
    try:
        print("Getting table ....")
        response = CHAT_HISTORY_TABLE.get_item(
            Key={"session_id": session_id}
        )
        item = response.get("Item", {})
        
        # Optional: verify client_type if stored inside item
        if item.get("client_type") == client_type:
            print("Fetched history")
            return item.get("chat_history", [])
        else:
            print("No one is sitting in the DB")
            return []
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return []

def format_chat_history_dynamo(session_id: str, client_type: str) -> str:
    print("Loading history ....")
    history = load_chat_history_dynamo(session_id, client_type)
    latest_turns = history[-3:] if history else []
    return "\n".join(
        f"User: {t['user_input']}\nAgent: {t['agent_response']}" for t in latest_turns
    ) if history else ""

#NS_Examples = load_text_from_s3(BUCKET, "NS_Examples.txt")

yaml_text = load_text_from_s3(BUCKET, "agent_config.yaml")
config = yaml.safe_load(io.StringIO(yaml_text))

orchestrator = None
SESSION_METADATA: Dict[str, dict] = {}
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def make_bedrock_client(region="us-east-1"):
    return boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=Config(
            connect_timeout=90,
            read_timeout=300,
            retries={"max_attempts": 3, "mode": "adaptive"},
            tcp_keepalive=True,
        )
    )


model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
#model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
classifier = BedrockClassifier(BedrockClassifierOptions(model_id=model_id,
                                                        region="us-east-1"))

class Body(BaseModel):
    input: str
    client_id: str  
    client_type: str
    session_id: str
    #chat_transaction_id: str

class LLMAgentCallbacks(AgentCallbacks):
    def __init__(self):
        self.token_count = 0
        self.tokens = []
        self.full_response = ""
        self.agent_token_usage = {}  

    async def on_llm_new_token(self, token: str, agent_tracking_info: Optional[dict] = None) -> None:
        self.token_count += 1
        self.tokens.append(token)
        self.full_response += token
        if agent_tracking_info:
            agent_name = agent_tracking_info.get("agent_name", "unknown")
            self.agent_token_usage.setdefault(agent_name, 0)
            self.agent_token_usage[agent_name] += 1
        print(token, end='', flush=True)

    async def on_llm_response(self, response: str, agent_tracking_info: Optional[dict] = None) -> None:
        tokens = response.split()
        self.token_count += len(tokens)
        self.tokens.extend(tokens)
        self.full_response = response
        if agent_tracking_info:
            agent_name = agent_tracking_info.get("agent_name", "unknown")
            self.agent_token_usage.setdefault(agent_name, 0)
            self.agent_token_usage[agent_name] += len(tokens)
        print(f"\n\n[Non-Streaming LLM Response Received]:\n{response}")



retriever=AmazonKnowledgeBasesRetriever(AmazonKnowledgeBasesRetrieverOptions(
            knowledge_base_id="G6NAFKNJ1Q",
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": 3,
                    "overrideSearchType": "SEMANTIC",
                },
                
            }
            ))

memory_storage = InMemoryChatStorage()
orchestrator = AgentSquad(options=AgentSquadConfig(
        LOG_AGENT_CHAT=True,
        LOG_CLASSIFIER_CHAT=True,
        LOG_CLASSIFIER_RAW_OUTPUT=True,
        LOG_CLASSIFIER_OUTPUT=True,
        LOG_EXECUTION_TIMES=True,
        MAX_RETRIES=3,
        USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED=True,
        NO_SELECTED_AGENT_MESSAGE="Please rephrase",
        MAX_MESSAGE_PAIRS_PER_AGENT=10
        ),
        classifier = classifier,
        storage=memory_storage
    )


fin_agent_callbacks = LLMAgentCallbacks()
def setup_core_agent():
    
    global GLOBAL_DB_CREDS

    FIN_AGENT_PROMPT = load_text_from_s3(BUCKET, "FIN_AGENT_PROMPT.txt")
    NS_TABLE_INFO = load_json_from_s3(BUCKET, "NS_TABLE_INFO.json")
    fd = load_text_from_s3(BUCKET, "forecast_agent_prompt.txt")
    fsp = load_text_from_s3(BUCKET, "search_agent_prompt.txt")

    fin_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
            name="finance_translation",
            description="Translates generic high level user requests into SQL query and returns structured results",
            model_id=model_id,
            streaming=True,
            callbacks=fin_agent_callbacks,
            inference_config={
                "maxTokens": 5000,
                "temperature": 0.1  
            },
            tool_config={
                "tool": data_retrieval_tool,
                "toolMaxRecursions": 20
            },
            retriever=retriever,  # keep using few-shot example retriever
            save_chat=True,
            custom_system_prompt={
                "template": FIN_AGENT_PROMPT,
                "variables": {
                    #"TABLEINFO": stringify_table_info(NS_TABLE_INFO),
                    "HISTORY":""
                }
            }, 
            client=make_bedrock_client("us-east-1")
        ))
    

    
    safe_add_agent(fin_agent, orchestrator)

    forecast_tool = AgentTools(tools=[
    AgentTool(
        name="forecast_generic_tool",
        description=(
            "Tool for forecasting numeric trends from time-series data.\n\n"
            "Accepts input as:\n"
            "```\n"
            "{\n"
            '  "metric_name": "revenue",\n'
            '  "forecast_horizon": 6,\n'
            '  "data": [\n'
            '    {"ds": "YYYY-MM-DD", "y": float},\n'
            '    ...\n'
            '  ]\n'
            "}\n"
            "```\n"
            "The tool auto-handles log transformation, detects seasonality, and uses Nixtla's TimeGPT-1 with exogenous regressors like month, quarter, and campaign.\n"
            "Always return a structured forecast JSON along with visualization metadata."
        ),
        func=forecast_generic
    )
])
    combined_tools = AgentTools(tools=[
    *data_retrieval_tool.tools,
    *forecast_tool.tools
])

    fd_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
                name="finance_projection",
                description="Forecasts future trends and projections into financial scenarios using historical data and user queries.",
                model_id=model_id,
                callbacks=financialprojectionagent_callbacks,
                streaming=True,
                save_chat=True, 
                tool_config={
                    "tool": combined_tools, 
                    "toolMaxRecursions": 20
                },
                custom_system_prompt={
                    "template": fd,
                    "variables": {
                        "TABLEINFO": stringify_table_info(NS_TABLE_INFO),
                        "HISTORY":""}
                },
               inference_config={
                "maxTokens": 5000,
                "temperature": 0.1  
             },
              client=make_bedrock_client("us-east-1")
                ))

    safe_add_agent(fd_agent, orchestrator)

    serper_tool = AgentTools(tools=[create_serper_search_tool()])

    serper_combined_tools = AgentTools(tools=[
                *data_retrieval_tool.tools,
                *serper_tool.tools
            ])
    
    serper_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
                name="finance_search",
                #description="Performs internet searches to gather competitor insights and external context based on the retrieved data.",
                description="Conducts real-time internet searches to augment internal data with competitor analysis and external market trends.",
                model_id=model_id,
                callbacks=financialcomparisonagent_callbacks,
                streaming=True,
                #save_chat=True, 
                tool_config={
                    "tool":  serper_combined_tools, 
                    "toolMaxRecursions": 15
                },
                custom_system_prompt={
                    "template": fsp,
                    "variables": {
                         "HISTORY":"",
                         "Company_Name":  GLOBAL_DB_CREDS["Company_Name"],
                         "Company_Domain": GLOBAL_DB_CREDS["Company_Domain"],
                    }
                },
                inference_config={
                     "maxTokens": 5000,
                    "temperature": 0.1  
                 },
                client=make_bedrock_client("us-east-1")
                ))

                
    
    safe_add_agent(serper_agent, orchestrator)


GLOBAL_DB_CREDS = {...}  # Your credentials
GLOBAL_MYSQL_CONNECTION = None
SCHEMA_CACHE_TTL = 3600 

def get_serper_key(app,environment,serper_config):

    secret_name = f"{app}-{environment}-{serper_config}"
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    serper_key = get_secret_value_response['SecretString']
    serper_key_dict = json.loads(serper_key)

    # Return only the specific key value
    return serper_key_dict


def create_serper_search_tool():
    # Fetch environment variables
    app = os.environ.get("app")
    environment = os.environ.get("environment")
    
    # Fetch the Serper API key
    serper_key_dict = get_serper_key(app, environment, "google-serper-key")
    serper_api_key = serper_key_dict["google-serper-key"]

    # Allowed sites
    allowed_sites = [
        "statista.com",
        "mckinsey.com",
        "forbes.com",
        "hbr.org",
        "pwc.com",
        "investopedia.com",
        "aicpa-cima.com",
        "ibisworld.com",
        "corporatefinanceinstitute.com",
        "bls.gov",
        "benchmarkintl.com",
        "userpilot.com",
        "stripe.com",
        "paddle.com",
        "sec.gov"
    ]

    # Internal search function
    def search_serper(query: str) -> str:
        endpoint = "https://google.serper.dev/search"
        headers = {
            "Accept": "application/json",
            "x-api-key": serper_api_key
        }
        params = {
            "q": query,
            "num": 10
        }
        response = requests.get(endpoint, headers=headers, params=params)
        
        # Debug info
        print("Final URL:", response.request.url)
        print("Request Headers:", response.request.headers)
        print("Status Code:", response.status_code)
        print("Response Text:", response.text)
        
        response.raise_for_status()
        data = response.json()
        print(">>>>>>>>>>>>>>")
        
        # Filter results to allowed sites
        filtered_results = [
            r["link"] for r in data.get("results", [])
            if any(site in r["link"] for site in allowed_sites)
        ]
        if not filtered_results:
            print("No links found in allowed sites, using all result links.")
            filtered_results = [r["link"] for r in data.get("results", [])]
        print("filetered resutlts", filtered_results)
        print(">>>>>>>>>>>>>>>>>>>")
        return "\n".join(filtered_results) or "No relevant results."

    # Return as AgentTool
    return AgentTool(
        name="internet_search",
        description="Searches the internet for insights using the Serper API",
        properties={
            "query": {
                "type": "string",
                "description": "The search query."
            }
        },
        required=["query"],
        func=search_serper
    )

ALLOWED_TASK_AGENTS = {'sentimentanalysisagent'}
## --------------------------
## Forecast Function
VARIANCE_THRESHOLD = 0.25

def generic_forecast(input_payload, api_key=None):
    # Step 1: Load time series
    df = pd.DataFrame(input_payload["data"])
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"])
    df["unique_id"] = input_payload.get("metric_name", "main")

    # Step 2: Optional log transform
    df["y_log"] = np.log1p(df["y"])

    # Step 3: Historical exogenous features
    df["month"] = df["ds"].dt.month
    df["quarter"] = df["ds"].dt.quarter
    df["campaign"] = (df["y_log"] > 3 * df["y_log"].rolling(3, min_periods=1).mean()).astype(int)
    hist_exog_list = ["month", "quarter", "campaign"]

    # Step 4: Future exogenous features
    future_dates = pd.date_range(start=df["ds"].max() + pd.DateOffset(months=1),
                                 periods=input_payload["forecast_horizon"], freq="MS")
    X_df = pd.DataFrame({"ds": future_dates, "unique_id": df["unique_id"].iloc[0]})
    X_df["month"] = X_df["ds"].dt.month
    X_df["quarter"] = X_df["ds"].dt.quarter
    X_df["campaign"] = 0

    # Step 5: Forecast via Nixtla
    nixtla_client = NixtlaClient(api_key=api_key)
    forecast_df = nixtla_client.forecast(
        df=df[["ds", "y_log", "unique_id"] + hist_exog_list].rename(columns={"y_log": "y"}),
        X_df=X_df,
        h=input_payload["forecast_horizon"],
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        model="timegpt-1-long-horizon",
        hist_exog_list=hist_exog_list
    )

    # Step 6: Inverse transform + return
    forecast_df["forecast"] = np.expm1(forecast_df["TimeGPT"])
    return forecast_df[["ds", "forecast"]]

def forecast_generic(input_data: str) -> str:
    start_time = time.time()
    try:
        print("[Tool] Starting Nixtla-based forecast...")
        data = json.loads(input_data)

        if "date" not in data or "value" not in data:
            raise ValueError("Missing required fields: 'date' and/or 'value'.")

        # Step 1: Prepare DataFrame
        df = pd.DataFrame({
            "ds": pd.to_datetime(data["date"], errors='coerce'),
            "y": pd.to_numeric(data["value"], errors="coerce")
        }).dropna().sort_values("ds")

        if df.shape[0] > 30:
            df = df.iloc[-30:]
        print(f"[Tool] Cleaned DataFrame shape: {df.shape}")

        if len(df) < 12:
            raise ValueError("Insufficient clean data for time-series forecasting (need ≥ 12 rows).")

        forecast_horizon = data.get("forecast_days", 30)

        # Step 2: Prepare payload for Nixtla TimeGPT
        input_payload = {
            "metric_name": "series_1",
            "forecast_horizon": forecast_horizon,
            "data": df.to_dict(orient="records")
        }

        # Step 3: Forecast
        forecast_df = generic_forecast(input_payload, api_key="nixak-Y36YU5kwISL4dBKuob2tsMQoXTs4MR9te6VXtHNTDLzU4S3TK2BvTOYv0SKgXgh848iqC3tHrkmwdE3y")

        # Step 4: Format output
        output = {
            "success": True,
            "model_used": "Nixtla TimeGPT-1 Long Horizon (log + exogenous)",
            "forecast_summary": f"Avg next {forecast_horizon} steps: {round(forecast_df['forecast'].mean(), 2)}",
            "forecast_data": [
                {
                    "ds": row.ds.isoformat(),
                    "yhat": round(row.forecast, 2)
                } for row in forecast_df.itertuples()
            ],
            "visualization": {
                "type": "line",
                "xAxis": "ds",
                "yAxis": ["yhat"]
            },
            "execution_time_ms": int((time.time() - start_time) * 1000)
        }

        return json.dumps(output, indent=2)

    except Exception as e:
        print(f"[Tool Error] {str(e)}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "schema": {
                "required_fields": ["date[]", "value[]"],
                "optional_fields": ["forecast_days"],
                "formats": {
                    "date[]": "ISO format like YYYY-MM-DD",
                    "value[]": "float",
                    "forecast_days": "int (e.g., 30, 90)"
                }
            },
            "execution_time_ms": int((time.time() - start_time) * 1000)
        }, indent=2)
    
 
def forecast_with_prophet(input_data: str) -> str:
    print("\U0001F4CA Starting Prophet forecast...")
    try:
        data = json.loads(input_data)

        # Validate input
        if "date" not in data or "revenue" not in data:
            return json.dumps({"error": "Missing 'date' or 'revenue' fields in input."})

        # Prepare DataFrame for Prophet
        df = pd.DataFrame({
            "ds": pd.to_datetime(data["date"]),
            "y": data["revenue"]
        })

        # Dynamically determine forecast window
        forecast_days = data.get("forecast_days")
        if not forecast_days:
            history_length = len(data["date"])
            forecast_days = max(7, int(history_length / 3))  # fallback if user didn't specify

        # Fit model
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=True,
            n_changepoints=min(20, max(1, len(df) - 1))  # safety for short datasets
        )
        model.fit(df)

        # Forecast
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)

        result_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_days)

        # Convert result to JSON-safe format
        result_dicts = []
        for row in result_df.itertuples(index=False):
            result_dicts.append({
                "ds": row.ds.isoformat(),
                "yhat": round(row.yhat, 2),
                "yhat_lower": round(row.yhat_lower, 2),
                "yhat_upper": round(row.yhat_upper, 2)
            })

        output = {
            "forecast_summary": f"Predicted average for next {forecast_days} days: {round(result_df['yhat'].mean(), 2)}",
            "forecast_data": result_dicts,
            "visualization": {
                "chart_type": "Line Chart",
                "xAxis": [row["ds"] for row in result_dicts],
                "yAxis": [{
                    "name": "Forecasted Value",
                    "data": [row["yhat"] for row in result_dicts]
                }]
            }
        }

        print("\u2705 Forecast complete. Returning output.")
        print("output",json.dumps(output))
        return json.dumps(output)

    except Exception as e:
        print(f"\u274C Forecast error: {e}")
        return json.dumps({"error": f"Error in Prophet forecast: {str(e)}"})
    
def safe_add_agent(agent, orchestrator):
    """
    Adds an agent to the orchestrator only if it doesn't already exist.
    Prevents ValueError on repeated invocations in Lambda or server mode.
    """
    try:
        if agent.id not in orchestrator.agents:
            orchestrator.add_agent(agent)
        else:
            print(f" Agent '{agent.id}' already exists. Skipping re-addition.")
    except ValueError as e:
        print(f"⚠️ Could not add agent '{agent.id}': {e}")

def stringify_table_info(table_info):
    if isinstance(table_info, list) and all(isinstance(d, dict) for d in table_info):
        return "\n".join(json.dumps(row, indent=2) for row in table_info)
    return str(table_info)

financialcomparisonagent_callbacks = LLMAgentCallbacks()
financialprojectionagent_callbacks = LLMAgentCallbacks()
llm_callbacks = LLMAgentCallbacks()

def add_task_agents_to_orchestrator(config,orchestrator):
    global GLOBAL_DB_CREDS
    serper_tool = AgentTools(tools=[create_serper_search_tool()])

    task_agents = {}
    for key, agent_data in config.get("task_agents", {}).items():


        if key in ALLOWED_TASK_AGENTS:
                agent = BedrockLLMAgent(BedrockLLMAgentOptions(
                name=agent_data["name"],
                description=agent_data["description"],
                model_id=model_id,
                callbacks=llm_callbacks,
                streaming=True,
                #save_chat=True, 
                #tool_config={
                #    "tool":  AgentTools([serper_search_tool]), 
                #    "toolMaxRecursions": 3
                #},
                custom_system_prompt={
                    "template": agent_data["prompt_template"],
                    "variables": {}
                },
                inference_config={
                    'maxTokens': 3500},
                client=make_bedrock_client("us-east-1")
                ))

                #safe_add_agent(agent, orchestrator)
                task_agents[key] = agent


    return task_agents

## DB TOOL 

def init_sqlalchemy_pool():
    global GLOBAL_ENGINE, GLOBAL_DB_CREDS

    if GLOBAL_ENGINE:
        return GLOBAL_ENGINE

    if not GLOBAL_DB_CREDS:
        raise Exception("Database credentials not loaded")

    connection_string = (
        f"mysql+pymysql://{GLOBAL_DB_CREDS['DB_USER']}:{GLOBAL_DB_CREDS['DB_PASSWORD']}"
        f"@{GLOBAL_DB_CREDS['DB_HOST']}:{GLOBAL_DB_CREDS['DB_PORT']}/{GLOBAL_DB_CREDS['DB_NAME']}"
    )

    print("Creating SQLAlchemy engine with connection pooling...")
    GLOBAL_ENGINE = create_engine(
        connection_string,
        poolclass=QueuePool,
        pool_size=10,          # Number of connections in pool
        max_overflow=20,       # Additional connections beyond pool_size
        pool_timeout=30,       # Wait time for a connection before throwing
        pool_recycle=1800,     # Recycle connections every 30 mins
        echo=False             # Set to True to debug SQL
    )
    return GLOBAL_ENGINE


def connect_to_mysql_():
    """Establish and return a connection to MySQL database"""
    global GLOBAL_DB_CREDS, GLOBAL_MYSQL_CONNECTION

    # Check if connection exists and is still alive
    if GLOBAL_MYSQL_CONNECTION:
        try:
            with GLOBAL_MYSQL_CONNECTION.cursor() as cursor:
                cursor.execute("SELECT 1")  # Lightweight test query
                return GLOBAL_MYSQL_CONNECTION
        except:
            # Connection is stale or broken, close if possible
            try:
                if GLOBAL_MYSQL_CONNECTION.open:
                    GLOBAL_MYSQL_CONNECTION.close()
            except:
                pass
    
    # Create a new connection
    try:    
        print("Creating new MySQL connection...", GLOBAL_DB_CREDS)
        GLOBAL_MYSQL_CONNECTION = pymysql.connect(
            host=GLOBAL_DB_CREDS['DB_HOST'],
            user=GLOBAL_DB_CREDS['DB_USER'],
            password=GLOBAL_DB_CREDS['DB_PASSWORD'],
            db=GLOBAL_DB_CREDS['DB_NAME'],
            port=int(GLOBAL_DB_CREDS['DB_PORT']),
            autocommit=True,
            connect_timeout=5,  # Short connection timeout
            read_timeout=30,    # Query timeout
            cursorclass=pymysql.cursors.DictCursor  # Return results as dictionaries
        )
        print("[✅ New DB connection established]")
        return GLOBAL_MYSQL_CONNECTION
    except Exception as connect_err:
        print(f"[❌ Failed to connect to MySQL]: {connect_err}")
        raise

def get_full_schema():
    try:
        engine = init_sqlalchemy_pool()
        with engine.connect() as conn:
            table_query = text("""
                SELECT TABLE_NAME FROM information_schema.TABLES
                WHERE TABLE_SCHEMA = :schema
            """)
            column_query = text("""
                SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, COLUMN_KEY,
                       IS_NULLABLE, COLUMN_COMMENT
                FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = :schema
                ORDER BY TABLE_NAME, ORDINAL_POSITION
            """)

            tables = conn.execute(table_query, {"schema": GLOBAL_DB_CREDS["DB_NAME"]}).fetchall()
            columns = conn.execute(column_query, {"schema": GLOBAL_DB_CREDS["DB_NAME"]}).fetchall()

            schema = {}
            for row in tables:
                schema[row[0]] = []

            for col in columns:
                schema[col[0]].append({
                    "name": col[1],
                    "type": col[2],
                    "key": col[3],
                    "nullable": col[4],
                    "comment": col[5],
                })

            return schema
    except SQLAlchemyError as e:
        print(f"❌ Error fetching schema: {str(e)}")
        return {"error": str(e)}

def run_query_tool_with_fallback(**kwargs):
    sql = kwargs.get("sql_query")
    fetch_schema_only = kwargs.get("fetch_schema_only", False)
    start_time = time.time()

    if fetch_schema_only:
        schema = get_full_schema()
        return json.dumps({
            "schema": schema,
            "execution_time_ms": int((time.time() - start_time) * 1000)
        }, indent=2)

    try:
        engine = init_sqlalchemy_pool()
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = [dict(row._mapping) for row in result]
            return json.dumps({
                "success": True,
                "rows": rows,
                "row_count": len(rows),
                "execution_time_ms": int((time.time() - start_time) * 1000)
            }, indent=2)
    except Exception as e:
        print(f"❌ SQL Error: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "schema": get_full_schema(),
            "execution_time_ms": int((time.time() - start_time) * 1000),
            "suggestion": "SQL execution failed. Schema returned to regenerate query."
        }, indent=2)
    
def get_db_credentials(app, environment, client_type, client_id):
    #riveron-ai-dev-ns-honeycombs-hc
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    secret_name = f"{app}-{environment}-{client_type}-{client_id}"
    print(f"Fetching secret for: app={app}, environment={environment}, client_type={client_type}, client_id={client_id}")
    print(f"Built SecretId: {secret_name}")
    print(f"Fetching DB credentials for {secret_name}")
    region_name = "us-east-1"
    print(secret_name)

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except client.exceptions.ResourceNotFoundException:
        print(f"❌ Secret not found: {secret_name}")
        raise Exception(f"Secret not found: {secret_name}")
    except Exception as e:
        print(f"❌ Other error: {str(e)}")
        raise e

    credentials = get_secret_value_response['SecretString']
    credentials = json.loads(credentials)
    return credentials

data_retrieval_tool = AgentTools(tools=[
    AgentTool(
        name="run_query",
        description="Primary database tool: executes SQL queries or retrieves schema information in a single call",
        func=run_query_tool_with_fallback,
        properties={
            "sql_query": {
                "type": "string",
                "description": "The SQL query to be executed."
            },
            "fetch_schema_only": {
                "type": "boolean",
                "description": "If true, only returns schema without executing query. Default: false."
            }
        },
        required=["sql_query"]
    )
])

def extract_sql_query(text: str) -> str:
    
    pattern = r"```sql\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

sa_callbacks = LLMAgentCallbacks()

def build_supervisor(user_input, task_agents,NS_TABLE_INFO):
    BUCKET = os.environ.get("bucket")
    if BUCKET is None:
        raise ValueError("Environment variable 'bucket' is not set.")
    SA_prompt = load_text_from_s3(BUCKET, "SA_prompt.txt")
    if hasattr(SupervisorAgent, "_configure_prompt"):
        SupervisorAgent._configure_prompt = lambda self: None

    formatted_prompt = SA_prompt.format(user_input=user_input)
    supervisor_lead_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
        name="SupervisorLeadAgent",
        description="Coordinates all financial task agents and generates final user response using user input and chain output response",
        model_id=model_id,
        custom_system_prompt={
            "template": formatted_prompt,
            "variables": {"ns_table_info": NS_TABLE_INFO}
        },
        inference_config={'maxTokens': 7000},
        callbacks=sa_callbacks,
        streaming=True
    ))

    supervisor_agent = SupervisorAgent(SupervisorAgentOptions(
        lead_agent=supervisor_lead_agent,
        name="SupervisorLeadAgent",
        description="Coordinates all financial task agents and generates final user response using user input and chain output response",
        team=list(task_agents.values()),
     #   extra_tools=[serper_tool],
        storage=memory_storage,
        trace=True
    ))

    supervisor_agent.prompt_template = formatted_prompt

    return supervisor_agent



def count_prompt_tokens(prompt: str) -> int:
    """
    Count tokens in a given prompt string using the specified tokenizer encoding.

    Args:
        prompt (str): The prompt text to tokenize.
        model_encoding (str): The tokenizer encoding to use (default: "cl100k_base").

    Returns:
        int: The number of tokens in the prompt.
    """
    model_encoding: str = "cl100k_base"
    encoding = tiktoken.get_encoding(model_encoding)
    tokens = encoding.encode(prompt)
    return len(tokens)


MAX_RETRIES = 3
RETRY_DELAY = 2
HEARTBEAT_INTERVAL = 10

def extract_next_questions(response_text: str) -> Tuple[str, List[str]]:
    pattern = r"### Next probable questions you might ask:\s*(.*?)(?=\n###|\Z)"
    match = re.search(pattern, response_text, re.DOTALL)

    if not match:
        return response_text.strip(), []

    questions_block = match.group(1).strip()
    
    # Clean and split into individual questions
    lines = questions_block.split("\n")
    cleaned = [
        re.sub(r"^[\-\*\d\.\)]*\s*", "", line).strip()
        for line in lines if line.strip()
    ]

    # Main content is everything before the matched pattern
    main_content = response_text[:match.start()].strip()
    print("MAIN CONTENT =======>>>>>",main_content)
    return main_content, cleaned[:3]
    
def try_fix_json_snippet(snippet: str) -> str:
    text = snippet.strip()
    text = re.sub(r",\s*([\]}])", r"\1", text)
    
    bracket_stack = []
    for ch in text:
        if ch in "{[":
            bracket_stack.append(ch)
        elif ch in "}]":
            if bracket_stack:
                bracket_stack.pop()

    for ch in reversed(bracket_stack):
        text += "}" if ch == "{" else "]"

    return text


async def response_generator(query, user_id, client_type, session_id):
    print("Available agents in orchestrator:", list(orchestrator.agents.keys()))

    global GLOBAL_DB_CREDS
    start_time = time.time()
    agent_output_chunks = []
    final_response = ""
    chat_history = []
    chain_response = ""

    environment = os.environ.get("environment")
    app = os.environ.get("app")
    BUCKET = os.environ.get("bucket")
    if BUCKET is None:
        raise ValueError("Environment variable 'bucket' is not set.")
    GLOBAL_DB_CREDS = get_db_credentials(app, environment, client_type, user_id)
    FIN_AGENT_PROMPT = load_text_from_s3(BUCKET, "FIN_AGENT_PROMPT.txt")
    NS_TABLE_INFO = load_json_from_s3(BUCKET, "NS_TABLE_INFO.json")
    keep_alive_running = True
    setup_core_agent()
    task_agents = add_task_agents_to_orchestrator(config=config,orchestrator=orchestrator)
    
    formatted_history = format_chat_history_dynamo(session_id,client_type)
    print("=======")
    print(f"Histroy {formatted_history}")
    print("========")#

    agent_one = orchestrator.agents.get("financetranslation") 
    agent_two = orchestrator.agents.get("financeprojection")
    agent_three = orchestrator.agents.get("financesearch")
    print("Agent",agent_one)
    if agent_one:
        agent_one.set_system_prompt(
            FIN_AGENT_PROMPT, 
            {
                #"TABLEINFO": stringify_table_info(NS_TABLE_INFO),
                "HISTORY": formatted_history
            }
        )
    print("Agent_two",agent_two)
    fd = load_text_from_s3(BUCKET, "forecast_agent_prompt.txt")
    if agent_two:
        agent_two.set_system_prompt(
            fd,  
            {
                #"TABLEINFO": stringify_table_info(NS_TABLE_INFO),
                "HISTORY": formatted_history
            }
        )
    print("Agent_three",agent_three)
    fsp = load_text_from_s3(BUCKET, "search_agent_prompt.txt")
    if agent_three:
        agent_three.set_system_prompt(
            fsp,  
            {
                #"TABLEINFO": stringify_table_info(NS_TABLE_INFO),
                "HISTORY": formatted_history,
                "Company_Name":  GLOBAL_DB_CREDS["Company_Name"],
                "Company_Domain": GLOBAL_DB_CREDS["Company_Domain"]
            }
        )
    
    response = await orchestrator.route_request(query, user_id, session_id, {}, True)
    
    if response.streaming:
        attempt = 0
        last_hb = time.time()
        accumulated_output = ""
        while True:
            try:
                async for chunk in response.output:
                    now = time.time()
                    if now - last_hb > HEARTBEAT_INTERVAL:
                        yield f"event: heartbeat\ndata: {json.dumps({'time': now})}\n\n"
                        last_hb = now
                    if isinstance(chunk, AgentStreamResponse):
                        chain_response += chunk.text
                        agent_output_chunks.append(chunk.text)
                        chat_history.append({"role": "agent", "text": chunk.text})
                        accumulated_output += chunk.text
                        safe_output = chunk.text
                        try:
                            if any(accumulated_output.strip().endswith(end) for end in ["}", "]", '"', 'true', 'false', 'null']): _ = json.loads(accumulated_output)
                        except json.JSONDecodeError as e:
                            print(f"[Warning] Malformed JSON detected: {str(e)}")
                            
                            safe_output = try_fix_json_snippet(chunk.text)
                        #yield chunk.text
                        yield f"event: delta\ndata: {json.dumps({'output': safe_output})}\n\n"
                break
            except Exception as e:
                attempt += 1
                print(f"[Stream Retry Error] Attempt {attempt}: {str(e)}")
                yield f"event: notice\ndata: {json.dumps({'notice': f'Network glitch: {str(e)} — retrying streaming...'})}\n\n"

                if attempt >= MAX_RETRIES:
                    yield f"event: error\ndata: {json.dumps({'error': f'Failed after {MAX_RETRIES} retries. Last error: {str(e)}'})}\n\n"
                    fallback = await orchestrator.route_request(query, user_id, session_id, {}, False)
                    final_text = (
                        fallback.output
                        if not getattr(fallback, "streaming", False)
                        else "".join(c.text for c in fallback.output)
                    )
                    yield f"event: delta\ndata: {json.dumps({'output': final_text})}\n\n"
                    return
                await asyncio.sleep(RETRY_DELAY)
   
    print("Updating Chat History ...")
    existing_history = load_chat_history_dynamo(session_id, client_type)
    updated_history = existing_history + [{"user_input": query, "agent_response": chain_response}]
    save_chat_history_dynamo(session_id, client_type, updated_history)
    print("Chat History Updated ....")
    #existing_history = SESSION_METADATA.get(session_id, {}).get("chat_history", [])
    #current_turn = {
    #"user_input": query,
    #"agent_response": chain_response
    #}
#
    ## Append new turn to history
    #updated_history = existing_history + [current_turn]
#
    ## Store only relevant metadata
    #SESSION_METADATA[session_id] = {
    #    "input": query,
    #    "chat_history": updated_history[-5:]
    #}
    print("======")
    print("======")
    #print("chain_response:", chain_response)
    #print("SQL Query:", sql_query)
    user_input_to_task = f"""
    User Request: {query}

    --- SQL Executed ---
    {chain_response}
    Use only the provided data and insights to complete the financial task.
    """
    
    if "SupervisorLeadAgent" not in orchestrator.agents:
        supervisor_agent = build_supervisor(
            user_input=user_input_to_task,
            task_agents=task_agents,
            NS_TABLE_INFO=NS_TABLE_INFO
            )
        safe_add_agent(supervisor_agent, orchestrator)
    
    await asyncio.sleep(1)

    response_sa = await supervisor_agent.process_request(
        input_text=user_input_to_task,
        user_id=user_id,
        session_id=session_id,
        chat_history=[],
        additional_params={"ns_table_info": NS_TABLE_INFO},
    )

    await asyncio.sleep(3)

    
    if isinstance(response_sa, collections.abc.AsyncIterable):
        print("\n** STREAMING SA RESPONSE **")
        attempt = 0
        accumulated_output = ""
        last_hb = time.time() 
        while True:
            try:
                async for chunk in response_sa:
                    now = time.time()
                    if now - last_hb > HEARTBEAT_INTERVAL:
                        yield f"event: heartbeat\ndata: {json.dumps({'time': now})}\n\n"
                        last_hb = now
                    final_response += chunk.text
                    chat_history.append({"role": "supervisor", "text": chunk.text})
                    accumulated_output += chunk.text
                    safe_output = chunk.text
                    try:
                        if any(accumulated_output.strip().endswith(end) for end in ["}", "]", '"', 'true', 'false', 'null']):
                            _ = json.loads(accumulated_output)
                    except json.JSONDecodeError as e:
                        print(f"[Warning] Malformed JSON detected (Supervisor): {str(e)}")
                        safe_output = try_fix_json_snippet(chunk.text)
                    yield f"event: delta\ndata: {json.dumps({'output': safe_output})}\n\n"
                    
                break
            except Exception as e:
                attempt += 1
                print(f"[Stream Retry Error] Attempt {attempt}: {str(e)}")
                yield f"event: notice\ndata: {json.dumps({'notice': f'Supervisor stream interrupted: {str(e)} — retrying streaming...'})}\n\n"
                if attempt >= MAX_RETRIES:
                    yield f"event: error\ndata: {json.dumps({'error': f'Failed after {MAX_RETRIES} retries. Last error: {str(e)}'})}\n\n"
                    # non-streaming fallback
                    fb_sa = await supervisor_agent.process_request(
                        input_text=user_input_to_task,
                        user_id="USER_ID",
                        session_id=session_id,
                        chat_history=[],
                        additional_params={"ns_table_info": NS_TABLE_INFO},
                    )
                    text = fb_sa.text if hasattr(fb_sa, "text") else "".join(c.text for c in fb_sa)
                    yield f"event: delta\ndata: {json.dumps({'output': text})}\n\n"
                    return
                await asyncio.sleep(RETRY_DELAY)

    print("Updating Chat History ...")
    existing_history = load_chat_history_dynamo(session_id, client_type)
    updated_history = existing_history + [{"user_input": query, "agent_response": final_response}]
    save_chat_history_dynamo(session_id, client_type, updated_history)
    print("Chat History Updated ....")

    response_time = round(time.time() - start_time, 2)
    print("--------------------")
    print(f"Final Response:{final_response}")
    print("--------------------")
    sql_query = extract_sql_query(user_input_to_task)
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    year = now.strftime('%Y')
    month = now.strftime('%m')
    day = now.strftime('%d')

    PREFIX = f'AWSLogs/844994419835/BedrockModelInvocationLogs/us-east-1/{year}/{month}/{day}/'

    await asyncio.sleep(RETRY_DELAY)

    match_result = find_matching_time('cfo-ai-bedrock-logging', PREFIX, query)


    if match_result:
        match_time = match_result["match_time"]
        matched = collect_token_usage_around_time('cfo-ai-bedrock-logging', PREFIX, query, match_time)
        summary = matched["summary"]

        input_token = summary["total_input_tokens"]
        completion_tokens = summary["total_output_tokens"]
        total_tokens = summary["total_tokens"]
        pricing = round(summary["pricing_usd"], 2)
    else:
        # Manual fallback logic
        print(json.dumps({"message": "No matching log found — using manual token calculation."}, indent=2))

        input_token = count_prompt_tokens(query)
        total_tokens = (
            sa_callbacks.token_count +
            llm_callbacks.token_count +
            financialprojectionagent_callbacks.token_count +
            financialcomparisonagent_callbacks.token_count
        )
        completion_tokens = total_tokens - input_token
        pricing = round((input_token * 0.003 + completion_tokens * 0.015) / 1000, 5)

    
    main_content,next_quest_list = extract_next_questions(final_response.strip())

    yield f"event: Test_event\ndata: {json.dumps(main_content)}\n\n"

    response_serialized = {
        "input": query,
        #"chat_transaction_id": chat_transaction_id,
        "chat_history": chain_response,
        "output": main_content,
        "next_quest": next_quest_list,
        "usage": {
            "tokens_used": total_tokens,  # Total tokens used
            "prompt_tokens": input_token,  # Tokens used for prompts
            "completion_tokens": completion_tokens,  # Tokens used for completions
            "successful_requests": 0,  # Total successful requests
            "total_cost_usd": pricing,
        },      
        "SQL Query": sql_query,
        #"user_input_to_task": user_input_to_task,
        "response_time": response_time
    }

    print("=============")
    yield f"event: final_response\ndata: {json.dumps(response_serialized)}\n\n"


#@app.post("/chat")
#async def stream_chat(body: Body):
#    return StreamingResponse(response_generator(body.input, body.client_id,body.client_type, body.session_id), media_type="text/event-stream")

@app.post("/chat")
async def stream_chat(body: Body):
    try:
        generator = response_generator(
            body.input,
            body.client_id,
            body.client_type,
            body.session_id
        )
        return StreamingResponse(generator, media_type="text/event-stream")

    except Exception as e:
        # Optional: log the error
        import traceback
        traceback.print_exc()

        # Return a proper error response in JSON
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to start chat stream: {str(e)}"}
        )
@app.get("/chat_metadata/{session_id}")
async def get_metadata(session_id: str):
    if session_id not in SESSION_METADATA:
        return JSONResponse({"message": "Metadata not ready"}, status_code=404)
    return JSONResponse(SESSION_METADATA[session_id])

@app.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "ok"}

