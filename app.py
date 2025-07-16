import asyncio
import json
import re
import time
from typing import Optional
from botocore.config import Config
import boto3
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
from agent_squad.storage import InMemoryChatStorage
from agent_squad.classifiers import BedrockClassifier, BedrockClassifierOptions
import tiktoken
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse,JSONResponse
from constants import AGENT_PROMPT
from test_sql import run_query_tool_with_fallback
orchestrator = None
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def make_bedrock_client(region="us-west-2"):
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
classifier = BedrockClassifier(BedrockClassifierOptions(model_id=model_id,
                                                        region="us-west-2"))

class Body(BaseModel):
    input: str
    user_id: str  
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

def setup_core_agent():
    
    global GLOBAL_DB_CREDS

    #AGENT_PROMPT 

    fin_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
            name="sql_translation",
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
            #retriever=retriever,  # keep using few-shot example retriever
            save_chat=True,
            custom_system_prompt={
                "template": AGENT_PROMPT,
                "variables": {
                    #"TABLEINFO": stringify_table_info(NS_TABLE_INFO),
                    "HISTORY":""
                }
            }, 
            client=make_bedrock_client("us-west-2")
        ))
    

    
    safe_add_agent(fin_agent, orchestrator)

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

MAX_RETRIES = 3
RETRY_DELAY = 2
HEARTBEAT_INTERVAL = 10

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

def count_prompt_tokens(prompt: str) -> int:
    
    model_encoding: str = "cl100k_base"
    encoding = tiktoken.get_encoding(model_encoding)
    tokens = encoding.encode(prompt)
    return len(tokens)

def extract_sql_query(text: str) -> str:
    
    pattern = r"```sql\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

async def response_generator(query, user_id,  session_id):
    print("Available agents in orchestrator:", list(orchestrator.agents.keys()))

    start_time = time.time()
    agent_output_chunks = []
    final_response = ""
    chat_history = []
    chain_response = ""
    
    setup_core_agent()

    #add history

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

    #update history

    #usage
    input_token = count_prompt_tokens(query)
    total_tokens = fin_agent_callbacks.token_count 
    completion_tokens = total_tokens - input_token
    pricing = round((input_token * 0.003 + completion_tokens * 0.015) / 1000, 5)

    sql_query = extract_sql_query(chain_response.strip())

    response_time = round(time.time() - start_time, 2)

    response_serialized = {
        "input": query,
        #"chat_transaction_id": chat_transaction_id,
        "output": chain_response.strip(),
        "usage": {
            "tokens_used": total_tokens,  # Total tokens used
            "prompt_tokens": input_token,  # Tokens used for prompts
            "completion_tokens": completion_tokens,  # Tokens used for completions
            "successful_requests": 0,  # Total successful requests
            "total_cost_usd": pricing,
        },      
        "SQL Query": sql_query,
        "response_time": response_time
    }

    print("=============")
    yield f"event: final_response\ndata: {json.dumps(response_serialized)}\n\n"


@app.post("/chat")
async def stream_chat(body: Body):
    try:
        generator = response_generator(
            body.input,
            body.user_id,
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

@app.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "ok"}

                