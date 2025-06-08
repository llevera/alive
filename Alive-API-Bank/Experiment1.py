import sys
import os
import json
from typing import TypedDict, Optional, List as TypingList, Sequence, Annotated, Union, Dict, Type, Any

import operator
import traceback # Added for more detailed error reporting
import pickle # For caching
import hashlib # For cache signature
import glob # For walking directory for signature
import logging # <<< ADDED FOR LOGGING

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from pydantic import BaseModel, Field
from langchain.tools import Tool
from langchain_openai import AzureChatOpenAI

# --- Path Setup (early for logging) ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- LOGGING SETUP ---
LOG_FILENAME = os.path.join(script_dir, "script_run.log")
# Configure root logger
logging.basicConfig(
    level=logging.INFO,  # Capture INFO and above (WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILENAME, mode='w', encoding='utf-8'), # Log to a file, overwrite each run
        logging.StreamHandler(sys.stdout) # Log to console
    ]
)
# --- END LOGGING SETUP ---

# --- Load Domain Information Context from File ---
DOMAIN_CONTEXT_FILENAME = "domain_context.txt"
DOMAIN_INFORMATION_CONTEXT = ""
USE_DOMAIN_CONTEXT = False
YOUR_DEPLOYMENT_NAME = "gpt-4o"


# --- Cache Configuration ---
CACHE_DIR = os.path.join(script_dir, ".langgraph_cache")
TOOL_MANAGER_CACHE_FILE = os.path.join(CACHE_DIR, "tool_manager.pkl")
APIS_DIR_SIGNATURE_FILE = os.path.join(CACHE_DIR, "apis_dir_signature.txt")

def ensure_cache_dir_exists():
    if not os.path.exists(CACHE_DIR):
        try:
            os.makedirs(CACHE_DIR)
            logging.info(f"Created cache directory: {CACHE_DIR}")
        except OSError as e:
            logging.warning(f"Could not create cache directory {CACHE_DIR}: {e}. Caching will be disabled.")
            return False
    return True

CACHE_ENABLED = ensure_cache_dir_exists()

def get_apis_dir_signature(apis_dir_path: str) -> Optional[str]:
    """Generates a signature for the APIs directory based on file names and mtimes."""
    if not os.path.isdir(apis_dir_path):
        logging.warning(f"API directory {apis_dir_path} not found for signature generation.")
        return None

    file_details = []
    for root, _, files in os.walk(apis_dir_path):
        for file_name in sorted(files):
            file_path = os.path.join(root, file_name)
            try:
                mtime = os.path.getmtime(file_path)
                relative_path = os.path.relpath(file_path, apis_dir_path)
                normalized_relative_path = relative_path.replace(os.sep, '/')
                file_details.append(f"{normalized_relative_path}:{mtime}")
            except OSError:
                continue
    
    if not file_details:
        logging.warning(f"No files found or accessible in {apis_dir_path} for signature generation.")
        return "empty_dir_placeholder_signature"

    full_signature_string = ";".join(sorted(file_details))
    return hashlib.sha256(full_signature_string.encode('utf-8')).hexdigest()


try:
    domain_context_filepath = os.path.join(script_dir, DOMAIN_CONTEXT_FILENAME)
    if os.path.exists(domain_context_filepath):
        with open(domain_context_filepath, 'r', encoding='utf-8') as f:
            DOMAIN_INFORMATION_CONTEXT = f.read()
        if USE_DOMAIN_CONTEXT and not DOMAIN_INFORMATION_CONTEXT.strip():
            logging.warning(f"Domain context file '{domain_context_filepath}' is empty but USE_DOMAIN_CONTEXT is True.")
    elif USE_DOMAIN_CONTEXT:
        logging.error(f"Domain context file '{DOMAIN_CONTEXT_FILENAME}' not found at {domain_context_filepath}, and USE_DOMAIN_CONTEXT is True.")
        sys.exit(1)
except Exception as e:
    if USE_DOMAIN_CONTEXT:
        logging.error(f"Failed to load domain context from '{domain_context_filepath}' while USE_DOMAIN_CONTEXT is True: {e}")
        sys.exit(1)
    else:
        logging.info(f"Could not load domain context file '{DOMAIN_CONTEXT_FILENAME}': {e}. Proceeding without it as USE_DOMAIN_CONTEXT is False.")


# --- Continue with other path setups and imports ---
api_bank_package_dir = os.path.join(script_dir, "api_bank")
absolute_apis_path_for_toolmanager = os.path.join(api_bank_package_dir, "apis")

if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from api_bank.tool_manager import ToolManager # type: ignore
except ImportError as e:
    logging.critical(f"Failed to import ToolManager from api_bank: {e}")
    logging.critical("Ensure 'api_bank' is in your PYTHONPATH or accessible.")
    sys.exit(1)

# --- ToolManager Instance (with Caching) ---
tool_manager_instance = None
loaded_from_cache = False
current_apis_dir_sig = None # Define before conditional assignment

if CACHE_ENABLED:
    current_apis_dir_sig = get_apis_dir_signature(absolute_apis_path_for_toolmanager)
    if current_apis_dir_sig and os.path.exists(TOOL_MANAGER_CACHE_FILE) and os.path.exists(APIS_DIR_SIGNATURE_FILE):
        try:
            with open(APIS_DIR_SIGNATURE_FILE, 'r', encoding='utf-8') as f:
                cached_apis_dir_sig = f.read().strip()
            
            if cached_apis_dir_sig == current_apis_dir_sig:
                logging.info(f"API directory signature matches. Attempting to load ToolManager from cache: {TOOL_MANAGER_CACHE_FILE}")
                with open(TOOL_MANAGER_CACHE_FILE, 'rb') as f:
                    tool_manager_instance = pickle.load(f)
                
                if hasattr(tool_manager_instance, 'api_call') and hasattr(tool_manager_instance, 'init_tool') and hasattr(tool_manager_instance, 'apis'):
                    logging.info("ToolManager loaded successfully from cache.")
                    loaded_from_cache = True
                else:
                    logging.warning("Cached ToolManager object appears invalid or incomplete. Re-initializing.")
                    tool_manager_instance = None
                    loaded_from_cache = False
            else:
                logging.info("API directory signature mismatch. Cache is stale. Re-initializing ToolManager.")
        except Exception as e:
            logging.warning(f"Failed to load ToolManager from cache: {e}. Re-initializing.")
            tool_manager_instance = None # Ensure reset on failure
    # If current_apis_dir_sig is None (e.g. dir not found), this block is skipped, leading to re-init.
else:
    logging.info("Caching is disabled or cache directory could not be accessed.")


if not loaded_from_cache or tool_manager_instance is None:
    logging.info("Initializing ToolManager from scratch.")
    try:
        if not os.path.isdir(absolute_apis_path_for_toolmanager):
            logging.error(f"'apis' directory not found at {absolute_apis_path_for_toolmanager}.")
            sys.exit(1) # This error is critical for ToolManager initialization
        tool_manager_instance = ToolManager(apis_dir=absolute_apis_path_for_toolmanager)
        logging.info("ToolManager initialized.")

        # Re-fetch signature if it wasn't fetched because CACHE_ENABLED was true but files didn't exist
        if CACHE_ENABLED and current_apis_dir_sig is None:
            current_apis_dir_sig = get_apis_dir_signature(absolute_apis_path_for_toolmanager)

        if CACHE_ENABLED and tool_manager_instance and current_apis_dir_sig:
            try:
                logging.info(f"Saving ToolManager to cache: {TOOL_MANAGER_CACHE_FILE}")
                with open(TOOL_MANAGER_CACHE_FILE, 'wb') as f:
                    pickle.dump(tool_manager_instance, f, pickle.HIGHEST_PROTOCOL)
                with open(APIS_DIR_SIGNATURE_FILE, 'w', encoding='utf-8') as f:
                    f.write(current_apis_dir_sig)
                logging.info("ToolManager saved to cache.")
            except Exception as e:
                logging.warning(f"Failed to save ToolManager to cache: {e}")
                if os.path.exists(TOOL_MANAGER_CACHE_FILE): os.remove(TOOL_MANAGER_CACHE_FILE)
                if os.path.exists(APIS_DIR_SIGNATURE_FILE): os.remove(APIS_DIR_SIGNATURE_FILE)

    except Exception as e:
        logging.error(f"Failed to initialize ToolManager: {e}")
        logging.debug(traceback.format_exc()) # Add traceback for debug
        tool_manager_instance = None

if tool_manager_instance is None:
    logging.critical("ToolManager instance could not be initialized or loaded. Exiting.")
    sys.exit(1)
# --- End ToolManager Instance Caching Logic ---


def create_pydantic_model_for_tool(tool_name: str, input_params: dict) -> Type[BaseModel]:
    model_fields: Dict[str, tuple] = {}
    annotations_dict: Dict[str, Type] = {}
    safe_model_name_prefix = "".join(c if c.isalnum() else '_' for c in tool_name)

    for param_name, param_info in input_params.items():
        if not isinstance(param_info, dict): continue
        param_type_str = param_info.get('type', 'str')
        if not isinstance(param_type_str, str): param_type_str = 'str'

        field_type_hint: Type = Any
        if param_type_str == 'str': field_type_hint = str
        elif param_type_str == 'int': field_type_hint = int
        elif param_type_str == 'float': field_type_hint = float
        elif param_type_str == 'bool': field_type_hint = bool
        elif param_type_str == 'list[str]' or param_type_str == 'list(str)': field_type_hint = TypingList[str]
        elif param_type_str == 'list': field_type_hint = TypingList[Any]
        elif param_type_str == 'dict': field_type_hint = Dict[Any, Any]

        annotations_dict[param_name] = field_type_hint
        model_fields[param_name] = (field_type_hint, Field(..., description=param_info.get('description', '')))

    final_model_fields = {}
    for name, (type_hint, field_instance) in model_fields.items():
        final_model_fields[name] = field_instance
    
    # Ensure Pydantic model names are unique and valid Python identifiers
    unique_model_name = f"{safe_model_name_prefix}InputModel_{hashlib.md5(json.dumps(input_params, sort_keys=True).encode()).hexdigest()[:8]}"
    return type(unique_model_name, (BaseModel,), {'__annotations__': annotations_dict, **final_model_fields})


def get_langchain_tools_from_manager(manager: ToolManager) -> TypingList[Tool]:
    langchain_tools = []
    if manager.apis is None:
        logging.warning("ToolManager has no APIs loaded (manager.apis is None).")
        return []

    for api_info in manager.apis:
        tool_name = api_info['name']
        input_params_for_model = api_info.get('input_parameters', {})
        if not isinstance(input_params_for_model, dict):
            logging.warning(f"input_parameters for tool '{tool_name}' is not a dict, using empty. Value: {input_params_for_model}")
            input_params_for_model = {}
            
        args_schema_to_use = create_pydantic_model_for_tool(tool_name, input_params_for_model)

        def execute_api_call(captured_tool_name: str, **kwargs_for_api: dict) -> dict:
            try:
                tool_call_result_payload = manager.api_call(tool_name=captured_tool_name, **kwargs_for_api)
                
                if isinstance(tool_call_result_payload, dict) and \
                   all(k in tool_call_result_payload for k in ['api_name', 'input', 'output', 'exception']):
                    final_payload = tool_call_result_payload
                    final_payload['api_name'] = captured_tool_name
                    final_payload['input'] = kwargs_for_api # Ensure input is the one passed to manager.api_call
                    return final_payload
                else: # Adapt non-standard output from manager.api_call
                    return {
                        'api_name': captured_tool_name,
                        'input': kwargs_for_api,
                        'output': tool_call_result_payload,
                        'exception': None 
                    }
            except Exception as e:
                return {
                    'api_name': captured_tool_name,
                    'input': kwargs_for_api,
                    'output': None,
                    'exception': f"Wrapper exception in execute_api_call for {captured_tool_name}: {str(e)}"
                }

        langchain_tools.append(Tool(
            name=tool_name,
            description=api_info['description'],
            func=lambda tn=tool_name, **kwargs: execute_api_call(tn, **kwargs), # type: ignore
            args_schema=args_schema_to_use
        ))
    return langchain_tools

langchain_tools_list = get_langchain_tools_from_manager(tool_manager_instance)
if not langchain_tools_list and tool_manager_instance.apis:
     logging.warning("ToolManager reported APIs, but no LangChain tools were generated. Check tool definitions.")
elif not langchain_tools_list:
     logging.warning("No LangChain tools were generated. The agent will have no tools.")


class AgentState(TypedDict): messages: Annotated[Sequence[BaseMessage], operator.add]

required_env_vars = {"AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY")}
if not required_env_vars["AZURE_OPENAI_API_KEY"]:
    logging.critical("AZURE_OPENAI_API_KEY environment variable is missing.")
    sys.exit(1)

try:
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://andrewrslearningsweden.openai.azure.com/"),
        api_key=required_env_vars["AZURE_OPENAI_API_KEY"],
        api_version=os.getenv("OPENAI_API_VERSION", "2024-02-15-preview"),
        deployment_name=YOUR_DEPLOYMENT_NAME,
        seed=55663218
    )
    if langchain_tools_list:
        llm_with_tools = llm.bind_tools(langchain_tools_list)
    else:
        logging.info("No tools available to bind to LLM.")
        llm_with_tools = llm
except Exception as e:
    logging.critical(f"Error during AzureChatOpenAI initialization: {e}")
    sys.exit(1)

def agent_node(state: AgentState) -> AgentState:
    ai_response_message = llm_with_tools.invoke(state['messages'])
    return {"messages": [ai_response_message]}

def tool_node(state: AgentState) -> AgentState:
    last_message = state['messages'][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"messages": []}

    if not langchain_tools_list:
        logging.error("tool_node called but no tools are loaded.")
        error_messages = []
        for tool_call in last_message.tool_calls:
             error_messages.append(ToolMessage(
                content=json.dumps({
                    'api_name': tool_call['name'],
                    'input': tool_call['args'],
                    'output': None,
                    'exception': "No tools loaded in the system to handle this call."
                }),
                tool_call_id=tool_call['id']
            ))
        return {"messages": error_messages}

    tool_messages = []
    for tool_call in last_message.tool_calls:
        selected_tool = next((t for t in langchain_tools_list if t.name == tool_call['name']), None)
        raw_tool_output_dict_from_execute: Dict[str, Any]
        
        if selected_tool:
            try:
                # The selected_tool.func now is our execute_api_call wrapper
                raw_tool_output_dict_from_execute = selected_tool.func(**tool_call['args'])
            except Exception as e: # Should ideally be caught by execute_api_call's wrapper
                raw_tool_output_dict_from_execute = {
                    'api_name': tool_call['name'],
                    'input': tool_call['args'],
                    'output': None,
                    'exception': f"Error directly invoking tool '{tool_call['name']}' in tool_node: {str(e)}"
                }
        else:
            raw_tool_output_dict_from_execute = {
                "api_name": tool_call['name'],
                "input": tool_call['args'],
                "output": None,
                "exception": f"Tool '{tool_call['name']}' not found in langchain_tools_list."
            }
        
        # Ensure the output_dict has the expected structure if it came from execute_api_call
        # or even if it was an error case.
        if not isinstance(raw_tool_output_dict_from_execute, dict) or \
           not all(k in raw_tool_output_dict_from_execute for k in ['api_name', 'input', 'output', 'exception']):
            logging.warning(f"Tool output for '{tool_call['name']}' is not in the expected dict format. Reconstructing.")
            raw_tool_output_dict_from_execute = {
                 "api_name": tool_call['name'],
                 "input": tool_call['args'],
                 "output": raw_tool_output_dict_from_execute, # Store original if it was not a dict
                 "exception": "Output was not in expected dict format from tool/wrapper."
            }

        # Content for ToolMessage should be a string (often JSON string of the output part)
        output_content_for_tool_message = raw_tool_output_dict_from_execute.get('output')
        if output_content_for_tool_message is None and raw_tool_output_dict_from_execute.get('exception') is not None:
            output_content_for_tool_message = {"error": raw_tool_output_dict_from_execute['exception']}
        elif output_content_for_tool_message is None:
             output_content_for_tool_message = "No output returned from tool."

        try:
            # Ensure content is JSON serializable string for ToolMessage
            content_str = json.dumps(output_content_for_tool_message)
        except TypeError:
            content_str = str(output_content_for_tool_message) # Fallback to string representation


        tool_messages.append(ToolMessage(
            content=content_str,
            tool_call_id=tool_call['id'],
            # Pass the full raw_tool_output_dict for validation purposes
            additional_kwargs={"raw_output_dict": raw_tool_output_dict_from_execute}
        ))
    return {"messages": tool_messages}

def should_continue(state: AgentState) -> str:
    last_message = state['messages'][-1]
    if isinstance(last_message, AIMessage):
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls and langchain_tools_list:
            return "invoke_tools"
        return END
    return END

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
if langchain_tools_list:
    workflow.add_node("tools", tool_node)
    workflow.add_conditional_edges("agent", should_continue, {"invoke_tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
else:
    workflow.add_conditional_edges("agent", should_continue, {END: END})

workflow.set_entry_point("agent")
app = workflow.compile()


def validate_test_case(
    actual_agent_actions: TypingList[Dict],
    expected_apis_to_validate: TypingList[Dict],
    tool_mgr: ToolManager
) -> tuple[bool, str]:
    """
    Validates a test case based on the agent's first API call compared to the first expected API call.
    """
    agent_actions_report_lines = ["Agent's Actual API Call Sequence (for informational purposes):"]
    if actual_agent_actions:
        for idx, action in enumerate(actual_agent_actions):
            action_output_dict = action.get('output_dict', {}) # This is our full raw_output_dict
            result_display = json.dumps(action_output_dict.get('output', "N/A"))
            exception_display = action_output_dict.get('exception')
            
            log_line = (
                f"  {idx+1}. API Name: {action_output_dict.get('api_name', action.get('name', 'N/A'))}, " # Prefer api_name from output_dict
                f"Args: {json.dumps(action_output_dict.get('input', action.get('args', {})))}, " # Prefer input from output_dict
                f"Output: {result_display}"
            )
            if exception_display:
                log_line += f", Exception: {exception_display}"
            agent_actions_report_lines.append(log_line)
    else:
        agent_actions_report_lines.append("  (No API calls were made by the agent)")
    agent_actions_report = "\n".join(agent_actions_report_lines)

    validation_messages = ["\nValidation of First Agent API Call vs. First Expected API Call (determines Pass/Fail):"]
    is_overall_pass = False

    if not actual_agent_actions:
        if not expected_apis_to_validate:
            validation_messages.append("  Agent made no API calls, and no API calls were expected. Test considered PASS.")
            is_overall_pass = True
        else:
            expected_first_api_name = expected_apis_to_validate[0].get('api_name', 'N/A') if expected_apis_to_validate else "N/A"
            validation_messages.append(f"  ❌ FAIL: Agent made no API calls, but an API call (e.g., '{expected_first_api_name}') was expected as the first action.")
            is_overall_pass = False
    elif not expected_apis_to_validate:
        actual_first_call_output_dict = actual_agent_actions[0].get('output_dict', {})
        actual_first_call_name = actual_first_call_output_dict.get('api_name', actual_agent_actions[0].get('name', 'N/A'))
        validation_messages.append(f"  ❌ FAIL: Agent made an API call ('{actual_first_call_name}'), but 'expected_apis_to_validate' was empty.")
        is_overall_pass = False
    else:
        actual_first_call_agent_perspective = actual_agent_actions[0] # name, args
        agent_executed_call_record = actual_first_call_agent_perspective.get('output_dict', {}) # output_dict from ToolMessage
        
        # Ensure agent_executed_call_record has the required structure. It should if coming from our tool_node.
        if not (isinstance(agent_executed_call_record, dict) and
                'api_name' in agent_executed_call_record and
                'input' in agent_executed_call_record): # output and exception are good to have but not strictly needed for matching api_name/input
            validation_messages.append(f"  ⚠️ WARNING: Agent's first call 'output_dict' is malformed or incomplete. Attempting to reconstruct for validation.")
            agent_executed_call_record = {
                 "api_name": actual_first_call_agent_perspective.get('name'), # From AI ToolCall
                 "input": actual_first_call_agent_perspective.get('args', {}), # From AI ToolCall
                 "output": agent_executed_call_record.get('output', "Error: output_dict malformed in validation"),
                 "exception": agent_executed_call_record.get('exception', "Error: output_dict malformed in validation")
            }

        expected_first_api_spec = expected_apis_to_validate[0]
        expected_api_name = expected_first_api_spec.get('api_name')
        expected_api_input_args = expected_first_api_spec.get('input', {})
        expected_api_output = expected_first_api_spec.get('output')
        expected_api_exception = expected_first_api_spec.get('exception')

        agent_actual_call_name_from_record = agent_executed_call_record.get('api_name')
        agent_actual_call_args_from_record = agent_executed_call_record.get('input')


        validation_messages.append(f"  Comparing Agent's First Call with First Expected API:")
        validation_messages.append(f"    - Agent's First Call: Name='{agent_actual_call_name_from_record}', Args Used={json.dumps(agent_actual_call_args_from_record)}")
        validation_messages.append(f"    - First Expected API: Name='{expected_api_name}', Expected Input Args={json.dumps(expected_api_input_args)}")

        if agent_actual_call_name_from_record != expected_api_name:
            validation_messages.append(f"    - ❌ NAME MISMATCH: Agent called '{agent_actual_call_name_from_record}', Expected '{expected_api_name}'.")
            is_overall_pass = False
        else:
            validation_messages.append(f"    - ✔️ NAME MATCH: Agent called '{agent_actual_call_name_from_record}' as expected.")
            groundtruth_record_for_check = {
                'api_name': expected_api_name,
                'input': expected_api_input_args,
                'output': expected_api_output,
                'exception': expected_api_exception
            }
            api_tool_instance = None
            try:
                api_tool_instance = tool_mgr.init_tool(expected_api_name)
            except Exception as e:
                is_overall_pass = False
                validation_messages.append(f"    - VALIDATION ERROR: Could not initialize tool '{expected_api_name}'. Error: {e}")
            
            if api_tool_instance and not hasattr(api_tool_instance, 'check_api_call_correctness'):
                is_overall_pass = False
                validation_messages.append(f"    - VALIDATION ERROR: Tool '{expected_api_name}' is missing 'check_api_call_correctness'.")
            elif api_tool_instance:
                try:
                    is_call_correct = api_tool_instance.check_api_call_correctness(
                        response=agent_executed_call_record,
                        groundtruth=groundtruth_record_for_check
                    )
                    if is_call_correct:
                        validation_messages.append(f"    - ✔️ ARGS/INVOCATION VALIDATION: PASSES via check_api_call_correctness.")
                        is_overall_pass = True
                    else:
                        is_overall_pass = False
                        validation_messages.append(
                            f"    - ❌ ARGS/INVOCATION VALIDATION: FAILS via check_api_call_correctness.\n"
                            f"        Agent's Executed Call (response):\n{json.dumps(agent_executed_call_record, indent=10)}\n"
                            f"        Expected API Spec (groundtruth):\n{json.dumps(groundtruth_record_for_check, indent=10)}"
                        )
                except Exception as e:
                    is_overall_pass = False
                    validation_messages.append(
                        f"    - ERROR during 'check_api_call_correctness' for '{expected_api_name}'. Error: {str(e)}\n"
                        f"      Traceback: {traceback.format_exc()}\n"
                        f"      Agent's Executed Call (response):\n{json.dumps(agent_executed_call_record, indent=10)}\n"
                        f"      Expected API Spec (groundtruth):\n{json.dumps(groundtruth_record_for_check, indent=10)}"
                    )
    final_summary_message = agent_actions_report + "\n" + "\n".join(validation_messages)
    return is_overall_pass, final_summary_message


# --- Running the Graph with test cases ---
if __name__ == "__main__":
    logging.info("\n--- Running LangGraph Validation (Single First API Call Based) ---")
    if USE_DOMAIN_CONTEXT and DOMAIN_INFORMATION_CONTEXT and DOMAIN_INFORMATION_CONTEXT.strip():
        logging.info(f"Using Domain Context from '{DOMAIN_CONTEXT_FILENAME}'")
    elif USE_DOMAIN_CONTEXT:
        logging.info(f"USE_DOMAIN_CONTEXT is True, but Domain Context from '{DOMAIN_CONTEXT_FILENAME}' is empty or only whitespace. Proceeding with minimal system message if any.")
    else:
        logging.info("Domain Context usage is disabled (USE_DOMAIN_CONTEXT=False).")

    simplecall_json_path = os.path.join(script_dir, "api_bank", "test_data", "simplecall.json")

    if not os.path.exists(simplecall_json_path):
        # Simplified path checking
        logging.error(f"simplecall.json not found at expected path: {simplecall_json_path}")
        sys.exit(1)
    logging.info(f"Using simplecall.json from: {simplecall_json_path}")

    try:
        with open(simplecall_json_path, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
    except Exception as e:
        logging.critical(f"Error loading or parsing simplecall.json from {simplecall_json_path}: {e}")
        sys.exit(1)
        
    total_cases = len(test_cases)
    passed_cases = 0
    failed_cases = 0
    skipped_cases = 0

    logging.info(f"ToolManager ready. Available APIs in manager: {len(tool_manager_instance.apis or [])}.")
    logging.info(f"Langchain tools generated: {len(langchain_tools_list)}.")

    for i, test_case in enumerate(test_cases):
        case_number = i + 1
        logging.info(f"\n" + "="*80)
        logging.info(f"--- Test Case {case_number}/{total_cases}: {test_case.get('scene_description', 'N/A')} ---")
        logging.info("="*80)

        user_query = test_case.get("initial_user_utterance")
        relevant_info = test_case.get("relevant_key_info_for_scenario_understanding")
        expected_apis_for_validation = test_case.get("apis", [])

        if not user_query:
            logging.warning(f"🛑 Test Case {case_number} SKIPPED: missing 'initial_user_utterance'.")
            skipped_cases +=1
            continue

        logging.info(f"🗣️ User Query (Original): '{user_query}'")

        full_user_query = user_query
        if relevant_info:
            try:
                relevant_info_str = json.dumps(relevant_info, indent=2)
                full_user_query += f"\n\nHere is some relevant information for this request:\n{relevant_info_str}"
                logging.info(f"ℹ️ Query Augmented with relevant_info:\n{relevant_info_str}")
            except TypeError:
                logging.warning(f"relevant_info for test case {case_number} is not JSON serializable, appending as string.")
                full_user_query += f"\n\nHere is some relevant information for this request:\n{relevant_info}"

        logging.info("\n📋 First Expected API Call for Validation (from simplecall.json 'apis'[0]):")
        if expected_apis_for_validation:
            first_expected_api = expected_apis_for_validation[0]
            logging.info(f"  1. API Name: {first_expected_api.get('api_name', 'N/A')}")
            logging.info(f"     Input   : {json.dumps(first_expected_api.get('input', {}))}")
            logging.info(f"     Expected Output (for groundtruth): {json.dumps(first_expected_api.get('output'))}")
            if first_expected_api.get('exception') is not None:
                logging.info(f"     Expected Exception (for groundtruth): {json.dumps(first_expected_api.get('exception'))}")
        else:
            logging.info("  (No specific API calls are listed in 'apis' key for this test case to validate against)")
        logging.info("-" * 30)

        actual_tool_actions_from_agent = []
        pending_tool_calls_for_agent = {} # Track AI's tool calls to match with ToolMessage

        initial_messages_for_graph: TypingList[BaseMessage] = []
        if USE_DOMAIN_CONTEXT and DOMAIN_INFORMATION_CONTEXT and DOMAIN_INFORMATION_CONTEXT.strip():
            system_context_content = f"IMPORTANT SYSTEM DOMAIN INFORMATION FOR TOOL SELECTION AND USAGE:\n\n{DOMAIN_INFORMATION_CONTEXT}"
            system_context_message = SystemMessage(content=system_context_content)
            initial_messages_for_graph.append(system_context_message)
            logging.info(f"System Context Prepended: True (Length: {len(system_context_content)})")


        initial_messages_for_graph.append(HumanMessage(content=full_user_query))
        
        if hasattr(tool_manager_instance, 'reset_dbs') and callable(tool_manager_instance.reset_dbs):
            logging.info("Resetting ToolManager databases for new test case.")
            tool_manager_instance.reset_dbs()

        final_state_stream = app.with_config({"recursion_limit": 15}).stream({"messages": initial_messages_for_graph})

        logging.info("\n💬 Agent Execution Log:")
        for event_idx, event_chunk in enumerate(final_state_stream):
            for node_name, output_from_node in event_chunk.items():
                # Log at DEBUG level for potentially verbose node outputs
                logging.debug(f"  Event {event_idx}, Node '{node_name}': Raw output: {output_from_node}")
                if 'messages' in output_from_node and output_from_node['messages']:
                    for msg_idx, msg in enumerate(output_from_node['messages']):
                        logging.info(f"    Msg {msg_idx} Type: {type(msg).__name__}")
                        if isinstance(msg, AIMessage):
                            ai_content = msg.content[:200] + ('...' if len(msg.content) > 200 else '')
                            logging.info(f"      AI: {ai_content}")
                            if msg.tool_calls:
                                logging.info(f"      Tool Calls by AI:")
                                for tc_idx, tc in enumerate(msg.tool_calls):
                                    logging.info(f"        {tc_idx+1}. ID: {tc['id']}, Name: {tc['name']}, Args: {json.dumps(tc['args'])}")
                                    pending_tool_calls_for_agent[tc['id']] = {"name": tc['name'], "args": tc['args']}
                        elif isinstance(msg, ToolMessage):
                            raw_output_dict = msg.additional_kwargs.get("raw_output_dict", {}) # Should be populated by tool_node
                            tool_call_name_for_log = raw_output_dict.get('api_name', 'N/A')
                            logging.info(f"      Tool ID: {msg.tool_call_id}, Name from raw_output_dict: {tool_call_name_for_log}")
                            logging.info(f"      Tool Content (part): {msg.content[:200]}{'...' if len(msg.content) > 200 else ''}")
                            logging.debug(f"      Tool Full raw_output_dict: {json.dumps(raw_output_dict)}")


                            # Populate actual_tool_actions_from_agent using the raw_output_dict
                            # Ensure the raw_output_dict is properly formed by tool_node
                            if msg.tool_call_id in pending_tool_calls_for_agent:
                                if isinstance(raw_output_dict, dict) and 'api_name' in raw_output_dict and 'input' in raw_output_dict:
                                    actual_tool_actions_from_agent.append({
                                        "name": raw_output_dict['api_name'], # From the tool execution record
                                        "args": raw_output_dict['input'],   # From the tool execution record
                                        "output_dict": raw_output_dict      # The full record
                                    })
                                    pending_tool_calls_for_agent.pop(msg.tool_call_id) # remove after processing
                                else:
                                    logging.warning(f"ToolMessage (ID: {msg.tool_call_id}) raw_output_dict is malformed. Details: {raw_output_dict}")
                            else:
                                logging.warning(f"Received ToolMessage for untracked tool_call_id: {msg.tool_call_id}. Raw output: {raw_output_dict}")
                                # Still try to record it if possible
                                if isinstance(raw_output_dict, dict) and raw_output_dict.get('api_name') and raw_output_dict.get('input') is not None:
                                    actual_tool_actions_from_agent.append({
                                        "name": raw_output_dict['api_name'],
                                        "args": raw_output_dict['input'],
                                        "output_dict": raw_output_dict
                                    })


        is_pass, validation_summary_message = validate_test_case(
            actual_tool_actions_from_agent,
            expected_apis_for_validation,
            tool_manager_instance
        )

        logging.info("\n--- Validation Summary (Single First Call) ---")
        # The validation_summary_message itself is multi-line and formatted, log as one info message
        for line in validation_summary_message.splitlines():
            logging.info(line)


        if is_pass:
            logging.info(f"\n✅ Test Case {case_number}: PASS (Based on first agent API call vs first expected API)")
            passed_cases += 1
        else:
            logging.info(f"\n❌ Test Case {case_number}: FAIL (Based on first agent API call vs first expected API)")
            failed_cases += 1

    logging.info("\n" + "="*80)
    logging.info("--- Overall Summary (Single First Call Validation) ---")
    logging.info(f"Total Cases: {total_cases}")
    logging.info(f"✅ Passed   : {passed_cases}")
    logging.info(f"❌ Failed   : {failed_cases}")
    if skipped_cases > 0:
        logging.info(f"🛑 Skipped  : {skipped_cases}")
    logging.info("="*80)
    logging.info("--- Evaluation Complete ---")

    logging.shutdown() # Ensure all handlers are closed properly