import sys
import os
import json
from typing import TypedDict, Optional, List as TypingList, Sequence, Annotated, Union, Dict, Type, Any

import operator
import traceback # Added for more detailed error reporting
import pickle # For caching
import hashlib # For cache signature
import glob # For walking directory for signature

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from pydantic import BaseModel, Field
from langchain.tools import Tool
from langchain_openai import AzureChatOpenAI

# --- Load Domain Information Context from File ---
DOMAIN_CONTEXT_FILENAME = "domain_context.txt"
DOMAIN_INFORMATION_CONTEXT = ""
USE_DOMAIN_CONTEXT = True 
YOUR_DEPLOYMENT_NAME = "gpt-4o-mini"


# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- Cache Configuration ---
CACHE_DIR = os.path.join(script_dir, ".langgraph_cache") # Changed name to be more specific
TOOL_MANAGER_CACHE_FILE = os.path.join(CACHE_DIR, "tool_manager.pkl")
APIS_DIR_SIGNATURE_FILE = os.path.join(CACHE_DIR, "apis_dir_signature.txt")

def ensure_cache_dir_exists():
    if not os.path.exists(CACHE_DIR):
        try:
            os.makedirs(CACHE_DIR)
            print(f"INFO: Created cache directory: {CACHE_DIR}")
        except OSError as e:
            print(f"WARNING: Could not create cache directory {CACHE_DIR}: {e}. Caching will be disabled.")
            return False
    return True

CACHE_ENABLED = ensure_cache_dir_exists()

def get_apis_dir_signature(apis_dir_path: str) -> Optional[str]:
    """Generates a signature for the APIs directory based on file names and mtimes."""
    if not os.path.isdir(apis_dir_path):
        print(f"WARNING: API directory {apis_dir_path} not found for signature generation.")
        return None

    file_details = []
    # Consider all files recursively, ToolManager might depend on non-Python files too.
    for root, _, files in os.walk(apis_dir_path):
        for file_name in sorted(files): # Sort files for consistent order
            file_path = os.path.join(root, file_name)
            try:
                mtime = os.path.getmtime(file_path)
                relative_path = os.path.relpath(file_path, apis_dir_path)
                # Normalize path separators for cross-platform consistency in signature
                normalized_relative_path = relative_path.replace(os.sep, '/')
                file_details.append(f"{normalized_relative_path}:{mtime}")
            except OSError:
                # File might be a broken symlink or deleted during walk
                continue
    
    if not file_details:
        # This could happen if apis_dir_path is empty or contains only unreadable files
        print(f"WARNING: No files found or accessible in {apis_dir_path} for signature generation.")
        return "empty_dir_placeholder_signature" # Or None, but a placeholder makes it distinct from error


    # Sort all collected details to ensure the final string is order-independent
    # The file_details list itself is already built from sorted file names within each directory,
    # and os.walk order can vary, so an overall sort of collected details is good.
    full_signature_string = ";".join(sorted(file_details))
    return hashlib.sha256(full_signature_string.encode('utf-8')).hexdigest()


try:
    domain_context_filepath = os.path.join(script_dir, DOMAIN_CONTEXT_FILENAME)
    if os.path.exists(domain_context_filepath):
        with open(domain_context_filepath, 'r', encoding='utf-8') as f:
            DOMAIN_INFORMATION_CONTEXT = f.read()
        if USE_DOMAIN_CONTEXT and not DOMAIN_INFORMATION_CONTEXT.strip():
            print(f"WARNING: Domain context file '{domain_context_filepath}' is empty but USE_DOMAIN_CONTEXT is True.")
    elif USE_DOMAIN_CONTEXT: 
        print(f"ERROR: Domain context file '{DOMAIN_CONTEXT_FILENAME}' not found at {domain_context_filepath}, and USE_DOMAIN_CONTEXT is True.")
        sys.exit(1)
except Exception as e:
    if USE_DOMAIN_CONTEXT:
        print(f"ERROR: Failed to load domain context from '{domain_context_filepath}' while USE_DOMAIN_CONTEXT is True: {e}")
        sys.exit(1)
    else:
        print(f"INFO: Could not load domain context file '{DOMAIN_CONTEXT_FILENAME}': {e}. Proceeding without it as USE_DOMAIN_CONTEXT is False.")


# --- Continue with other path setups and imports ---
api_bank_package_dir = os.path.join(script_dir, "api_bank")
absolute_apis_path_for_toolmanager = os.path.join(api_bank_package_dir, "apis")

if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from api_bank.tool_manager import ToolManager # type: ignore
except ImportError as e:
    print(f"ERROR: Failed to import ToolManager from api_bank: {e}")
    print("Ensure 'api_bank' is in your PYTHONPATH or accessible.")
    sys.exit(1)

# --- ToolManager Instance (with Caching) ---
tool_manager_instance = None
loaded_from_cache = False

if CACHE_ENABLED:
    current_apis_dir_sig = get_apis_dir_signature(absolute_apis_path_for_toolmanager)
    if current_apis_dir_sig and os.path.exists(TOOL_MANAGER_CACHE_FILE) and os.path.exists(APIS_DIR_SIGNATURE_FILE):
        try:
            with open(APIS_DIR_SIGNATURE_FILE, 'r', encoding='utf-8') as f:
                cached_apis_dir_sig = f.read().strip()
            
            if cached_apis_dir_sig == current_apis_dir_sig:
                print(f"INFO: API directory signature matches. Attempting to load ToolManager from cache: {TOOL_MANAGER_CACHE_FILE}")
                with open(TOOL_MANAGER_CACHE_FILE, 'rb') as f:
                    tool_manager_instance = pickle.load(f)
                
                if hasattr(tool_manager_instance, 'api_call') and hasattr(tool_manager_instance, 'init_tool') and hasattr(tool_manager_instance, 'apis'):
                    print("INFO: ToolManager loaded successfully from cache.")
                    loaded_from_cache = True
                else:
                    print("WARNING: Cached ToolManager object appears invalid or incomplete. Re-initializing.")
                    tool_manager_instance = None 
                    loaded_from_cache = False 
            else:
                print("INFO: API directory signature mismatch. Cache is stale. Re-initializing ToolManager.")
        except Exception as e:
            print(f"WARNING: Failed to load ToolManager from cache: {e}. Re-initializing.")
            tool_manager_instance = None
else:
    print("INFO: Caching is disabled or cache directory could not be accessed.")


if not loaded_from_cache or tool_manager_instance is None:
    print("INFO: Initializing ToolManager from scratch.")
    try:
        if not os.path.isdir(absolute_apis_path_for_toolmanager):
            print(f"ERROR: 'apis' directory not found at {absolute_apis_path_for_toolmanager}.")
            sys.exit(1)
        tool_manager_instance = ToolManager(apis_dir=absolute_apis_path_for_toolmanager)
        print("INFO: ToolManager initialized.")

        if CACHE_ENABLED and tool_manager_instance and current_apis_dir_sig:
            try:
                print(f"INFO: Saving ToolManager to cache: {TOOL_MANAGER_CACHE_FILE}")
                with open(TOOL_MANAGER_CACHE_FILE, 'wb') as f:
                    pickle.dump(tool_manager_instance, f, pickle.HIGHEST_PROTOCOL)
                with open(APIS_DIR_SIGNATURE_FILE, 'w', encoding='utf-8') as f:
                    f.write(current_apis_dir_sig)
                print("INFO: ToolManager saved to cache.")
            except Exception as e:
                print(f"WARNING: Failed to save ToolManager to cache: {e}")
                # If saving fails, remove potentially corrupt cache files
                if os.path.exists(TOOL_MANAGER_CACHE_FILE): os.remove(TOOL_MANAGER_CACHE_FILE)
                if os.path.exists(APIS_DIR_SIGNATURE_FILE): os.remove(APIS_DIR_SIGNATURE_FILE)

    except Exception as e:
        print(f"ERROR: Failed to initialize ToolManager: {e}")
        # Ensure tool_manager_instance is None if initialization fails critically
        tool_manager_instance = None 
        # No sys.exit(1) here, will be caught by the check below

if tool_manager_instance is None:
    print("CRITICAL ERROR: ToolManager instance could not be initialized or loaded. Exiting.")
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
    
    return type(f"{safe_model_name_prefix}InputModel", (BaseModel,), {'__annotations__': annotations_dict, **final_model_fields})


def get_langchain_tools_from_manager(manager: ToolManager) -> TypingList[Tool]:
    langchain_tools = []
    if manager.apis is None: # manager.apis should exist if ToolManager loaded correctly
        print("Warning: ToolManager has no APIs loaded (manager.apis is None).")
        return []

    for api_info in manager.apis:
        tool_name = api_info['name']
        # Ensure input_parameters is a dictionary, provide empty if missing or not a dict
        input_params_for_model = api_info.get('input_parameters', {})
        if not isinstance(input_params_for_model, dict):
            print(f"Warning: input_parameters for tool '{tool_name}' is not a dict, using empty. Value: {input_params_for_model}")
            input_params_for_model = {}
            
        args_schema_to_use = create_pydantic_model_for_tool(tool_name, input_params_for_model)

        def execute_api_call(captured_tool_name: str, **kwargs_for_api: dict) -> dict:
            try:
                tool_call_result_payload = manager.api_call(tool_name=captured_tool_name, **kwargs_for_api)
                
                if isinstance(tool_call_result_payload, dict) and \
                   all(k in tool_call_result_payload for k in ['api_name', 'input', 'output', 'exception']):
                    final_payload = tool_call_result_payload
                    final_payload['api_name'] = captured_tool_name 
                    final_payload['input'] = kwargs_for_api
                    return final_payload
                else:
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
                    'exception': f"Wrapper exception in execute_api_call: {str(e)}"
                }

        langchain_tools.append(Tool(
            name=tool_name,
            description=api_info['description'],
            func=lambda tn=tool_name, **kwargs: execute_api_call(tn, **kwargs), # type: ignore
            args_schema=args_schema_to_use
        ))
    return langchain_tools

langchain_tools_list = get_langchain_tools_from_manager(tool_manager_instance)
if not langchain_tools_list and tool_manager_instance.apis: # If apis were present but no tools made
     print("WARNING: ToolManager reported APIs, but no LangChain tools were generated. Check tool definitions.")
elif not langchain_tools_list:
     print("WARNING: No LangChain tools were generated. The agent will have no tools.")


class AgentState(TypedDict): messages: Annotated[Sequence[BaseMessage], operator.add]

required_env_vars = {"AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY")}
if not required_env_vars["AZURE_OPENAI_API_KEY"]:
    print("ERROR: AZURE_OPENAI_API_KEY environment variable is missing.")
    sys.exit(1)

try:
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://andrewrslearningsweden.openai.azure.com/"),
        api_key=required_env_vars["AZURE_OPENAI_API_KEY"],
        api_version=os.getenv("OPENAI_API_VERSION", "2024-02-15-preview"),
        deployment_name=YOUR_DEPLOYMENT_NAME,
        seed=55663218
    )
    if langchain_tools_list: # Only bind tools if there are any
        llm_with_tools = llm.bind_tools(langchain_tools_list)
    else: # No tools to bind
        print("INFO: No tools available to bind to LLM.")
        llm_with_tools = llm 
except Exception as e:
    print(f"ERROR: Error during AzureChatOpenAI initialization: {e}")
    sys.exit(1)

def agent_node(state: AgentState) -> AgentState:
    # llm_with_tools will be just llm if langchain_tools_list was empty
    ai_response_message = llm_with_tools.invoke(state['messages'])
    return {"messages": [ai_response_message]}

def tool_node(state: AgentState) -> AgentState:
    last_message = state['messages'][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"messages": []} 

    if not langchain_tools_list: # Defensive check
        print("ERROR: tool_node called but no tools are loaded.")
        # Return error messages or handle as appropriate
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
                raw_tool_output_dict_from_execute = selected_tool.func(**tool_call['args'])
            except Exception as e:
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

        tool_messages.append(ToolMessage(
            content=json.dumps(raw_tool_output_dict_from_execute.get('output', 'No output due to error or structure.')), 
            tool_call_id=tool_call['id'],
            additional_kwargs={"raw_output_dict": raw_tool_output_dict_from_execute} 
        ))
    return {"messages": tool_messages}

def should_continue(state: AgentState) -> str:
    last_message = state['messages'][-1]
    if isinstance(last_message, AIMessage):
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls and langchain_tools_list: # Also check if tools exist
            return "invoke_tools"
        return END
    return END

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
# Only add tool node if tools are available
if langchain_tools_list:
    workflow.add_node("tools", tool_node)
    workflow.add_conditional_edges("agent", should_continue, {"invoke_tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
else: # No tools, agent directly goes to END if no tool calls are made (or should_continue handles it)
    workflow.add_conditional_edges("agent", should_continue, {END: END}) # "invoke_tools" won't be an option

workflow.set_entry_point("agent")
app = workflow.compile()


def validate_test_case(
    actual_agent_actions: TypingList[Dict], 
    expected_apis_to_validate: TypingList[Dict], 
    tool_mgr: ToolManager # tool_mgr is the tool_manager_instance
) -> tuple[bool, str]:
    """
    Validates a test case.
    Part 1: Reports the agent's actual API calls (informational).
    Part 2: Validates if directly calling `expected_apis_to_validate` yields specified outcomes.
            This part determines the pass/fail status of the test case.
    """

    # --- Part 1: Report Agent's Actual Calls (Informational) ---
    agent_actions_report_lines = ["Agent's Actual API Call Sequence (for informational purposes):"]
    if actual_agent_actions:
        for idx, action in enumerate(actual_agent_actions):
            action_output_dict = action.get('output_dict', {})
            result_display = json.dumps(action_output_dict.get('output', "N/A"))
            exception_display = action_output_dict.get('exception')
            
            log_line = (
                f"  {idx+1}. API Name: {action.get('name', 'N/A')}, "
                f"Args: {json.dumps(action.get('args', {}))}, "
                f"Output: {result_display}"
            )
            if exception_display:
                log_line += f", Exception: {exception_display}"
            agent_actions_report_lines.append(log_line)
    else:
        agent_actions_report_lines.append("  (No API calls were made by the agent)")
    agent_actions_report = "\n".join(agent_actions_report_lines)

    # --- Part 2: Validate Expected API Outcomes (Determines Pass/Fail) ---
    all_expected_outcomes_match = True 
    outcome_validation_messages = ["\nValidation of Expected API Outcomes (determines Pass/Fail):"]

    if not expected_apis_to_validate:
        outcome_validation_messages.append("  No expected API outcomes to validate. Test considered PASS by default for this part.")
    else:
        for i, expected_api_spec in enumerate(expected_apis_to_validate):
            call_number = i + 1
            expected_tool_name = expected_api_spec.get('api_name')
            expected_input_args = expected_api_spec.get('input', {})
            expected_output_groundtruth = expected_api_spec.get('output')
            expected_exception_groundtruth = expected_api_spec.get('exception') 

            outcome_validation_messages.append(f"\n  Validating Expected API Call {call_number}: {expected_tool_name}")
            outcome_validation_messages.append(f"    - Expected Input (from JSON): {json.dumps(expected_input_args)}")
            outcome_validation_messages.append(f"    - Expected Output (from JSON): {json.dumps(expected_output_groundtruth)}")
            if expected_exception_groundtruth is not None: 
                 outcome_validation_messages.append(f"    - Expected Exception (from JSON): {expected_exception_groundtruth}")

            standardized_direct_call_record: Dict[str, Any]
            try:
                direct_call_raw_output_or_dict = tool_mgr.api_call(tool_name=expected_tool_name, **expected_input_args)
                
                if isinstance(direct_call_raw_output_or_dict, dict) and \
                   all(k in direct_call_raw_output_or_dict for k in ['api_name', 'input', 'output', 'exception']):
                    standardized_direct_call_record = direct_call_raw_output_or_dict
                    standardized_direct_call_record['api_name'] = expected_tool_name
                    standardized_direct_call_record['input'] = expected_input_args
                else:
                    standardized_direct_call_record = {
                        'api_name': expected_tool_name,
                        'input': expected_input_args,
                        'output': direct_call_raw_output_or_dict,
                        'exception': None 
                    }
                outcome_validation_messages.append(f"    - Direct Call Output: {json.dumps(standardized_direct_call_record.get('output'))}")
                if standardized_direct_call_record.get('exception'):
                    outcome_validation_messages.append(f"    - Direct Call Exception: {standardized_direct_call_record.get('exception')}")

            except Exception as direct_call_e:
                standardized_direct_call_record = {
                    'api_name': expected_tool_name,
                    'input': expected_input_args,
                    'output': None,
                    'exception': str(direct_call_e)
                }
                outcome_validation_messages.append(f"    - Direct Call Raised Exception: {str(direct_call_e)}")
            
            groundtruth_record_for_check = {
                'api_name': expected_tool_name,
                'input': expected_input_args, 
                'output': expected_output_groundtruth, 
                'exception': expected_exception_groundtruth 
            }
            
            api_tool_instance = None
            try:
                api_tool_instance = tool_mgr.init_tool(expected_tool_name)
            except Exception as e:
                all_expected_outcomes_match = False
                msg = f"    - VALIDATION ERROR: Could not initialize tool '{expected_tool_name}' for check. Error: {e}"
                outcome_validation_messages.append(msg)
                continue 

            if not hasattr(api_tool_instance, 'check_api_call_correctness'):
                all_expected_outcomes_match = False
                msg = f"    - VALIDATION ERROR: Tool '{expected_tool_name}' is missing 'check_api_call_correctness' method."
                outcome_validation_messages.append(msg)
                continue 

            try:
                is_call_correct = api_tool_instance.check_api_call_correctness(
                    response=standardized_direct_call_record, 
                    groundtruth=groundtruth_record_for_check
                )
                if is_call_correct:
                    outcome_validation_messages.append(f"    - ✔️ OUTCOME VALIDATION: PASSES. Result of direct call matches expected.")
                else:
                    all_expected_outcomes_match = False
                    msg = (
                        f"    - ❌ OUTCOME VALIDATION: FAILS. Result of direct call does NOT match expected.\n"
                        f"        Groundtruth for check (from JSON):\n{json.dumps(groundtruth_record_for_check, indent=10)}\n"
                        f"        Actual from direct call (passed to check):\n{json.dumps(standardized_direct_call_record, indent=10)}"
                    )
                    outcome_validation_messages.append(msg)
            except Exception as e:
                all_expected_outcomes_match = False
                msg = (
                    f"    - ERROR during 'check_api_call_correctness' for '{expected_tool_name}'. Error: {str(e)}\n"
                    f"      Traceback: {traceback.format_exc()}\n"
                    f"      Groundtruth for check (from JSON):\n{json.dumps(groundtruth_record_for_check, indent=10)}\n"
                    f"      Actual from direct call (passed to check):\n{json.dumps(standardized_direct_call_record, indent=10)}"
                )
                outcome_validation_messages.append(msg)

    final_summary_message = agent_actions_report + "\n" + "\n".join(outcome_validation_messages)
    return all_expected_outcomes_match, final_summary_message

# --- Running the Graph with test cases ---
if __name__ == "__main__":
    print("\n--- Running LangGraph Validation (API Outcome Based) ---")
    if USE_DOMAIN_CONTEXT and DOMAIN_INFORMATION_CONTEXT and DOMAIN_INFORMATION_CONTEXT.strip():
        print(f"INFO: Using Domain Context from '{DOMAIN_CONTEXT_FILENAME}'")
    elif USE_DOMAIN_CONTEXT: 
        print(f"INFO: USE_DOMAIN_CONTEXT is True, but Domain Context from '{DOMAIN_CONTEXT_FILENAME}' is empty. Proceeding with minimal system message if any.")
    else: 
        print("INFO: Domain Context usage is disabled (USE_DOMAIN_CONTEXT=False).")

    level3_json_path = os.path.join(script_dir, "api_bank", "test_data", "level3.json")

    if not os.path.exists(level3_json_path):
        alt_path_1 = os.path.join(os.getcwd(), "api_bank", "test_data", "level3.json")
        alt_path_2 = os.path.join(os.path.dirname(script_dir), "api_bank", "test_data", "level3.json")

        if os.path.exists(alt_path_1):
            level3_json_path = alt_path_1
        elif os.path.exists(alt_path_2) and os.path.dirname(script_dir) != script_dir : 
             level3_json_path = alt_path_2
        else:
            print(f"ERROR: level3.json not found at initial path: {level3_json_path} or common alternative relative paths.")
            sys.exit(1)
    print(f"Using level3.json from: {level3_json_path}")

    try:
        with open(level3_json_path, 'r', encoding='utf-8') as f: 
            test_cases = json.load(f)
    except Exception as e:
        print(f"Error loading or parsing level3.json from {level3_json_path}: {e}")
        sys.exit(1)
        
    total_cases = len(test_cases)
    passed_cases = 0
    failed_cases = 0
    skipped_cases = 0

    # This check is now after the caching logic ensures tool_manager_instance is valid or exits
    print(f"ToolManager ready. Available APIs in manager: {len(tool_manager_instance.apis or [])}.")
    print(f"Langchain tools generated: {len(langchain_tools_list)}.")


    for i, test_case in enumerate(test_cases):
        case_number = i + 1
        print(f"\n" + "="*80)
        print(f"--- Test Case {case_number}/{total_cases}: {test_case.get('scene_description', 'N/A')} ---")
        print("="*80)

        user_query = test_case.get("initial_user_utterance")
        relevant_info = test_case.get("relevant_key_info_for_scenario_understanding")
        expected_apis_for_outcome_validation = test_case.get("apis", []) 

        if not user_query:
            print(f"🛑 Test Case {case_number} SKIPPED: missing 'initial_user_utterance'.")
            skipped_cases +=1
            continue

        print(f"🗣️ User Query (Original): '{user_query}'")

        full_user_query = user_query
        if relevant_info: 
            try:
                relevant_info_str = json.dumps(relevant_info, indent=2)
                full_user_query += f"\n\nHere is some relevant information for this request:\n{relevant_info_str}"
                print(f"ℹ️ Query Augmented with relevant_info:\n{relevant_info_str}")
            except TypeError:
                print(f"⚠️ Warning: relevant_info for test case {case_number} is not JSON serializable, appending as string.")
                full_user_query += f"\n\nHere is some relevant information for this request:\n{relevant_info}"


        print("\n📋 Expected API Outcomes to Validate (from level3.json):")
        if expected_apis_for_outcome_validation:
            for idx, api_call_spec in enumerate(expected_apis_for_outcome_validation):
                print(f"  {idx+1}. API Name: {api_call_spec.get('api_name', 'N/A')}")
                print(f"     Input   : {json.dumps(api_call_spec.get('input', {}))}")
                print(f"     Expected Output: {json.dumps(api_call_spec.get('output'))}")
                if api_call_spec.get('exception') is not None:
                    print(f"     Expected Exception: {json.dumps(api_call_spec.get('exception'))}")
        else:
            print("  (No specific API outcomes are expected to be validated for this test case)")
        print("-" * 30)

        actual_tool_actions_from_agent = [] # Renamed for clarity from previous user query context
        pending_tool_calls_for_agent = {}

        initial_messages_for_graph: TypingList[BaseMessage] = []
        if USE_DOMAIN_CONTEXT and DOMAIN_INFORMATION_CONTEXT and DOMAIN_INFORMATION_CONTEXT.strip():
            system_context_content = f"IMPORTANT SYSTEM DOMAIN INFORMATION FOR TOOL SELECTION AND USAGE:\n\n{DOMAIN_INFORMATION_CONTEXT}"
            system_context_message = SystemMessage(content=system_context_content)
            initial_messages_for_graph.append(system_context_message)
            print(f"System Context Prepended: {system_context_content[:150]}{'...' if len(system_context_content)>150 else ''}")

        initial_messages_for_graph.append(HumanMessage(content=full_user_query))
        
        # Reset ToolManager's DBs if it has such a method and state needs to be clean per test case
        if hasattr(tool_manager_instance, 'reset_dbs') and callable(tool_manager_instance.reset_dbs):
            print("INFO: Resetting ToolManager databases for new test case.")
            tool_manager_instance.reset_dbs()


        final_state_stream = app.with_config({"recursion_limit": 15}).stream({"messages": initial_messages_for_graph})

        print("\n💬 Agent Execution Log:")
        for event_idx, event_chunk in enumerate(final_state_stream):
            for node_name, output_from_node in event_chunk.items():
                print(f"  Event {event_idx}, Node '{node_name}':")
                if 'messages' in output_from_node and output_from_node['messages']:
                    for msg_idx, msg in enumerate(output_from_node['messages']):
                        print(f"    Msg {msg_idx} Type: {type(msg).__name__}")
                        if isinstance(msg, AIMessage):
                            print(f"      AI: {msg.content[:200]}{'...' if len(msg.content) > 200 else ''}")
                            if msg.tool_calls:
                                print(f"      Tool Calls by AI:")
                                for tc_idx, tc in enumerate(msg.tool_calls):
                                    print(f"        {tc_idx+1}. ID: {tc['id']}, Name: {tc['name']}, Args: {json.dumps(tc['args'])}")
                                    pending_tool_calls_for_agent[tc['id']] = {"name": tc['name'], "args": tc['args']}
                        elif isinstance(msg, ToolMessage):
                            print(f"      Tool ID: {msg.tool_call_id}, Name: {msg.name if hasattr(msg, 'name') else 'N/A'}")
                            raw_output_dict = msg.additional_kwargs.get("raw_output_dict", {})
                            # print(f"      Tool Output (raw_output_dict): {json.dumps(raw_output_dict, indent=2)}") # Can be verbose

                            if msg.tool_call_id in pending_tool_calls_for_agent:
                                tool_call_details = pending_tool_calls_for_agent.pop(msg.tool_call_id)
                                
                                if not isinstance(raw_output_dict, dict) or 'api_name' not in raw_output_dict:
                                    print(f"WARNING: raw_output_dict missing/malformed in ToolMessage for {tool_call_details.get('name')}. Reconstructing.")
                                    reconstructed_dict = {
                                        "api_name": tool_call_details.get("name"),
                                        "input": tool_call_details.get("args"),
                                        "output": None,
                                        "exception": "raw_output_dict missing/malformed"
                                    }
                                    try:
                                        parsed_content = json.loads(msg.content)
                                        if isinstance(parsed_content, dict) and 'api_name' in parsed_content:
                                            reconstructed_dict = parsed_content
                                        else:
                                            reconstructed_dict['output'] = parsed_content
                                    except (json.JSONDecodeError, TypeError):
                                         reconstructed_dict['output'] = msg.content # Store raw content if not JSON
                                    raw_output_dict = reconstructed_dict


                                actual_tool_actions_from_agent.append({ # Corrected variable name usage here
                                    "name": tool_call_details["name"], 
                                    "args": tool_call_details["args"], 
                                    "output_dict": raw_output_dict     
                                })
                            else:
                                print(f"Warning: Received ToolMessage for unknown tool_call_id: {msg.tool_call_id}")
                        # Other message types can be logged as needed

        is_pass, validation_summary_message = validate_test_case(
            actual_tool_actions_from_agent,  # Corrected variable name usage here
            expected_apis_for_outcome_validation, 
            tool_manager_instance
        )

        print("\n--- Validation Summary ---")
        print(validation_summary_message) 

        if is_pass:
            print(f"\n✅ Test Case {case_number}: PASS (Based on expected API outcome validation)")
            passed_cases += 1
        else:
            print(f"\n❌ Test Case {case_number}: FAIL (Based on expected API outcome validation)")
            failed_cases += 1

    print("\n" + "="*80)
    print("--- Overall Summary ---")
    print(f"Total Cases: {total_cases}")
    print(f"✅ Passed   : {passed_cases}")
    print(f"❌ Failed   : {failed_cases}")
    if skipped_cases > 0:
        print(f"🛑 Skipped  : {skipped_cases}")
    print("="*80)
    print("--- Evaluation Complete ---")