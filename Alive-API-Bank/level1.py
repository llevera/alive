import sys
import os
import json
from typing import TypedDict, Optional, List as TypingList, Sequence, Annotated, Union, Dict, Type, Any, Set
import argparse # Added for command-line arguments

import operator

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from pydantic import BaseModel, Field
from langchain.tools import Tool
from langchain_openai import AzureChatOpenAI

# --- Path Setup (script_dir is needed early for domain context file) ---
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- Load Domain Information Context from File ---
DOMAIN_CONTEXT_FILENAME = "domain_context.txt"
DOMAIN_INFORMATION_CONTEXT = "" # Default to empty

test_files = ["ScheduleConflictingMeeting.jsonl"]
#test_files = ["QueryMeeting-level-1-1.jsonl"]
use_domain_context = True
YOUR_DEPLOYMENT_NAME = "gpt-4o" # Change this to your deployment name


try:
    domain_context_filepath = os.path.join(script_dir, DOMAIN_CONTEXT_FILENAME)
    with open(domain_context_filepath, 'r', encoding='utf-8') as f:
        DOMAIN_INFORMATION_CONTEXT = f.read()
    if not DOMAIN_INFORMATION_CONTEXT.strip():
        print(f"WARNING: Domain context file '{domain_context_filepath}' is empty.")
        # Allow proceeding with empty context, the SystemMessage will just have the prefix.
except FileNotFoundError:
    print(f"ERROR: Domain context file '{DOMAIN_CONTEXT_FILENAME}' not found at {domain_context_filepath}.")
    print("Please create this file with the domain model text. The script will exit as this context is crucial for guiding the LLM.")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Failed to load domain context from '{domain_context_filepath}': {e}")
    print("The script will exit due to this error.")
    sys.exit(1)

# --- Continue with other path setups and imports ---
api_bank_package_dir = os.path.join(script_dir, "api_bank")
absolute_apis_path_for_toolmanager = os.path.join(api_bank_package_dir, "apis")

if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from api_bank.tool_manager import ToolManager # type: ignore
except ImportError as e:
    print(f"ERROR: Failed to import ToolManager from api_bank: {e}")
    sys.exit(1)

# --- ToolManager Instance ---
tool_manager_instance = None
try:
    if not os.path.isdir(absolute_apis_path_for_toolmanager):
        print(f"ERROR: 'apis' directory not found at {absolute_apis_path_for_toolmanager}.")
        sys.exit(1)
    tool_manager_instance = ToolManager(apis_dir=absolute_apis_path_for_toolmanager)
except Exception as e:
    print(f"ERROR: Failed to initialize ToolManager: {e}")
    sys.exit(1)


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
    if manager.apis is None:
        print("Warning: ToolManager has no APIs loaded.")
        return []

    for api_info in manager.apis:
        tool_name = api_info['name']
        args_schema_to_use = create_pydantic_model_for_tool(tool_name, api_info['input_parameters'])

        def execute_api_call(captured_tool_name: str, **kwargs_for_api: dict) -> dict:
            try:
                tool_call_result_payload = manager.api_call(tool_name=captured_tool_name, **kwargs_for_api)

                if isinstance(tool_call_result_payload, dict) and \
                   all(k in tool_call_result_payload for k in ['api_name', 'input', 'output', 'exception']):
                    return tool_call_result_payload
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
                    'exception': f"Wrapper exception: {str(e)}"
                }

        langchain_tools.append(Tool(
            name=tool_name,
            description=api_info['description'],
            func=lambda tn=tool_name, **kwargs: execute_api_call(tn, **kwargs),
            args_schema=args_schema_to_use
        ))
    return langchain_tools

langchain_tools_list = get_langchain_tools_from_manager(tool_manager_instance)

class AgentState(TypedDict): messages: Annotated[Sequence[BaseMessage], operator.add]

required_env_vars = {"AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY")}
if not required_env_vars["AZURE_OPENAI_API_KEY"]:
    print("ERROR: AZURE_OPENAI_API_KEY missing.")
    sys.exit(1)

try:
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://andrewslearningsweden.openai.azure.com/"),
        api_key=required_env_vars["AZURE_OPENAI_API_KEY"],
        api_version=os.getenv("OPENAI_API_VERSION", "2024-02-15-preview"),
        deployment_name=YOUR_DEPLOYMENT_NAME,
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(langchain_tools_list)
except Exception as e:
    print(f"ERROR: Error during AzureChatOpenAI init: {e}")
    sys.exit(1)

def agent_node(state: AgentState) -> AgentState:
    ai_response_message = llm_with_tools.invoke(state['messages'])
    return {"messages": [ai_response_message]}

def tool_node(state: AgentState) -> AgentState:
    last_message = state['messages'][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"messages": []}

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
                    'exception': f"Error directly invoking tool '{tool_call['name']}': {str(e)}"
                }
        else:
            raw_tool_output_dict_from_execute = {
                "api_name": tool_call['name'],
                "input": tool_call['args'],
                "output": None,
                "exception": f"Tool '{tool_call['name']}' not found."
            }
        
        tool_messages.append(ToolMessage(
            content=json.dumps(raw_tool_output_dict_from_execute),
            tool_call_id=tool_call['id'],
            additional_kwargs={"raw_output_dict": raw_tool_output_dict_from_execute}
        ))
    return {"messages": tool_messages}


def should_continue(state: AgentState) -> str:
    last_message = state['messages'][-1]
    if isinstance(last_message, AIMessage):
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "invoke_tools"
        return END
    return END

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"invoke_tools": "tools", END: END})
workflow.add_edge("tools", "agent")
app = workflow.compile()

class Sample:
    def __init__(self, chat_history: TypingList[Dict], apis: Set[str], ground_truth: Dict):
        self.chat_history = chat_history
        self.apis = apis
        self.ground_truth = ground_truth
        self.source_file: Optional[str] = None

    @classmethod
    def from_chat_history(cls, chat_history_list_of_dicts: TypingList[Dict]) -> TypingList['Sample']:
        samples = []
        for i, item in enumerate(chat_history_list_of_dicts):
            if item.get('role') == 'API':
                prediction_context_history = chat_history_list_of_dicts[:i]
                apis_in_context = set()
                for history_item in prediction_context_history:
                    if history_item.get('role') == 'API' and 'api_name' in history_item:
                        apis_in_context.add(history_item['api_name'])
                sample = cls(
                    chat_history=prediction_context_history,
                    apis=apis_in_context,
                    ground_truth=item
                )
                samples.append(sample)
        return samples

def validate_tool_sequence(actual_actions: TypingList[Dict], expected_apis: TypingList[Dict], tool_mgr: ToolManager) -> tuple[bool, str]:
    if not expected_apis:
        return False, "Validation Error: No expected API call provided for the sample."

    expected_api_spec = expected_apis[0]
    if not actual_actions:
        return False, (
            f"API Call Mismatch: Expected 1 API call ('{expected_api_spec.get('api_name', 'N/A')}'), but agent made 0 calls.\n"
            f"  Expected Input: {json.dumps(expected_api_spec.get('input', {}), indent=2)}"
        )
    
    actual_action_details = actual_actions[0]
    actual_tool_name = actual_action_details.get('name')
    expected_tool_name = expected_api_spec.get('api_name')

    if actual_tool_name != expected_tool_name:
        return False, (
            f"API Name Mismatch.\n"
            f"  Expected API Name: '{expected_tool_name}'\n"
            f"  Actual API Name  : '{actual_tool_name}'\n"
            f"  Actual Args    : {json.dumps(actual_action_details.get('args', {}), indent=2)}"
        )
    actual_call_result_dict = actual_action_details.get('output_dict', {})

    actual_payload_main_output = actual_call_result_dict.get('output')
    actual_payload_main_exception = actual_call_result_dict.get('exception')

    final_output_for_check = actual_payload_main_output
    final_exception_for_check = actual_payload_main_exception

    if isinstance(actual_payload_main_output, dict) and \
    'output' in actual_payload_main_output and \
    'exception' in actual_payload_main_output and \
    actual_payload_main_exception is None:
        final_output_for_check = actual_payload_main_output.get('output')
        final_exception_for_check = actual_payload_main_output.get('exception')

    standardized_actual_record = {
        'api_name': actual_call_result_dict.get('api_name', actual_tool_name),
        'input': actual_call_result_dict.get('input', actual_action_details.get('args', {})),
        'output': final_output_for_check,
        'exception': final_exception_for_check
    }
    
    expected_call_record_for_check = {
        'api_name': expected_tool_name,
        'input': expected_api_spec.get('input', {}),
        'output': expected_api_spec.get('output'),
        'exception': expected_api_spec.get('exception')
    }
    api_tool_instance = None
    try:
        api_tool_instance = tool_mgr.init_tool(actual_tool_name)
    except Exception as e:
        return False, f"Validation Error for '{actual_tool_name}': Could not init tool. Error: {e}"

    if not hasattr(api_tool_instance, 'check_api_call_correctness'):
        return False, f"Validation Error for '{actual_tool_name}': Tool missing 'check_api_call_correctness'."

    try:
        is_call_correct = api_tool_instance.check_api_call_correctness(
            response=standardized_actual_record,
            groundtruth=expected_call_record_for_check
        )
        if not is_call_correct:
            reason_details = []
            if standardized_actual_record.get('input') != expected_call_record_for_check.get('input'):
                reason_details.append(
                    f"  Input Mismatch:\n"
                    f"    Expected: {json.dumps(expected_call_record_for_check.get('input'), indent=4)}\n"
                    f"    Actual  : {json.dumps(standardized_actual_record.get('input'), indent=4)}"
                )
            if standardized_actual_record.get('output') != expected_call_record_for_check.get('output'):
                 reason_details.append(
                    f"  Output Mismatch:\n"
                    f"    Expected: {json.dumps(expected_call_record_for_check.get('output'), indent=4)}\n"
                    f"    Actual  : {json.dumps(standardized_actual_record.get('output'), indent=4)}"
                )
            if standardized_actual_record.get('exception') != expected_call_record_for_check.get('exception'):
                 reason_details.append(
                    f"  Exception Mismatch:\n"
                    f"    Expected: {json.dumps(expected_call_record_for_check.get('exception'), indent=4)}\n"
                    f"    Actual  : {json.dumps(standardized_actual_record.get('exception'), indent=4)}"
                )
            
            full_reason = (
                f"Validation Failed by tool '{actual_tool_name}' via 'check_api_call_correctness'.\n"
                f"  Details:\n" + "\n".join(reason_details) + "\n"
                f"  Full EXPECTED Record (for check_api_call_correctness):\n"
                f"{json.dumps(expected_call_record_for_check, indent=4)}\n"
                f"  Full ACTUAL Record (passed to check_api_call_correctness):\n"
                f"{json.dumps(standardized_actual_record, indent=4)}"
            )
            return False, full_reason
            
    except Exception as e:
        import traceback
        return False, (
            f"Error during 'check_api_call_correctness' for '{actual_tool_name}'. Error: {str(e)}\n"
            f"  Traceback: {traceback.format_exc()}\n"
            f"  EXPECTED (Groundtruth for check_api_call_correctness):\n"
            f"{json.dumps(expected_call_record_for_check, indent=4)}\n"
            f"  ACTUAL (Passed to check_api_call_correctness):\n"
            f"{json.dumps(standardized_actual_record, indent=4)}"
        )

    return True, f"API call '{actual_tool_name}' is correct (validated by tool-specific method)."

if __name__ == "__main__":



    # --- Command Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run LangGraph Validation Tests for Level-1 Given Description Cases.")
    parser.add_argument(
        '--test_files',
        nargs='+',
        default=[], # Default to an empty list, meaning run all if not specified
        help="Optional: Specific .jsonl test file names (e.g., file1.jsonl file2.jsonl) to run. If not provided, all .jsonl files in the data directory will be attempted."
    )
    cli_args = parser.parse_args()

    print("\n--- Running LangGraph Validation (Level-1 Given Description Cases) ---")

    level1_data_path = os.path.join(os.path.dirname(api_bank_package_dir), "lv1-lv2-samples", "level-1-given-desc")

    if not os.path.isdir(level1_data_path):
        alt_path = os.path.join(api_bank_package_dir, "lv1-lv2-samples", "level-1-given-desc")
        if os.path.isdir(alt_path):
            level1_data_path = alt_path
        else:
            print(f"ERROR: level-1-given-desc data directory not found at primary path: {level1_data_path} or alt path: {alt_path}")
            sys.exit(1)
    print(f"Using level-1-given-desc data from: {level1_data_path}")

    # --- Determine which .jsonl files to process based on CLI args ---
    all_jsonl_in_dir = [f for f in os.listdir(level1_data_path) if f.endswith('.jsonl')]
    
    files_to_process_from_cli: TypingList[str] = []

    #test_files = cli_args.test_files


    if test_files: # If specific files are listed on command line
        requested_files_set = set(test_files)
        # Ensure specified files exist in the directory
        for f_name in all_jsonl_in_dir:
            if f_name in requested_files_set:
                files_to_process_from_cli.append(f_name)
                requested_files_set.remove(f_name) # Mark as found
        
        if not files_to_process_from_cli:
            print(f"ERROR: None of the specified test files were found or are valid .jsonl files in {level1_data_path}: {test_files}")
            sys.exit(1)
        
        if requested_files_set: # Some specified files were not found
            print(f"WARNING: The following specified test files were not found in {level1_data_path} and will be skipped: {list(requested_files_set)}")
        
        print(f"INFO: Will process specified test files: {files_to_process_from_cli}")
    else: # No specific files listed, process all found .jsonl files
        files_to_process_from_cli = all_jsonl_in_dir
        if not files_to_process_from_cli:
            print(f"ERROR: No .jsonl files found in {level1_data_path}. Nothing to test.")
            sys.exit(1)
        print(f"INFO: No specific test files provided. Will process all {len(files_to_process_from_cli)} .jsonl files found.")
    # --- End file selection logic ---

    all_test_samples: TypingList[Sample] = []
    for file_name in files_to_process_from_cli: # Use the filtered list of files
        full_file_path = os.path.join(level1_data_path, file_name)
        current_file_history: TypingList[Dict] = []
        with open(full_file_path, 'r') as f_jsonl:
            for line_idx, line in enumerate(f_jsonl):
                try:
                    current_file_history.append(json.loads(line))
                except json.JSONDecodeError as je:
                    print(f"Warning: Skipping malformed JSON line {line_idx+1} in {file_name}: {je}")
                    continue
        
        samples_from_file = Sample.from_chat_history(current_file_history)
        for s in samples_from_file:
            if s.ground_truth and s.ground_truth.get('role') == 'API':
                s.source_file = file_name
                all_test_samples.append(s)

    total_cases = len(all_test_samples)
    passed_cases = 0
    failed_cases = 0
    
    TEST_LIMIT = None
    samples_to_run = all_test_samples
    if TEST_LIMIT is not None and TEST_LIMIT < total_cases:
        print(f"INFO: Limiting tests to the first {TEST_LIMIT} samples.")
        samples_to_run = all_test_samples[:TEST_LIMIT]
    
    actual_cases_to_run = len(samples_to_run)

    if not tool_manager_instance:
        print("ERROR: ToolManager instance is not available.")
        sys.exit(1)
    
    print(f"ToolManager initialized. Loaded tools: {len(tool_manager_instance.list_all_apis() or [])} tools.")
    print(f"Total test samples to process: {actual_cases_to_run} (out of {total_cases} total found from selected files)")


    for i, test_sample in enumerate(samples_to_run):
        case_number = i + 1
        
        details_for_failure_output: TypingList[str] = []

        expected_gt_api_name = test_sample.ground_truth.get('api_name', 'N/A')
        
        details_for_failure_output.append(f"\n" + "="*80)
        details_for_failure_output.append(f"--- Test Case {case_number}/{actual_cases_to_run} (File: {test_sample.source_file}, Expected API: {expected_gt_api_name}) ---")
        details_for_failure_output.append("="*80)

        initial_messages_for_graph: TypingList[BaseMessage] = []
        dialogue_history_lines = ["Dialogue History (context for agent):"]

        # Add Domain Context as an initial SystemMessage (loaded from file)
        if use_domain_context and DOMAIN_INFORMATION_CONTEXT and DOMAIN_INFORMATION_CONTEXT.strip():
            system_context_content = f"IMPORTANT SYSTEM DOMAIN INFORMATION FOR TOOL SELECTION AND USAGE:\n\n{DOMAIN_INFORMATION_CONTEXT}"
            system_context_message = SystemMessage(content=system_context_content)
            initial_messages_for_graph.append(system_context_message)
            dialogue_history_lines.append(f"  System Instruction: {system_context_content[:150]}{'...' if len(system_context_content)>150 else ''} (Full context provided to LLM)")
        else:
            dialogue_history_lines.append("  System Instruction: (No domain context provided or file was empty)")


        for turn_idx, turn_data in enumerate(test_sample.chat_history):
            role = turn_data.get('role')
            text_content = turn_data.get('text', '')
            if role == 'User':
                msg = HumanMessage(content=text_content)
                dialogue_history_lines.append(f"  User: {msg.content}")
                initial_messages_for_graph.append(msg)
            elif role == 'AI':
                msg = AIMessage(content=text_content)
                dialogue_history_lines.append(f"  AI  : {msg.content}")
                initial_messages_for_graph.append(msg)
            elif role == 'API':
                api_name = turn_data.get('api_name', 'UnknownAPI')
                param_dict = turn_data.get('param_dict', {})
                api_result = turn_data.get('result', {})
                synthetic_tool_call_id = f"history_tc_{case_number}_{turn_idx}"
                
                ai_call_msg = AIMessage(content="", tool_calls=[{"id": synthetic_tool_call_id, "name": api_name, "args": param_dict}])
                initial_messages_for_graph.append(ai_call_msg)
                dialogue_history_lines.append(f"  AI  : [Tool Call: {api_name}({json.dumps(param_dict)}) id: {synthetic_tool_call_id}]")

                tool_response_content_str = json.dumps(api_result) if isinstance(api_result, (dict, list)) else str(api_result)
                tool_res_msg = ToolMessage(
                    tool_call_id=synthetic_tool_call_id, 
                    content=tool_response_content_str, 
                    additional_kwargs={"raw_output_dict": api_result}
                )
                initial_messages_for_graph.append(tool_res_msg)
                dialogue_history_lines.append(f"  Tool: [Result for {synthetic_tool_call_id}] {tool_response_content_str[:200]}{'...' if len(tool_response_content_str)>200 else ''}")
        details_for_failure_output.extend(dialogue_history_lines)

        expected_gt_input = test_sample.ground_truth.get('param_dict', {})
        ground_truth_full_result_record = test_sample.ground_truth.get('result', {})
        expected_output_payload_for_validation = None
        expected_exception_for_validation = None
        
        if isinstance(ground_truth_full_result_record, dict):
            expected_output_payload_for_validation = ground_truth_full_result_record.get('output')
            expected_exception_for_validation = ground_truth_full_result_record.get('exception')
        else:
            expected_output_payload_for_validation = ground_truth_full_result_record
            
        expected_api_call_lines = [f"\n📋 Expected API Call (Ground Truth for case {case_number}):"]
        expected_api_call_lines.append(f"  API Name: {expected_gt_api_name}")
        expected_api_call_lines.append(f"  Input   : {json.dumps(expected_gt_input, indent=2)}")
        expected_api_call_lines.append(f"  Expected Output Payload (for validation): {json.dumps(expected_output_payload_for_validation, indent=2)}")
        if expected_exception_for_validation is not None:
            expected_api_call_lines.append(f"  Expected Exception (for validation): {json.dumps(expected_exception_for_validation, indent=2)}")
        expected_api_call_lines.append("-" * 30)
        details_for_failure_output.extend(expected_api_call_lines)
            
        if not initial_messages_for_graph or \
           (len(initial_messages_for_graph) == 1 and isinstance(initial_messages_for_graph[0], SystemMessage)) or \
           all(isinstance(m, SystemMessage) for m in initial_messages_for_graph): # Check if only system message(s)
            pseudo_query_text = f"Based on the provided system domain information, please perform the action: {expected_gt_api_name} with parameters {json.dumps(expected_gt_input)}."
            details_for_failure_output.append(f"WARNING: Test sample {case_number} has no user/ai/tool history beyond the system context. Using pseudo-query: {pseudo_query_text}")
            initial_messages_for_graph.append(HumanMessage(content=pseudo_query_text))


        actual_tool_actions = []
        pending_tool_calls: Dict[str, Dict] = {}
        agent_trace_lines = ["\n⚙️ Agent Execution Trace:"]
        try:
            final_state_stream = app.with_config({"recursion_limit": 10}).stream({"messages": initial_messages_for_graph})
            for event_idx, event_chunk in enumerate(final_state_stream):
                event_key = list(event_chunk.keys())[0]
                if 'messages' in event_chunk[event_key]:
                    for msg_idx, msg_obj in enumerate(event_chunk[event_key]['messages']):
                        if isinstance(msg_obj, AIMessage):
                            agent_trace_lines.append(f"  AI (Agent Node): Content='{msg_obj.content}'")
                            if msg_obj.tool_calls:
                                for tc in msg_obj.tool_calls:
                                    agent_trace_lines.append(f"    -> Calls Tool: ID={tc['id']}, Name={tc['name']}, Args={json.dumps(tc['args'])}")
                                    pending_tool_calls[tc['id']] = {"name": tc['name'], "args": tc['args']}
                        elif isinstance(msg_obj, ToolMessage):
                            agent_trace_lines.append(f"  Tool (Tools Node): ID={msg_obj.tool_call_id}, RawContent='{msg_obj.content[:100]}{'...' if len(msg_obj.content) > 100 else ''}'")
                                
                            if msg_obj.tool_call_id in pending_tool_calls:
                                tool_call_invocation_details = pending_tool_calls.pop(msg_obj.tool_call_id)
                                raw_output_dict_from_tool_message = msg_obj.additional_kwargs.get("raw_output_dict")
                                if raw_output_dict_from_tool_message is None:
                                    try:
                                        parsed_content = json.loads(msg_obj.content)
                                        if isinstance(parsed_content, dict) and 'api_name' in parsed_content:
                                           raw_output_dict_from_tool_message = parsed_content
                                        else:
                                           raw_output_dict_from_tool_message = {
                                                "api_name": tool_call_invocation_details.get("name"), "input": tool_call_invocation_details.get("args"),
                                                "output": parsed_content, "exception": None
                                            }
                                    except (json.JSONDecodeError, TypeError):
                                        raw_output_dict_from_tool_message = {
                                            "api_name": tool_call_invocation_details.get("name"), "input": tool_call_invocation_details.get("args"),
                                            "output": None, "exception": f"Failed to parse tool output string: {msg_obj.content[:100]}"
                                        }
                                actual_tool_actions.append({
                                    "name": tool_call_invocation_details["name"], "args": tool_call_invocation_details["args"],
                                    "output_dict": raw_output_dict_from_tool_message
                                })
        except Exception as ex_graph:
            agent_trace_lines.append(f"ERROR during graph execution for case {case_number}: {ex_graph}")
        details_for_failure_output.extend(agent_trace_lines)

        actual_api_calls_lines = ["\n🛠️ Actual API Calls Made by Agent:"]
        if actual_tool_actions:
            expected_api_spec_for_validation_list = [{
                'api_name': expected_gt_api_name, 'input': expected_gt_input,
                'output': expected_output_payload_for_validation, 'exception': expected_exception_for_validation
            }]
            if len(actual_tool_actions) > 1 and len(expected_api_spec_for_validation_list) == 1 :
                 details_for_failure_output.append(f"WARNING: Agent made {len(actual_tool_actions)} calls, but only 1 was expected. Validating the first call only.")

            for idx, action in enumerate(actual_tool_actions):
                actual_api_calls_lines.append(f"  {idx+1}. API Name: {action.get('name', 'N/A')}")
                actual_api_calls_lines.append(f"     Args    : {json.dumps(action.get('args', {}), indent=2)}")
                actual_api_calls_lines.append(f"     Result (output_dict): {json.dumps(action.get('output_dict', {}), indent=2)}")
        else:
            actual_api_calls_lines.append("  (No API calls were made by the agent)")
        actual_api_calls_lines.append("-" * 30)
        details_for_failure_output.extend(actual_api_calls_lines)
        
        expected_api_spec_for_validation = [{
            'api_name': expected_gt_api_name, 'input': expected_gt_input,
            'output': expected_output_payload_for_validation, 'exception': expected_exception_for_validation
        }]

        is_correct, message = validate_tool_sequence(actual_tool_actions, expected_api_spec_for_validation, tool_manager_instance)
        
        if is_correct:
            passed_cases += 1
            print(f"✅ TC {case_number}/{actual_cases_to_run} ({test_sample.source_file} - {expected_gt_api_name}): PASS")
        else:
            failed_cases += 1
            for line in details_for_failure_output:
                print(line)
            print("\n--- Validation ---")
            print(f"❌ TC {case_number}/{actual_cases_to_run} ({test_sample.source_file} - {expected_gt_api_name}): FAIL")
            print(f"   Reason:\n{message}")

    print("\n" + "="*80)
    print("--- Overall Summary (Level-1 Given Description) ---")
    print(f"Total Test Samples Found (from selected files): {total_cases}")
    print(f"Attempted to Run (after TEST_LIMIT if any): {actual_cases_to_run}")
    print(f"✅ Passed   : {passed_cases}")
    print(f"❌ Failed   : {failed_cases}")
    print("="*80)
    print("--- Evaluation Complete ---")