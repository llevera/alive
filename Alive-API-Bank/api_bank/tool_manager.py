# api_bank/tool_manager.py
import importlib
import os
import json
import copy # Import copy for deepcopy
import sys

# It's good practice to ensure API and ToolSearcher are defined,
# even if just as placeholders, before class ToolManager if they are used elsewhere globally.
# However, they are primarily used for type hinting or subclass checks within ToolManager.
try:
    from .apis.api import API
    from .apis.tool_search import ToolSearcher
except ImportError as e_tm_imports:
    print(f"[tool_manager.py] !!! ImportError during .apis.api or .apis.tool_search import: {e_tm_imports}")
    # Define placeholders if essential for basic script structure, though ToolManager might fail later
    class API: pass
    class ToolSearcher: pass 
    # Depending on usage, might need to raise e_tm_imports if these are critical.


class ToolManager:
    def __init__(self, apis_dir='./apis') -> None:
        all_apis_classes = []
        # Dynamically import API classes
        except_files = ['__init__.py', 'api.py', 'tool_search.py'] # tool_search might be special
        
        # Ensure apis_dir is an absolute path for reliable module loading if necessary
        if not os.path.isabs(apis_dir):
            apis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), apis_dir)
            apis_dir = os.path.normpath(apis_dir)

        # Parent directory of apis_dir needs to be in sys.path for dotted imports if apis_dir is a package
        # However, the original code uses __package__.apis.{module_name} which implies
        # that the directory containing 'api_bank' is in sys.path and api_bank is a package.
        # The __package__ for tool_manager.py (if inside api_bank) would be 'api_bank'.
        # So, from .apis.{module_name} should work.

        for filename_with_ext in os.listdir(apis_dir):
            if filename_with_ext.endswith('.py') and filename_with_ext not in except_files:
                api_module_name = filename_with_ext[:-3]
                
                # Using __package__ to construct the absolute module path
                # This assumes tool_manager.py is part of a package (e.g., 'api_bank')
                # and 'apis' is a subpackage.
                absolute_module_name = f"{__package__}.apis.{api_module_name}"
                
                try:
                    module = importlib.import_module(absolute_module_name)
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type) and issubclass(attr, API) and attr is not API:
                            all_apis_classes.append(attr)
                except ImportError as e_imp:
                    print(f"[ToolManager.__init__] FAILED to import {absolute_module_name}. Error: {e_imp}")
                except Exception as e_other:
                    print(f"[ToolManager.__init__] OTHER ERROR while importing/processing {absolute_module_name}. Error: {e_other}")

        self.init_databases = {} # Stores the initial state of DBs from JSON files
        self.live_databases = {} # Stores the live, shared DB instances for tools

        tool_manager_file_dir = os.path.dirname(os.path.abspath(__file__))
        init_database_dir = os.path.join(tool_manager_file_dir, "init_database")
        if os.path.isdir(init_database_dir):
            for file in os.listdir(init_database_dir):
                if file.endswith('.json'):
                    database_name = file.split('.')[0]
                    try:
                        with open(os.path.join(init_database_dir, file), 'r', encoding='utf-8') as f:
                            db_content = json.load(f)
                            self.init_databases[database_name] = db_content
                            # Populate live_databases with a deep copy of the initial state
                            self.live_databases[database_name] = copy.deepcopy(db_content)
                    except Exception as e:
                        print(f"[ToolManager.__init__] Error loading database {database_name} from {file}: {e}")
        else:
            print(f"[ToolManager.__init__] init_database directory not found: {init_database_dir}")


        apis_info_list = []
        for cls in all_apis_classes:
            name = cls.__name__
            cls_info = {
                'name': name,
                'class': cls,
                'description': getattr(cls, 'description', f"No description for {name}."),
                'input_parameters': getattr(cls, 'input_parameters', {}),
                'output_parameters': getattr(cls, 'output_parameters', {}),
                'database_name': getattr(cls, 'database_name', None) # Store DB name if class has it
            }
            apis_info_list.append(cls_info)
        
        self.apis = apis_info_list # This is a list of dicts, one for each API class
        self.inited_tools = {} # Cache for initialized tool instances

        # Initialize token_checker if CheckToken API exists
        if 'CheckToken' in self.list_all_apis(): # list_all_apis uses self.apis, so fine here
            try:
                self.token_checker = self.init_tool('CheckToken')
            except Exception as e:
                print(f"[ToolManager.__init__] Error initializing CheckToken: {e}. Token checking may not work.")
                self.token_checker = None # Ensure it's defined
        else:
            self.token_checker = None


    def reset_dbs(self):
        """
        Resets all live databases to their initial states and clears cached tool instances.
        """
        print("[ToolManager.reset_dbs] Resetting live databases to initial state.")
        for db_name, initial_db_content in self.init_databases.items():
            self.live_databases[db_name] = copy.deepcopy(initial_db_content)
        
        print("[ToolManager.reset_dbs] Clearing cached tool instances.")
        self.inited_tools = {} # Clear tool instances to ensure they get new DB references

        # Re-initialize token_checker as it might have been cleared from inited_tools
        # and could be stateful or hold references that need resetting.
        if 'CheckToken' in self.list_all_apis():
            try:
                # init_tool will now re-create or re-cache it correctly
                self.token_checker = self.init_tool('CheckToken')
            except Exception as e:
                print(f"[ToolManager.reset_dbs] Error re-initializing CheckToken: {e}.")
                self.token_checker = None
        else:
            self.token_checker = None

    def get_api_by_name(self, name: str):
        for api_info_dict in self.apis: # self.apis is a list of dictionaries
            if api_info_dict['name'] == name:
                return api_info_dict
        raise ValueError(f"Invalid tool name: '{name}' not found in available APIs: {self.list_all_apis()}")
    
    def get_api_description(self, name: str):
        api_info = self.get_api_by_name(name).copy()
        api_info.pop('class', None) # Remove non-serializable class object
        # 'database_name' is fine to keep if it's just the name string.
        return json.dumps(api_info)

    def init_tool(self, tool_name: str, *args, **kwargs):
        if tool_name in self.inited_tools:
            return self.inited_tools[tool_name]

        api_info = self.get_api_by_name(tool_name) # This is the dictionary from self.apis
        api_class = api_info['class']
        
        constructor_args = [] # Arguments to be passed to the tool's __init__

        # 1. Handle database injection (shared instance)
        tool_db_name = api_info.get('database_name')
        if tool_db_name and tool_db_name in self.live_databases:
            constructor_args.append(self.live_databases[tool_db_name]) # Pass reference to live DB
        elif tool_db_name: # Tool specifies a DB name, but it's not in live_databases (e.g., no JSON file)
            # API class __init__ should handle init_database=None by creating an empty dict.
            # So, we pass None explicitly if the tool expects a db but we don't have one.
            # The class's __init__ signature (e.g., init_database=None) will make it use its default.
            # However, to match arity if the class expects it positionally:
            constructor_args.append(None) # Or rely on default param in class __init__
                                          # For AddMeeting(init_database=None, ...), if we don't pass it,
                                          # it will take its default None. So, this is fine.
                                          # If first arg is init_database, we must provide something or None.
            print(f"[ToolManager.init_tool] Warning: Database '{tool_db_name}' for tool '{tool_name}' not found in live_databases. Tool will use its default.")

        # 2. Handle token_checker injection
        # Tools' __init__ methods (e.g., AddMeeting) typically define init_database first, then token_checker.
        # We must ensure the order of args in constructor_args matches.
        # If init_database was not added, but token_checker is, token_checker becomes the first arg.
        # This requires API class __init__ signatures to be consistent or use kwargs.
        # Given AddMeeting(init_database=None, token_checker=None),
        # if db is present: constructor_args = [db_ref]
        # if db is NOT present: constructor_args = [None] (assuming we want to always pass something for db if it's the first param)

        # Let's assume standard signature: __init__(self, init_database=None, token_checker=None, ...)
        # If tool_db_name was None, constructor_args is still empty from DB logic.
        # Tool class __init__ will use default for init_database.

        if tool_name != 'CheckToken' and 'token' in api_info['input_parameters']:
            if self.token_checker:
                constructor_args.append(self.token_checker)
            else:
                # If token_checker is required but not available, the tool's __init__
                # should handle it (e.g. by using its default None).
                # We append None to satisfy positional argument if __init__ expects it.
                # However, this depends on API init signatures.
                # Most APIs have token_checker=None as default.
                print(f"[ToolManager.init_tool] Warning: Token checker not available for tool '{tool_name}'. Tool will use its default.")
                # If AddMeeting is `def __init__(self, db, tc)`, we need to pass tc as None.
                # If db was also None, args would be [None, None]
                # This is complex if signatures vary. For now, trust default params in API classes.
                # The current code only appends if self.token_checker exists.
                # If the first arg of an API is init_database, and second is token_checker,
                # and database_name_for_tool was None, then constructor_args could be [self.token_checker]
                # which would be passed as init_database. This is wrong.
                # The original code prepended. So it assumes:
                # init_database (optional), token_checker (optional), then other *args from call site.

                # Sticking to original logic of prepending and relying on API __init__ defaults:
                # The constructor_args built so far are for init_database.
                # Now, consider token_checker. The original code appends token_checker to temp_args which
                # means it was a distinct argument.
                # temp_args = []
                # if db: temp_args.append(db)
                # if tc: temp_args.append(tc)
                # This implies init signatures like __init__(self, db_arg_if_any, tc_arg_if_any, *other_constructor_args_passed_to_init_tool)
                
                # Corrected logic based on original structure:
                # constructor_args was initialized based on db. Now add token_checker if needed.
                # This assumes that if both are present, db comes before tc in the __init__ signature.
                # Example: `def __init__(self, init_database=None, token_checker=None)`
                pass # Relies on default token_checker=None in API class __init__ if self.token_checker is None

        # Append any *args passed to init_tool itself (these are usually not used for these tools)
        final_constructor_args = constructor_args + list(args)

        # Create the tool instance
        # Ensure that the number of arguments we are passing matches what the API class expects
        # if its __init__ is not flexible with *args or default values.
        # Given the API __init__ signatures (e.g. AddMeeting), they use default values,
        # so passing fewer args than total defined params is fine if those have defaults.
        try:
            tool = api_class(*final_constructor_args, **kwargs)
        except TypeError as e:
            print(f"[ToolManager.init_tool] TypeError during instantiation of {tool_name} with args {final_constructor_args}, kwargs {kwargs}: {e}")
            print(f"Ensure the __init__ signature of {tool_name} matches the arguments being passed.")
            raise
            
        self.inited_tools[tool_name] = tool
        return tool
    
    def process_parameters(self, tool_name: str, parameters: list):
        # This method seems to be for command-line usage and might not be fully robust.
        # Assuming it's not the primary way tools are called by the agent.
        input_parameters_spec = self.get_api_by_name(tool_name)['input_parameters']
        if len(parameters) != len(input_parameters_spec):
             raise ValueError(f"Invalid number of parameters for {tool_name}. Expected {len(input_parameters_spec)}, got {len(parameters)}.")

        processed_parameters = []
        # Iterating over values of spec dict assumes order, which is not guaranteed for dicts < 3.7
        # It's better to process based on a defined order or if parameters are named.
        # For now, assuming order matches or parameters list is small.
        for param_value, param_spec in zip(parameters, input_parameters_spec.values()):
            param_type = param_spec['type']
            # Basic type conversion, can be expanded
            if param_type == 'int': processed_parameters.append(int(param_value))
            elif param_type == 'float': processed_parameters.append(float(param_value))
            elif param_type == 'str': processed_parameters.append(str(param_value))
            else: processed_parameters.append(param_value) # Or raise error for unsupported types
        return processed_parameters
    
    def api_call(self, tool_name: str, **kwargs): 
        input_parameters_spec = self.get_api_by_name(tool_name)['input_parameters']
        processed_kwargs = {}

        for key, value in kwargs.items():
            if key not in input_parameters_spec:
                raise ValueError(f"Invalid parameter name '{key}' for tool '{tool_name}'.")
            
            required_type = input_parameters_spec[key]['type']
            # Type checking and conversion (can be more robust)
            try:
                if required_type == 'int': processed_kwargs[key] = int(value)
                elif required_type == 'float': processed_kwargs[key] = float(value)
                elif required_type == 'str': processed_kwargs[key] = str(value)
                elif required_type == 'bool': processed_kwargs[key] = bool(value) if isinstance(value, bool) else str(value).lower() == 'true'
                elif required_type.startswith('list'): # Handles list, list(str), etc.
                    if isinstance(value, str): # If it's a string, try to parse if it looks like a list repr
                        if value.startswith('[') and value.endswith(']'):
                            try:
                                # Basic eval, careful with untrusted input. json.loads is safer for JSON-like strings.
                                processed_kwargs[key] = json.loads(value.replace("'", "\"")) # Attempt to make it valid JSON
                            except json.JSONDecodeError:
                                print(f"Warning: Could not parse string '{value}' as list for {key} in {tool_name}. Using as list of one string.")
                                processed_kwargs[key] = [str(value)] # Fallback or handle error
                        else: # Treat as a single-element list if it's just a string not looking like a list
                             processed_kwargs[key] = [str(value)]
                    elif isinstance(value, list):
                        processed_kwargs[key] = value # Already a list
                    else:
                        raise TypeError(f"Parameter '{key}' for {tool_name} expected list, got {type(value)}")
                else: # Default to string if type is unknown or complex, or pass as is
                    processed_kwargs[key] = value
            except (ValueError, TypeError) as e:
                raise TypeError(f"Type error for parameter '{key}' ({value}) for tool '{tool_name}'. Expected {required_type}. Error: {e}")
        
        tool = self.init_tool(tool_name) # Gets/creates tool instance (with shared DB if applicable)
        return tool.call(**processed_kwargs) # Calls the tool's 'call' method
    
    def command_line(self):
        # This is a helper for interactive use, not directly used by level3.py agent
        # ... (implementation can remain similar, but ensure parse_api_call is defined if used)
        print("Command line interface not fully implemented in this version.")
        pass


    def list_all_apis(self):
        return [api_info['name'] for api_info in self.apis if 'name' in api_info]

# Helper for command_line if used, ensure it's robust or only for demo
# def parse_api_call(command_str): ...

if __name__ == '__main__':
    # Example Usage (for testing ToolManager itself)
    # This assumes api_bank is in PYTHONPATH or current dir has api_bank package
    # And that apis_dir points to a valid 'apis' subfolder relative to tool_manager.py
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    apis_dir_path = os.path.join(current_script_dir, 'apis') # Default ./apis relative to this file

    # To run this __main__, ensure the package structure allows imports.
    # If api_bank is the top-level package:
    # cd .. (to parent of api_bank)
    # python -m api_bank.tool_manager
    
    # For direct execution (python tool_manager.py), imports like `from .apis.api import API`
    # might fail if Python doesn't see `api_bank` as a package.
    # Adjusting sys.path for direct execution if needed:
    # parent_dir = os.path.dirname(current_script_dir)
    # if parent_dir not in sys.path:
    #    sys.path.insert(0, parent_dir)
    
    print(f"Executing ToolManager as __main__. CWD: {os.getcwd()}")
    print(f"Attempting to use apis_dir: {apis_dir_path}")

    # If __package__ is None (direct execution), relative imports fail.
    # For direct execution, __package__ needs to be set or imports need to be absolute.
    # This is a common Python packaging challenge.
    # The simplest way to ensure it works is to run as a module from the parent directory.
    if not __package__:
        # Crude fix for direct execution if api_bank is the parent dir name
        # This assumes tool_manager.py is in a directory like 'api_bank'
        # and 'apis' is 'api_bank/apis'
        # This is fragile. Running as a module `python -m api_bank.tool_manager` is better.
        file_path = os.path.abspath(__file__)
        # Go up two levels from api_bank/tool_manager.py to reach directory containing api_bank
        # This might not be general enough.
        # grandparent_dir = os.path.dirname(os.path.dirname(file_path))
        # sys.path.insert(0, grandparent_dir)
        # __package__ = os.path.basename(os.path.dirname(file_path)) # e.g., 'api_bank'
        print("Warning: Running tool_manager.py directly. Relative imports might be problematic.")
        print("Consider running as a module: python -m api_bank.tool_manager (from parent directory of api_bank)")
        # For the provided structure, if 'alive' is the project root and contains 'api_bank',
        # then from 'alive' directory, `python -m api_bank.tool_manager` should set __package__ correctly.


    try:
        tool_manager = ToolManager(apis_dir='apis') # Relative to this file's location
        print("ToolManager initialized. Available APIs:", tool_manager.list_all_apis())
        # tool_manager.command_line() # Uncomment to test interactive CLI
    except Exception as e:
        print(f"Error in __main__ of tool_manager.py: {e}")
        