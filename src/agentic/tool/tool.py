import re
import json
from inspect import signature

class ToolRegistry:
    """
    A class to manage and register available tools (functions).
    
    This registry allows the user to register tools, retrieve them, and generate
    JSON schemas for each tool's function signature.
    """

    def __init__(self):
        """
        Initialize a ToolRegistry instance.
        
        Attributes
        ----------
        tools : dict
            A dictionary that stores tools with their names as keys and functions as values.
        """
        self.tools = {}

    def register(self, name: str, func):
        """
        Register a new function as a tool.
        
        Parameters
        ----------
        name : str
            The name of the tool to be registered.
        func : function
            The function to register as a tool.
        
        Raises
        ------
        TypeError
            If `func` is not callable.
        """
        if not callable(func):
            raise TypeError(f"The tool '{name}' must be a callable function.")
        self.tools[name] = func

    def get_function(self, name: str):
        """
        Get a registered function by its name.
        
        Parameters
        ----------
        name : str
            The name of the tool (function) to retrieve.
        
        Returns
        -------
        function or None
            The function registered under the given name, or None if not found.
        """
        return self.tools.get(name)

    def get_json_schema(self, func):
        """
        Generate the JSON schema for a given function.
        
        Parameters
        ----------
        func : function
            The function for which to generate the JSON schema.
        
        Returns
        -------
        str
            A JSON-formatted string representing the function's signature schema.
        
        Raises
        ------
        ValueError
            If the provided object is not a callable function.
        """
        if not callable(func):
            raise ValueError("The argument must be a callable function.")
        
        sigs = signature(func)
        desc = func.__doc__ or "No description provided."
        name = func.__name__

        schema = {
            "name": name,
            "description": desc,
            "parameters": {
                "properties": {
                    param_name: {"type": param.annotation.__name__ if param.annotation != param.empty else "unknown"}
                    for param_name, param in sigs.parameters.items()
                }
            }
        }

        return json.dumps(schema, indent=4)

    def generate_tools_schema(self):
        """
        Generate JSON schemas for all registered tools.
        
        Returns
        -------
        str
            A concatenation of JSON schemas for all tools, each schema on a new line.
        """
        schemas = [self.get_json_schema(func) for func in self.tools.values()]
        return "\n".join(schemas)


class ToolCallProcessor:
    """
    A class to process tool calls and return the results formatted in XML-like tags.

    This class takes a string containing multiple tool calls, parses each call to 
    identify the tool name and its arguments, executes the corresponding function 
    from the tool registry, and collects the results.
    """

    def __init__(self, tool_registry):
        """
        Initialize the ToolCallProcessor with a tool registry.

        Parameters
        ----------
        tool_registry : object
            An object that has a method `get_function` which retrieves a function 
            based on the tool's name.
        """
        self.tool_registry = tool_registry

    def parse_multiple_tool_calls(self, content):
        """
        Parse multiple <tool_call> entries from a string and return a list of corresponding dictionaries.

        Parameters
        ----------
        content : str
            A string containing multiple XML-like <tool_call> entries, where each entry contains JSON data.

        Returns
        -------
        list of dict
            A list of dictionaries parsed from the JSON content inside each <tool_call> tag.
        """
        call_request_list = []

        for match in re.finditer(r"<tool_call>\n(.+)?\n</tool_call>", content):
            try:
                group_json = json.loads(match.group(1))
                call_request_list.append(group_json)
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError: {e}")
                print(f"Failed to decode JSON: {match.group(1)}")
                # Optionally, you can log or handle the malformed content here.
                continue

        return call_request_list

    def is_function_registered(self, tool_name):
        """
        Check if a function is registered in the tool registry.

        Parameters
        ----------
        tool_name : str
            The name of the tool for which to check function registration.

        Returns
        -------
        bool
            True if the function is registered, False otherwise.
        """
        return self.tool_registry.get_function(tool_name) is not None

    def validate_parameters(self, func, arguments):
        """
        Validate the presence and types of arguments required by the function.

        Parameters
        ----------
        func : callable
            The function whose parameters need to be validated.
        arguments : dict
            A dictionary containing the arguments to be passed to the function.

        Returns
        -------
        bool
            True if the arguments conform to the expected types and presence, False otherwise.
        """
        from inspect import signature

        sig = signature(func)
        for param_name, param in sig.parameters.items():
            if param_name not in arguments:
                return False  # Missing required argument
            # Optionally, you could add checks for parameter types here using `param.annotation`.

        return True

    def process_tool_calls(self, tool_call_str):
        """
        Process a string of tool calls and return the results formatted in XML-like tags.

        Parameters
        ----------
        tool_call_str : str
            A string containing multiple tool call segments formatted as 
            '<tool_call>...</tool_call>', where each segment contains JSON data 
            specifying the tool name, arguments, and a unique identifier.

        Returns
        -------
        str
            A string that contains the results of the executed tool calls, 
            formatted within '<tool_response>' tags. Each result corresponds 
            to the respective tool call and is presented in the order they were 
            processed.
        """
        tool_call_result = []
        for tool_info in self.parse_multiple_tool_calls(tool_call_str):
            tool_name = tool_info['name']
            arguments = tool_info['arguments']
            tool_id = tool_info['id']

            # Check if the function is registered
            if not self.is_function_registered(tool_name):
                tool_call_result.append({
                    "id": tool_id,
                    "result": f"Error: Function '{tool_name}' not found in the registry."
                })
                continue

            # Retrieve the function
            func = self.tool_registry.get_function(tool_name)

            # Validate parameters
            if not self.validate_parameters(func, arguments):
                tool_call_result.append({
                    "id": tool_id,
                    "result": f"Error: Invalid or missing parameters for function '{tool_name}'."
                })
                continue

            # Try to execute the function with the provided arguments
            try:
                result = func(**arguments)
                tool_call_result.append({
                    "id": tool_id,
                    "result": result
                })
            except Exception as e:
                tool_call_result.append({
                    "id": tool_id,
                    "result": f"Error executing '{tool_name}': {e}"
                })

        # Format the results into XML-like output
        output = ''.join(f'<tool_response>\n{item["result"]}\n</tool_response>\n' for item in tool_call_result)
        return output