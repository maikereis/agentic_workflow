import json
from typing import Any, Callable, List, Union

from ollama._client import Client

from src.tool import ToolCallProcessor, ToolKit


TOOL_PROMPT = """
You are a function calling AI model.
If a function or tool is unavailable, respond with trained data.
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags, here are the available tools:
<tools>
%s
</tools>

For each function call to a listed function, return a JSON object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>, "id": <monotonically-increasing-id>}
</tool_call>
"""

class ToolAgent:
    """
    An agent that interacts with a model capable of dynamically calling functions (tools) based on the user's query.

    Parameters
    ----------
    name : str, optional
        Name of the ToolAgent instance.
    client : Client
        A client instance to interact with the AI model.
    model : str
        Model identifier for the AI model to use.
    toolkit : Union[ToolKit, List[Callable[..., Any]]], optional
        A toolkit containing callable tools for use in responses. If a list is provided, it is converted to a ToolKit instance.
    system_message : str, optional
        System message template containing available tools and instructions, by default TOOL_PROMPT.

    Attributes
    ----------
    client : Client
        The client instance to interact with the AI model.
    model : str
        The model identifier for the AI model in use.
    toolkit : ToolKit
        A toolkit containing tools available for calling within the AI's responses.
    input_messages : list
        A list of dictionaries containing system and user messages.
    output_messages : list
        A list of dictionaries containing user and tool response messages.
    system_message : str
        The system message generated with available tools for the AI model.
    """

    def __init__(
        self,
        name: str = None,
        client: Client = None,
        model: str = None,
        toolkit: Union[ToolKit, List[Callable[..., Any]]] = None,
        system_message: str = TOOL_PROMPT,
    ):
        self.client = client
        self.model = model
        if isinstance(toolkit, list):
            self.toolkit = ToolKit(tools=toolkit)
        else:
            self.toolkit = toolkit
        self.input_messages = []
        self.output_messages = []
        self.system_message = self._initialize_system_message(system_message)

    def _tool_schemas_str(self) -> str:
        """
        Generate a formatted string of tool schemas.

        Returns
        -------
        str
            JSON-formatted string of all available tool schemas.
        """
        return "\n\n".join(
            [json.dumps(schema, indent=4) for schema in self.toolkit.tools_schemas()]
        )

    def _initialize_system_message(self, system_message: str) -> str:
        """
        Format and initialize the system message with available tool schemas.

        Parameters
        ----------
        system_message : str
            System message template to format with tool schemas.

        Returns
        -------
        str
            The formatted system message with tool schemas embedded.
        """
        return system_message % self._tool_schemas_str()

    def start(self, message: str) -> str:
        """
        Send a message to the model and handle tool calls returned by the model.

        Parameters
        ----------
        message : str
            The user's message or query to the model.

        Returns
        -------
        str
            The model's final response after processing any tool calls.
        """
        self.input_messages.append(
            {
                "role": "system",
                "content": self.system_message,
            }
        )
        self.input_messages.append({"role": "user", "content": message})

        response = self.client.chat(model=self.model, messages=self.input_messages)

        processor = ToolCallProcessor(response["message"]["content"])
        self.output_messages.append({"role": "user", "content": message})

        for call in processor.calls:
            try:
                result = self.toolkit.get_tool_by_name(call.name).run(call.arguments)
                self.output_messages.append(
                    {
                        "role": "tool",
                        "content": f"result of call to {call.name}: {result}",
                    }
                )
            except AttributeError as ae:
                print(ae)

        result = self.client.chat(model=self.model, messages=self.output_messages)

        return result["message"]["content"]
