import json
import re
from typing import Any, Callable, List, Union

from ollama._client import Client

from agentic.tool import ToolCallProcessor, ToolKit

REACT_PROMPT = """
You are a function calling AI model. You operate breaking a task given by a user's question into steps: <thought>, <tool_calls>, <tool_response>.
Take special attention to the functions params dtypes.
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags, here are the available tools:
<tools>
%s
</tools>

The reasoning and thoughts should be enclosed within <thought></thought> XML tags.
<thought>
thought/reasoning
</thought>

If a function call is needed and the function is listed, return a JSON object containing the function name and its arguments, enclosed within <tool_calls></tool_calls> XML tags.
<tool_calls>
{"name": <function-name>, "arguments": <args-json-object>, "id": <monotonically-increasing-id>}
</tool_calls>

The function call response will be enclosed within <tool_response></tool_response> XML tags.
<tool_response>
tool response
</tool_response>

The final response to the user will be enclosed within <response></response> XML tags.
<response>
Response after reasoning and acting (ReAct)
</response>

Example session:

<question>Can you verify if it's friday and if yes, can you show a message \"IT'S FRIDAY\"?</question>
<thought>I need to check if today is friday, if today is friday i should show a message saying \"IT'S FRIDAY\".First i will verify if it's friday.</thought>
<tool_calls>{\"name\": \"is_friday\",\"arguments\": {\"date\": \"10/01/2024\"}, \"id\": 0}</tool_calls>

Then you will be feeded with the result of the function call:
<tool_response>True</tool_response>

Then you can go to the next step with depends on first:
<thought>It's friday, now i should show a message:</thought>
<tool_calls>{\"name\": \"show_message\",\"arguments\": {\"message\": \"IT'S FRIDAY\"}, \"id\": 1}</tool_calls>

Then you will be feeded with the result of the function call:
<tool_response>{\"success\": True}</tool_response>

And if no more function calls are needed, you can respond with:
<response>Today is friday! I showed a message remenbering you of this.</response>
"""


class PlanningAgent:
    """
    A class to manage the planning agent that interacts with a client and
    utilizes tools from a toolkit to respond to user messages.

    Parameters
    ----------
    name : str, optional
        The name of the planning agent.
    client : Client, optional
        An instance of the Client class used for communication.
    model : str, optional
        The model identifier for the chat API.
    toolkit : Union[ToolKit, List[Callable[..., Any]]], optional
        A ToolKit instance or a list of callables to initialize the toolkit.
    system_message : str, optional
        A formatted system message used to initialize the agent. Defaults to `REACT_PROMPT`.

    Attributes
    ----------
    client : Client
        The client instance for communicating with the chat API.
    model : str
        The model identifier for the chat API.
    toolkit : ToolKit
        The toolkit containing callable functions for the agent to use.
    input_messages : List[dict]
        A list of messages exchanged between the user, system, and assistant.
    system_message : str
        The system message initialized for the agent.

    Methods
    -------
    _tool_schemas_str() -> str
        Returns a string representation of the schemas for each tool in the toolkit.
    _initialize_system_message(system_message: str) -> str
        Initializes the system message using the provided template and tool schemas.
    start(message: str) -> str
        Starts the interaction with the user by processing the input message
        and returning the final response from the assistant.
    """

    def __init__(
        self,
        name: str = None,
        client: Client = None,
        model: str = None,
        toolkit: Union[ToolKit, List[Callable[..., Any]]] = None,
        system_message: str = REACT_PROMPT,
        max_iter=20,
    ):
        """
        Initializes the PlanningAgent with the provided parameters.

        Parameters
        ----------
        name : str, optional
            The name of the planning agent.
        client : Client, optional
            An instance of the Client class used for communication.
        model : str, optional
            The model identifier for the chat API.
        toolkit : Union[ToolKit, List[Callable[..., Any]]], optional
            A ToolKit instance or a list of callables to initialize the toolkit.
        system_message : str, optional
            A formatted system message used to initialize the agent. Defaults to `REACT_PROMPT`.
        """
        self.client = client
        self.model = model
        if isinstance(toolkit, list):
            self.toolkit = ToolKit(tools=toolkit)
        else:
            self.toolkit = toolkit
        self.input_messages = []
        self.system_message = self._initialize_system_message(system_message)
        self.max_iter = max_iter

    def _tool_schemas_str(self) -> str:
        """
        Returns a string representation of the schemas for each tool in the toolkit.

        Returns
        -------
        str
            A string containing the formatted JSON schemas of the tools.
        """
        return "\n\n".join(
            [json.dumps(schema, indent=4) for schema in self.toolkit.tools_schemas()]
        )

    def _initialize_system_message(self, system_message: str) -> str:
        """
        Initializes the system message using the provided template and tool schemas.

        Parameters
        ----------
        system_message : str
            The message template that includes placeholders for tool schemas.

        Returns
        -------
        str
            The initialized system message with tool schemas included.
        """
        return system_message % self._tool_schemas_str()

    def _is_final_response(self, content, pattern=r"<response>(.*?)</response>"):
        """
        Check if the input string contains <response></response> tags.

        Args:
            input_string (str): The string to check for <response> tags.

        Returns:
            bool: True if the tags are found, False otherwise.
        """
        # Search for the pattern in the input string
        match = bool(re.search(pattern, content, re.DOTALL))

        # Return True if the tags are found
        return match

    def start(self, message: str) -> str:
        """
        Starts the interaction with the user by processing the input message
        and returning the final response from the assistant.

        Parameters
        ----------
        message : str
            The user's input message to be processed.

        Returns
        -------
        str
            The final response generated by the assistant.
        """
        self.input_messages.append(
            {
                "role": "system",
                "content": self.system_message,
            }
        )
        self.input_messages.append(
            {"role": "user", "content": f"<question>{message}</question>"}
        )

        final_response = None

        i = 0
        while True or i < self.max_iter:
            response = self.client.chat(model=self.model, messages=self.input_messages)

            response_content = response["message"]["content"]

            if self._is_final_response(response_content):
                final_response = response_content
                break

            processor = ToolCallProcessor(response_content)

            self.input_messages.append(
                {"role": "assistant", "content": response_content}
            )

            for call in processor.calls:
                try:
                    result = self.toolkit.get_tool_by_name(call.name).run(
                        call.arguments
                    )

                    self.input_messages.append(
                        {
                            "role": "tool",
                            "content": f"<tool_response>{result}</tool_response>",
                        }
                    )
                except AttributeError as ae:
                    print(ae)
            i += 1

        return final_response
