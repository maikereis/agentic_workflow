import json
import inspect
from inspect import signature
from pydantic import BaseModel, create_model
from typing import List, Any, Callable, Union, Mapping


from typing import Callable, Any, Mapping
from pydantic import BaseModel, create_model
import inspect
from inspect import signature

class Function(BaseModel):
    """
    A wrapper around a callable to store its metadata and provide a way to run it with validation.

    Parameters
    ----------
    func : Callable[..., Any]
        The function to be wrapped, whose metadata will be stored.

    Attributes
    ----------
    _func : Callable[..., Any]
        The original function being wrapped.
    _name : str
        The name of the function.
    _doc : str
        The docstring of the function.
    _signature : inspect.Signature
        The signature of the function.
    _return_type : Any
        The expected return type of the function.
    _dynamic_model : Type[BaseModel]
        A dynamically created Pydantic model for validating the function's arguments.

    Methods
    -------
    get_args_model(signature: inspect.Signature)
        Creates a dynamic Pydantic model for validating the arguments of the function.
    run(args: Mapping[str, Any]) -> Any
        Runs the function with validated arguments and checks the return type.
    """
    
    def __init__(self, func: Callable[..., Any]):
        """
        Initializes the Function wrapper.

        Parameters
        ----------
        func : Callable[..., Any]
            The function to wrap.
        """
        super().__init__()
        self._func = func
        self._name = func.__name__
        self._doc = func.__doc__
        self._signature = signature(func)
        self._return_type = self._signature.return_annotation
        self._dynamic_model = self.get_args_model(self._signature)

    def get_args_model(self, signature: inspect.Signature):
        """
        Creates a dynamic Pydantic model based on the function's signature.

        Parameters
        ----------
        signature : inspect.Signature
            The signature of the function whose arguments need validation.

        Returns
        -------
        BaseModel
            A Pydantic model for validating the function's arguments.
        """
        model = create_model(
            'DynamicArgs',
            **{
                param_name: (param.annotation, ...)
                if param.annotation != param.empty
                else (Any, ...)
                for param_name, param in signature.parameters.items()
            }
        )
        return model

    @property
    def name(self):
        """
        Returns the name of the wrapped function.

        Returns
        -------
        str
            The name of the function.
        """
        return self._name

    @property
    def doc(self):
        """
        Returns the docstring of the wrapped function.

        Returns
        -------
        str
            The docstring of the function.
        """
        return self._doc

    @property
    def return_type(self):
        """
        Returns the expected return type of the function.

        Returns
        -------
        Any
            The return type of the function.
        """
        return self._return_type

    def run(self, args: Mapping[str, Any]) -> Any:
        """
        Runs the function with the provided arguments after validation.

        Parameters
        ----------
        args : Mapping[str, Any]
            A dictionary of argument names and their values to be passed to the function.

        Returns
        -------
        Any
            The result of the function call.

        Raises
        ------
        AssertionError
            If the type of the result does not match the expected return type.
        """
        validated_args = self._dynamic_model.model_validate(args)
        result = self._func(**validated_args.model_dump())
        assert self._return_type == type(result), f"Expected return type {self._return_type}, but got {type(result)}"
        return result

    def __repr__(self):
        """
        Returns a string representation of the Function object.

        Returns
        -------
        str
            A representation including the function name, signature, and a preview of the docstring.
        """
        doc_preview = (self._doc[:30] + '...') if self._doc and len(self._doc) > 30 else self._doc
        return f"Function(name={self._name}, signature={self._signature}, doc={doc_preview})"


class Tool(Function):
    """
    A specialized version of the `Function` class that provides additional
    functionality for schema extraction.

    Parameters
    ----------
    func : Callable[..., Any]
        The function to be wrapped and processed as a Tool.

    Attributes
    ----------
    _func : Callable[..., Any]
        The original function being wrapped.
    _name : str
        The name of the function.
    _doc : str
        The docstring of the function.
    _signature : inspect.Signature
        The signature of the function.
    _return_type : Any
        The expected return type of the function.
    _dynamic_model : Type[BaseModel]
        A dynamically created Pydantic model for validating the function's arguments.

    Methods
    -------
    schema : dict
        Provides a schema representation of the function's metadata.
    """
    
    def __init__(self, func: Callable[..., Any]):
        """
        Initializes the Tool wrapper.

        Parameters
        ----------
        func : Callable[..., Any]
            The function to wrap.
        """
        super().__init__(func)
    
    @property
    def schema(self):
        """
        Generates a schema dictionary representing the function's metadata.

        Returns
        -------
        dict
            A dictionary containing the function's name, docstring, and parameters.
            The parameters include their types as specified in the function signature.
        """
        schema = {
            "name": self.name,
            "doc": self.doc,
            "parameters": {
                "properties": {
                    param_name: {"type": param.annotation.__name__ if param.annotation != param.empty else "unknown"}
                    for param_name, param in self._signature.parameters.items()
                }
            }
        }
        return schema

    def __repr__(self):
        """
        Returns a string representation of the Tool object.

        Returns
        -------
        str
            A representation including the function name, signature, and a preview of the docstring.
        """
        doc_preview = (self._doc[:30] + '...') if self._doc and len(self._doc) > 30 else self._doc
        return f"Tool(name={self._name}, signature={self._signature}, doc={doc_preview})"


from typing import Callable, Any, List, Union
from pydantic import BaseModel

class ToolKit(BaseModel):
    """
    A collection of `Tool` instances that allows managing and accessing
    a set of callable functions as tools.

    Parameters
    ----------
    tools : Union[Callable[..., Any], List[Callable[..., Any]]]
        A single callable or a list of callables to be added to the toolkit.

    Attributes
    ----------
    _tools : dict
        A dictionary mapping tool names to their respective `Tool` instances.

    Methods
    -------
    tools : List[Tool]
        Returns the list of `Tool` instances in the toolkit.
    tools_schemas() -> List[dict]
        Returns a list of schemas for each `Tool` in the toolkit.
    get_tool_by_name(name: str) -> Union[Tool, None]
        Retrieves a `Tool` by its name.
    """
    
    def __init__(self, tools: Union[Callable[..., Any], List[Callable[..., Any]]]):
        """
        Initializes the ToolKit with the provided tools.

        Parameters
        ----------
        tools : Union[Callable[..., Any], List[Callable[..., Any]]]
            A callable or list of callables to initialize the toolkit.
        """
        super().__init__()
        self._tools = {}
        self.tools = tools  # Use the setter to initialize tools

    @property
    def tools(self) -> List[Tool]:
        """
        Returns the list of `Tool` instances.

        Returns
        -------
        List[Tool]
            A list of `Tool` instances contained in the toolkit.
        """
        return list(self._tools.values())

    @tools.setter
    def tools(self, tools: Union[Callable[..., Any], List[Callable[..., Any]]]):
        """
        Adds tools to the toolkit, ensuring unique names.

        Parameters
        ----------
        tools : Union[Callable[..., Any], List[Callable[..., Any]]]
            A callable or list of callables to add to the toolkit.
        """
        if not isinstance(tools, list):
            tools = [tools]

        for tool in tools:
            tool_instance = Tool(tool)
            name = tool_instance.name

            # Handle conflicts by appending a number if a name already exists
            if name in self._tools:
                counter = 1
                new_name = f"{name}_{counter}"
                while new_name in self._tools:
                    counter += 1
                    new_name = f"{name}_{counter}"
                name = new_name

            # Store the tool instance with the resolved name
            self._tools[name] = tool_instance

    def tools_schemas(self) -> List[dict]:
        """
        Returns a list of schemas for each `Tool` in the toolkit.

        Returns
        -------
        List[dict]
            A list of schema dictionaries, one for each `Tool`.
        """
        return [tool.schema for tool in self._tools.values()]

    def get_tool_by_name(self, name: str) -> Union[Tool, None]:
        """
        Retrieve a `Tool` by its name.

        Parameters
        ----------
        name : str
            The name of the tool to retrieve.

        Returns
        -------
        Union[Tool, None]
            The `Tool` instance if found, otherwise `None`.
        """
        return self._tools.get(name)

    def __repr__(self):
        """
        Returns a string representation of the ToolKit object.

        Returns
        -------
        str
            A representation including the number of tools and a preview of their names.
        """
        tool_names = list(self._tools.keys())
        tool_preview = ', '.join(tool_names[:3])  # Show up to the first 3 tool names
        if len(tool_names) > 3:
            tool_preview += ', ...'  # Indicate there are more tools if the list is long
        return f"ToolKit(num_tools={len(tool_names)}, tools=[{tool_preview}])"
