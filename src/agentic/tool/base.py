import re
import json
from inspect import signature
from pydantic import BaseModel
from typing import Any, Callable



class Function(BaseModel):
    def __init__(self, func: Callable[..., Any]):
        super().__init__()
        self._func = func
        self._name = func.__name__
        self._doc = func.__doc__
        self._signature = signature(func)
        self._return_type = self._signature.return_annotation

    def run(self, args: BaseModel) -> Any:
        result = self._run(**args.model_dump())
        assert self._return_type == type(result)
        return result