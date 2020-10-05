#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

from abc import ABC, abstractmethod
from enum import Enum
from inspect import isclass
from itertools import chain
from typing import Any, ClassVar, Dict, List, Optional, Sized, Type, TypeVar, Union

import attr


# This decorator must be applied to all classes that are intended to be used as
# schemas. It parses the class-level attributes defined as attr.ibs and produces
# __init__, __eq__, __hash__, __str__ and other magic methods for them.
# TODO Remove noqa when flake8 will understand kw_only added in attrs-18.2.0.
schema = attr.s(kw_only=True, slots=True, frozen=True)  # noqa


TRUE_STRINGS = {"1", "y", "yes", "true", "on"}
FALSE_STRINGS = {"0", "n", "no", "false", "off"}


# Optional[foo] is an alias for Union[foo, NoneType], but Unions are weird.
def unpack_optional(type_):
    try:
        (candidate_arg,) = set(type_.__args__) - {type(None)}
    except (AttributeError, LookupError, ValueError):
        raise TypeError("Not an optional type")
    if type_ != Optional[candidate_arg]:
        raise TypeError("Not an optional type")
    return candidate_arg


def has_origin(type_, base_type):
    try:
        return issubclass(type_.__origin__, base_type)
    except (AttributeError, TypeError):
        return False


def mixed_case_to_lowercase(key: str) -> str:
    return "".join("_%s" % c.lower() if c.isupper() else c for c in key)


class DeepTypeError(TypeError):
    def __init__(self, message):
        self.message = message
        self.path = ""

    def prepend_attr(self, attr: str):
        self.path = ".%s%s" % (attr, self.path)

    def prepend_index(self, idx: int):
        self.path = "[%d]%s" % (idx, self.path)

    def prepend_key(self, key):
        self.path = "[%r]%s" % (key, self.path)

    def __str__(self):
        path = self.path.lstrip(".")
        if not path:
            return self.message
        return "%s: %s" % (path, self.message)


class Mapper(ABC):
    @abstractmethod
    def map_bool(self, data: Any) -> bool:
        pass

    @staticmethod
    def map_int(data: Any) -> int:
        if not isinstance(data, int):
            raise DeepTypeError("Not an int")
        return data

    @staticmethod
    def map_float(data: Any) -> float:
        # Integers are real numbers: ints are floats.
        if not isinstance(data, (int, float)):
            raise DeepTypeError("Not a float")
        return float(data)

    @staticmethod
    def map_str(data: Any) -> str:
        if not isinstance(data, str):
            raise DeepTypeError("Not a str")
        return data

    @abstractmethod
    def map_enum(self, data: Any, type_: Type[Enum]) -> Any:
        pass

    def map_list(self, data: Any, type_) -> List:
        if not isinstance(data, list):
            raise DeepTypeError("Not a list")
        (element_type,) = type_.__args__
        result = []
        for idx, element in enumerate(data):
            try:
                result.append(self.map_with_type(element, element_type))
            except DeepTypeError as err:
                err.prepend_index(idx)
                raise err
        return result

    def map_dict(self, data: Any, type_) -> Dict:
        if not isinstance(data, dict):
            raise DeepTypeError("Not a dict")
        key_type, value_type = type_.__args__
        result = {}
        for key, value in data.items():
            # We don't distinguish between errors on keys and on values.
            try:
                result[self.map_with_type(key, key_type)] = self.map_with_type(
                    value, value_type
                )
            except DeepTypeError as err:
                err.prepend_key(key)
                raise err
        return result

    @abstractmethod
    def map_schema(self, data: Any, type_: Type["Schema"]) -> Any:
        pass

    def map_with_type(self, data: Any, type_: Type) -> Any:
        # Needs to come first as in this case type_ is an instance, not a class.
        try:
            base_type = unpack_optional(type_)
        except TypeError:
            pass
        else:
            if data is None:
                return None
            return self.map_with_type(data, base_type)
        if isclass(type_) and issubclass(type_, bool):
            return self.map_bool(data)
        if isclass(type_) and issubclass(type_, int):
            return self.map_int(data)
        if isclass(type_) and issubclass(type_, float):
            return self.map_float(data)
        if isclass(type_) and issubclass(type_, str):
            return self.map_str(data)
        if isclass(type_) and issubclass(type_, Enum):
            return self.map_enum(data, type_)
        if has_origin(type_, list):
            return self.map_list(data, type_)
        if has_origin(type_, dict):
            return self.map_dict(data, type_)
        if isclass(type_) and issubclass(type_, Schema):
            return self.map_schema(data, type_)
        raise NotImplementedError("Unknown type: %s" % type_)


class Loader(Mapper):
    @staticmethod
    def map_bool(data: Any) -> bool:
        if not isinstance(data, bool):
            # Be lenient.
            if isinstance(data, int):
                if data == 0:
                    return False
                if data == 1:
                    return True
            if isinstance(data, str):
                if data.lower() in TRUE_STRINGS:
                    return True
                if data.lower() in FALSE_STRINGS:
                    return False
            raise DeepTypeError("Not a bool")
        return data

    @staticmethod
    def map_enum(data: Any, type_: Type[Enum]) -> Enum:
        if not isinstance(data, str):
            # Be lenient.
            if isinstance(data, type_):
                return data
            raise DeepTypeError("Not a str: %s" % data)
        try:
            return type_[data.upper()]
        except KeyError:
            raise DeepTypeError("Unknown option: %s" % data) from None

    def map_schema(self, data: Any, type_: Type["Schema"]) -> "Schema":
        if not isinstance(data, dict):
            raise DeepTypeError("Not a schema")
        fields = attr.fields_dict(type_)
        kwargs = {}
        for key, value in data.items():
            # Convert legacy mixedCase to lower_case.
            key = mixed_case_to_lowercase(key)
            try:
                field = fields[key]
            except LookupError:
                raise DeepTypeError("Unknown key: %s" % key) from None
            if field.type is None:
                raise RuntimeError("Unannotated field: %s" % key)
            try:
                kwargs[key] = self.map_with_type(value, field.type)
            except DeepTypeError as err:
                err.prepend_attr(key)
                raise err
        # TODO Remove noqa when flake8 will understand kw_only added in attrs-18.2.0.
        try:
            return type_(**kwargs)  # noqa
        except (ValueError, TypeError) as err:
            raise DeepTypeError(str(err)) from None


class Dumper(Mapper):
    @staticmethod
    def map_bool(data: Any) -> bool:
        if not isinstance(data, bool):
            raise DeepTypeError("Not a bool")
        return data

    @staticmethod
    def map_enum(data: Any, type_: Type[Enum]) -> str:
        if not isinstance(data, type_):
            raise TypeError("Not a %s" % type_.__name__)
        return data.name.lower()

    def map_schema(self, data: Any, type_: Type["Schema"]) -> Dict[str, Any]:
        result = {}
        for key, field in attr.fields_dict(type_).items():
            if field.type is None:
                raise RuntimeError("Unannotated field: %s" % key)
            try:
                result[key] = self.map_with_type(getattr(data, key), field.type)
            except DeepTypeError as err:
                err.prepend_attr(key)
                raise err
        return result


TSchema = TypeVar("TSchema", bound="Schema")


@schema
class Schema:
    """A class representing a configuration as a set of keys and values.

    When defining the configuration schema supported by some module (say, the
    model, the training, ...), subclass this class, decorate it with @schema and
    define the configuration keys you require as class attributes with type
    annotations whose values are attr.ib() instances. Give default values if
    appropriate and provide descriptions for the fields by using the `help`
    key of the `metadata` argument. Doing this automatically gives you
    space-optimized classes (using slots) equipped with the most common magic
    methods (__init__, __eq__, ...) plus some convenience methods to  produce
    help text and to parse instances of these schemas serialized as dicts (with
    type checking!).

    """

    NAME: ClassVar[str] = ""  # The name of this config, shown in the help text.

    @classmethod
    def represent_type(cls, type_):
        try:
            base_type = unpack_optional(type_)
        except TypeError:
            pass
        else:
            return "?%s" % cls.represent_type(base_type)
        if isclass(type_) and issubclass(type_, Enum):
            return "(%s)" % "|".join(member.name.lower() for member in type_)
        if has_origin(type_, list):
            (element_type,) = type_.__args__
            return "[%s]" % cls.represent_type(element_type)
        if has_origin(type_, dict):
            key_type, value_type = type_.__args__
            return "{%s: %s}" % (
                cls.represent_type(key_type),
                cls.represent_type(value_type),
            )
        if isclass(type_) and issubclass(type_, Schema):
            return "%s" % type_.NAME
        return type_.__name__

    @classmethod
    def help(cls):
        subschemas = []

        def append_if_subschema(s):
            if isclass(s) and issubclass(s, Schema) and s not in subschemas:
                subschemas.append(s)

        lines = []
        lines.append("%s:" % cls.NAME)
        lines.append("")
        for field in attr.fields(cls):
            type_ = field.type
            lines.append("  %s (%s)" % (field.name, cls.represent_type(type_)))
            help = field.metadata.get("help", None)
            if help is not None:
                lines.append("\t%s" % help)
            try:
                type_ = unpack_optional(type_)
            except TypeError:
                pass
            append_if_subschema(type_)
            if has_origin(type_, list):
                (element_type,) = type_.__args__
                append_if_subschema(element_type)
            elif has_origin(type_, dict):
                _, value_type = type_.__args__
                append_if_subschema(value_type)
        lines.append("")
        lines.append("")

        return list(chain(*(subschema.help() for subschema in subschemas), lines))

    @classmethod
    def from_dict(cls: Type[TSchema], data: Dict[str, Any]) -> TSchema:
        return Loader().map_with_type(data, cls)

    def to_dict(self) -> Dict[str, Any]:
        return Dumper().map_with_type(self, type(self))


def non_negative(_, field: attr.Attribute, value: Union[int, float]):
    if value < 0:
        raise ValueError("%s must be >= 0, got %s" % (field.name, value))


def positive(_, field: attr.Attribute, value: Union[int, float]):
    if value <= 0:
        raise ValueError("%s must be > 0, got %s" % (field.name, value))


def non_empty(_, field: attr.Attribute, value: Sized):
    if len(value) == 0:
        raise ValueError("%s must be non-empty" % field.name)


def extract_nested_type(type_: Any, path: List[str]) -> Any:
    try:
        type_ = unpack_optional(type_)
    except TypeError:
        pass
    if len(path) == 0:
        return type_
    if has_origin(type_, list):
        (element_type,) = type_.__args__
        return extract_nested_type(element_type, path[1:])
    if has_origin(type_, dict):
        _, value_type = type_.__args__
        return extract_nested_type(value_type, path[1:])
    if isclass(type_) and issubclass(type_, Schema):
        child_type = attr.fields_dict(type_)[path[0]].type
        return extract_nested_type(child_type, path[1:])
    raise NotImplementedError("Unknown type %r" % type_)


def inject_nested_value(data: Any, path: List[str], value: Any) -> Any:
    if len(path) == 0:
        raise ValueError("Got empty path")
    if isinstance(data, list):
        index, path = int(path[0]), path[1:]
        if len(path) == 0:
            data[index] = value
        else:
            inject_nested_value(data[index], path, value)
    elif isinstance(data, dict):
        key, path = path[0], path[1:]
        if len(path) == 0:
            data[key] = value
        else:
            inject_nested_value(data[key], path, value)
    else:
        raise NotImplementedError("Data of unknown type: %r" % data)
