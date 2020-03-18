#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

from enum import Enum
from typing import ClassVar, Dict, List, Optional, Union
from unittest import TestCase, main

import attr
from torchbiggraph.schema import (
    Dumper,
    Loader,
    Schema,
    extract_nested_type,
    inject_nested_value,
    schema,
    unpack_optional,
)


class TestUnpackOptional(TestCase):
    def test_has_no_args(self):
        with self.assertRaises(TypeError):
            unpack_optional(None)
        with self.assertRaises(TypeError):
            unpack_optional(42)
        with self.assertRaises(TypeError):
            unpack_optional("foo")
        with self.assertRaises(TypeError):
            unpack_optional(bool)
        with self.assertRaises(TypeError):
            unpack_optional(float)

    def test_is_no_union(self):
        with self.assertRaises(TypeError):
            unpack_optional(List[int])
        with self.assertRaises(TypeError):
            unpack_optional(Dict[str, str])

    def test_is_no_optional(self):
        with self.assertRaises(TypeError):
            unpack_optional(Union[int])  # In fact this is int.
        with self.assertRaises(TypeError):
            unpack_optional(Union[int, str])

    def test_is_optional(self):
        self.assertIs(unpack_optional(Optional[int]), int)
        self.assertIs(unpack_optional(Optional[str]), str)
        self.assertIs(unpack_optional(Union[None, str]), str)
        self.assertIs(unpack_optional(Union[int, type(None)]), int)


class SampleEnum(Enum):
    SPAM = "spam"
    SPAM_ALIAS = "spam"
    HAM = "bacon"
    EGGS = "eggs"


@schema
class SampleConfigTypes(Schema):

    NAME: ClassVar[str] = "my_types"

    my_bool: bool = attr.ib()
    my_int: int = attr.ib()
    my_float: float = attr.ib()
    my_str: str = attr.ib()
    my_enum: SampleEnum = attr.ib()
    my_optional_str: Optional[str] = attr.ib()
    my_list: List[str] = attr.ib()
    my_dict: Dict[str, int] = attr.ib()


@schema
class SampleConfigDefault(Schema):

    NAME: ClassVar[str] = "my_default"

    my_default: int = attr.ib(default=1)


@schema
class SampleConfigHelp(Schema):

    NAME: ClassVar[str] = "my_help"

    my_help_text: str = attr.ib(metadata={"help": "Pass a nice value here!"})


@schema
class SampleOuterConfig(Schema):

    NAME: ClassVar[str] = "my_outer_name"

    my_schema: Optional[SampleConfigTypes] = attr.ib()
    my_list_of_schemas: List[SampleConfigDefault] = attr.ib()
    my_dict_of_schemas: Dict[str, SampleConfigHelp] = attr.ib()


class BaseMapperMixin:
    def test_map_bool(self):
        self.assertIs(self.mapper.map_with_type(False, bool), False)
        self.assertIs(self.mapper.map_with_type(True, bool), True)
        with self.assertRaises(TypeError):
            self.mapper.map_with_type(None, bool)
        with self.assertRaises(TypeError):
            self.mapper.map_with_type(0xF00, bool)

    def test_map_int(self):
        self.assertEqual(self.mapper.map_with_type(0xF00, int), 0xF00)
        with self.assertRaises(TypeError):
            self.mapper.map_with_type(None, int)
        with self.assertRaises(TypeError):
            self.mapper.map_with_type("foo", int)

    def test_map_float(self):
        self.assertEqual(self.mapper.map_with_type(4.2, float), 4.2)
        with self.assertRaises(TypeError):
            self.mapper.map_with_type(None, float)
        with self.assertRaises(TypeError):
            self.mapper.map_with_type("foo", float)

    def test_map_str(self):
        self.assertEqual(self.mapper.map_with_type("foo", str), "foo")
        with self.assertRaises(TypeError):
            self.mapper.map_with_type(None, str)
        with self.assertRaises(TypeError):
            self.mapper.map_with_type(0xF00, str)

    def test_map_optional_str(self):
        self.assertEqual(self.mapper.map_with_type(None, Optional[str]), None)
        self.assertEqual(self.mapper.map_with_type("foo", Optional[str]), "foo")
        with self.assertRaises(TypeError):
            self.mapper.map_with_type(0xF00, Optional[str])

    def test_map_list_of_basic(self):
        self.assertEqual(self.mapper.map_with_type([], List[str]), [])
        self.assertEqual(
            self.mapper.map_with_type(["foo", "bar"], List[str]), ["foo", "bar"]
        )
        with self.assertRaises(TypeError):
            self.mapper.map_with_type("[foo]", List[str])
        with self.assertRaises(TypeError):
            self.mapper.map_with_type(["foo", 0xBA2], List[str])

    def test_map_dict_of_basic(self):
        self.assertEqual(self.mapper.map_with_type({}, Dict[str, int]), {})
        self.assertEqual(
            self.mapper.map_with_type({"foo": 42}, Dict[str, int]), {"foo": 42}
        )
        with self.assertRaises(TypeError):
            self.mapper.map_with_type("{}", Dict[str, int])
        with self.assertRaises(TypeError):
            self.mapper.map_with_type({0xF00: 4.2}, Dict[str, int])
        with self.assertRaises(TypeError):
            self.mapper.map_with_type({"foo": "bar"}, Dict[str, int])


class TestLoader(BaseMapperMixin, TestCase):
    def setUp(self):
        self.mapper = Loader()

    def test_load_enum(self):
        self.assertEqual(self.mapper.map_with_type("spam", SampleEnum), SampleEnum.SPAM)
        self.assertEqual(
            self.mapper.map_with_type("spam_alias", SampleEnum), SampleEnum.SPAM
        )
        self.assertEqual(self.mapper.map_with_type("ham", SampleEnum), SampleEnum.HAM)
        with self.assertRaises(TypeError):
            self.mapper.map_with_type(None, SampleEnum)
        with self.assertRaises(TypeError):
            self.mapper.map_with_type(0xF00, SampleEnum)

    def test_load_schema_bad_type(self):
        with self.assertRaises(TypeError):
            self.mapper.map_with_type("{}", Schema)
        with self.assertRaises(TypeError):
            self.mapper.map_with_type([], Schema)
        with self.assertRaises(TypeError):
            self.mapper.map_with_type({"foo": "bar"}, Schema)

    def test_load_schema_empty(self):
        self.assertEqual(self.mapper.map_with_type({}, Schema), Schema())

    def test_load_schema(self):
        self.assertEqual(
            self.mapper.map_with_type(
                {
                    "my_schema": {
                        "my_bool": True,
                        "my_int": -2,
                        "my_float": 3.14,
                        # Test mixedCase to lower_case conversion.
                        "myStr": "bar",
                        "my_enum": "eggs",
                        "my_optional_str": None,
                        "my_list": ["spam", "ham"],
                        "my_dict": {"eggs": 6},
                    },
                    "my_list_of_schemas": [{}],
                    "my_dict_of_schemas": {"only": {"my_help_text": "a nice value!"}},
                },
                SampleOuterConfig,
            ),
            SampleOuterConfig(
                my_schema=SampleConfigTypes(
                    my_bool=True,
                    my_int=-2,
                    my_float=3.14,
                    my_str="bar",
                    my_enum=SampleEnum.EGGS,
                    my_optional_str=None,
                    my_list=["spam", "ham"],
                    my_dict={"eggs": 6},
                ),
                my_list_of_schemas=[SampleConfigDefault(my_default=1)],
                my_dict_of_schemas={
                    "only": SampleConfigHelp(my_help_text="a nice value!")
                },
            ),
        )

    def test_load_schema_missing_no_default(self):
        with self.assertRaises(TypeError):
            self.mapper.map_with_type({}, SampleConfigTypes)

    def test_load_schema_excess_field(self):
        with self.assertRaisesRegex(TypeError, "something completely different"):
            self.mapper.map_with_type(
                {"something completely differenty": None}, SampleConfigDefault
            )

    def test_load_schema_bad_types(self):
        with self.assertRaises(TypeError):
            self.mapper.map_with_type(
                {
                    "my_bool": None,
                    "my_int": None,
                    "my_float": None,
                    "my_str": None,
                    "my_enum": None,
                    "my_optional_str": False,
                    "my_list": None,
                    "my_dict": None,
                },
                SampleConfigTypes,
            )

    def test_load_schema_optional_types_are_not_optional_fields(self):
        with self.assertRaisesRegex(TypeError, "my_optional_str"):
            self.mapper.map_with_type(
                {
                    "my_bool": True,
                    "my_int": -2,
                    "my_float": 3.14,
                    "my_str": "bar",
                    "my_enum": "ham",
                    "my_list": ["spam", "ham"],
                    "my_dict": {"eggs": 6},
                },
                SampleConfigTypes,
            )


class TestDumper(BaseMapperMixin, TestCase):
    def setUp(self):
        self.mapper = Dumper()

    def test_dump_enum(self):
        self.assertEqual(self.mapper.map_with_type(SampleEnum.SPAM, SampleEnum), "spam")
        self.assertEqual(
            self.mapper.map_with_type(SampleEnum.SPAM_ALIAS, SampleEnum), "spam"
        )
        self.assertEqual(self.mapper.map_with_type(SampleEnum.HAM, SampleEnum), "ham")
        with self.assertRaises(TypeError):
            self.mapper.map_with_type(None, SampleEnum)
        with self.assertRaises(TypeError):
            self.mapper.map_with_type(0xF00, SampleEnum)

    def test_dump_schema(self):
        self.assertEqual(
            self.mapper.map_with_type(
                SampleOuterConfig(
                    my_schema=SampleConfigTypes(
                        my_bool=True,
                        my_int=-2,
                        my_float=3.14,
                        my_str="bar",
                        my_enum=SampleEnum.EGGS,
                        my_optional_str=None,
                        my_list=["spam", "ham"],
                        my_dict={"eggs": 6},
                    ),
                    my_list_of_schemas=[SampleConfigDefault()],
                    my_dict_of_schemas={
                        "only": SampleConfigHelp(my_help_text="a nice value!")
                    },
                ),
                SampleOuterConfig,
            ),
            {
                "my_schema": {
                    "my_bool": True,
                    "my_int": -2,
                    "my_float": 3.14,
                    "my_str": "bar",
                    "my_enum": "eggs",
                    "my_optional_str": None,
                    "my_list": ["spam", "ham"],
                    "my_dict": {"eggs": 6},
                },
                "my_list_of_schemas": [{"my_default": 1}],
                "my_dict_of_schemas": {"only": {"my_help_text": "a nice value!"}},
            },
        )


class TestConfig(TestCase):
    def test_help(self):
        self.assertEqual(
            SampleOuterConfig.help(),
            [
                "my_types:",
                "",
                "  my_bool (bool)",
                "  my_int (int)",
                "  my_float (float)",
                "  my_str (str)",
                "  my_enum ((spam|ham|eggs))",
                "  my_optional_str (?str)",
                "  my_list ([str])",
                "  my_dict ({str: int})",
                "",
                "",
                "my_default:",
                "",
                "  my_default (int)",
                "",
                "",
                "my_help:",
                "",
                "  my_help_text (str)",
                "\tPass a nice value here!",
                "",
                "",
                "my_outer_name:",
                "",
                "  my_schema (?my_types)",
                "  my_list_of_schemas ([my_default])",
                "  my_dict_of_schemas ({str: my_help})",
                "",
                "",
            ],
        )


class TestExtractNestedType(TestCase):
    def test_empty(self):
        self.assertIs(extract_nested_type(SampleConfigTypes, []), SampleConfigTypes)

    def test_optional(self):
        self.assertIs(
            extract_nested_type(SampleOuterConfig, ["my_schema", "my_optional_str"]),
            str,
        )

    def test_list(self):
        self.assertIs(
            extract_nested_type(
                SampleOuterConfig, ["my_dict_of_schemas", "key", "my_help_text"]
            ),
            str,
        )

    def test_dict(self):
        self.assertIs(
            extract_nested_type(
                SampleOuterConfig, ["my_list_of_schemas", "42", "my_default"]
            ),
            int,
        )


class TestInjectNestedValue(TestCase):
    def test_empty(self):
        with self.assertRaises(ValueError):
            inject_nested_value({"foo": 42}, [], {"bar": 43})

    def test_mixed(self):
        data = {"foo": [{"bar": True, "baz": [42, 43]}]}
        inject_nested_value(data, ["foo", "0", "baz", "1"], 1)
        self.assertEqual(data, {"foo": [{"bar": True, "baz": [42, 1]}]})


if __name__ == "__main__":
    main()
