"""
The module is aimed to validate input json data according to various schemas.
"""
from pathlib import Path
import json
import os
from jsonschema import validate, RefResolver, Draft4Validator

package_directory = os.path.dirname(os.path.abspath(__file__))


class SpecificationValidator:
    """
    todo
    """

    def __init__(self):
        schemas = (json.load(open(source))
                   for source in Path(os.path.join(package_directory, "schemas")).iterdir())
        self.schema_store = {schema["$id"]: schema for schema in schemas}

    def validate(self, instance, schema_id):
        resolver = RefResolver.from_schema(
            self.schema_store[schema_id], store=self.schema_store)
        validator = Draft4Validator(
            self.schema_store[schema_id], resolver=resolver)
        validator.validate(instance)
