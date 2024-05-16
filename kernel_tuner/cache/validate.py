""" Validator of the cache files """
import json
import jsonschema
from datetime import datetime
import sys  

import jsonschema.exceptions
from .paths import get_schema_path

def validate_data(data_file):
    """
    Validate the input data in a cache file against its corresponding schema.

    Parameters:

    data_file (str): The path to the cache file that needs to be validated.
    Raises:

    ValueError: If the cache file does not have a "schema_version" field, or if the version does not match
    a valid version.
    FileNotFoundError: If the schema file cannot be found at the specified path.
    RuntimeError: If an error occurs while retrieving the schema path.
    """
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    try:
        schema_path = get_schema_path(data["schema_version"])
    except KeyError:
        raise ValueError(f"Schema version '{data['schema_version']}' not found in the data.")
    except FileNotFoundError as e:
        print(f"Schema file not found for version '{data['schema_version']}': {e}")
        exit(1)  
    except Exception as e:
        raise RuntimeError(f"An error occurred while retrieving the schema path: {e}")

    try:
        with open(schema_path, 'r') as f:
            schema = json.load(f)
    except FileNotFoundError as e:
        print(f"Schema file not found at path '{schema_path}': {e}")
        exit(1) 

    format_checker = jsonschema.FormatChecker()
    
    @format_checker.checks("date-time")
    def _check_iso_datetime(instance):
        try:
            datetime.fromisoformat(instance)
            return True
        except ValueError:
            return False
    
    try:
        jsonschema.validate(instance=data, schema=schema, format_checker=format_checker)
        print("The input data conforms to the schema.")
    except jsonschema.exceptions.ValidationError as e:
        print("Error validating input data")
        print(e)


