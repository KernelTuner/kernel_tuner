""" Validator of the cache files """
import json
import jsonschema
from datetime import datetime
import sys  # Import the sys module

import jsonschema.exceptions
from .paths import get_schema_path

def validate_data(data_file):
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Get schema path using schema version from the data
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
        print("De invoerdata voldoet aan het schema")
    except jsonschema.exceptions.ValidationError as e:
        print("Fout bij valideren van invoerdata")
        print(e)


