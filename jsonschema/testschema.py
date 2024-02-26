import json, jsonschema
import sys

# Ffor now we use "exampleFile.json"
# as default datafile and "cacheschema.json" as default schemafile, if no
# two arguments are provided.
# You may replace with your own if you want.

if (len(sys.argv) == 1):
    dfName = "exampleFile.json"
    sfName = "cacheschema.json"

elif (len(sys.argv) != 3):
    print("Usage: {} <datafile name> <schemafile name>".format(sys.argv[0]))
    exit()

else:
    dfName = sys.argv[1]
    sfName = sys.argv[2]

with open(dfName) as f:
    dataFile = json.load(f)

# Do the same for the schema
with open("cacheschema.json") as f:
    dataSchema = json.load(f)


try:
    jsonschema.validate(instance=dataFile, schema=dataSchema)
    print("\n[*] Validation on file {} and JSON schema {} succesful;\
 valid JSON schema and instance.\n".format(dfName, sfName))

except jsonschema.exceptions.ValidationError:
    print("\n[!] Error in JSON file {}.\n".format(str(dfName)))

except jsonschema.exceptions.SchemaError:
    print("\n[!] Error in JSON schema {}.\n".format(str(sfName)))
