
"""
The CLI tool to manipulate kernel tuner cache files. Works with the cache file manipulation library (cache.py), conversion functionality
(convert.py) and other helper functions (defined in cli_fct.py).
See the main() function for more detailed inner working.

Basic usage:
$ cli {convert, inspect, merge}

We can:
	  - `convert`: using the functionality from convert.py one can convert to a specified version. Write 'T4' for converting to T4 format.
              Withing conversion (all required yet nonpositional arguments):
                - `-i/--infile`: the input file to read from.
                - `-v/--version`: The version to convert to. Write 'T4' to convert to T4 format.
                - `-o/--output`: The output file.

	   - `inspect`: one can inspect a specific cache file. Arguments are:
                - (Required) `-i, --infile`: The input file to read from.
	        - `-c, --check`: check if a certain cacheline is in the cachefile. I expect this to be very slow as you might have to search through the whole search
		  space. Maybe we can also add --lin? This option would imply the cachelines are ordered low to high in an N-dimensional (N params) space, allowing for binary search; a huge speedup.
		- `-r, --remove`: remove a certain cacheline from the cache file.
		- `-a, --add`: add a list of cachelines to the cachefile, reading it (newline-separated) from file specified by `-f/--file`.
                - `-f, --file`: The input file to read the input cachelines from.
                - `-o, --output`: The (optional) output file to write to. Required if `-r` or `-a` is used.

	  - `merge`: merge several (>=2) cache files that are of the same jsonschema and have equivalent metadata. Arguments are (all required):
                - <files>: The list of (space separated) input files to read in order to merge the cachefiles.
                - `-o, --output`: The output file to write the merged result to.

Example usages:
$ cli convert --infile 1.json -v 1.1.0 -o 2.json
$ cli inspect -i a.json --check 1,1,1,1
$ cli inspect --infile x.json -r 1,1,1,1 -o y.json
$ cli inspect --infile test.json -a --file appendices.txt -o wow.json
$ cli merge 1.json 2.json 3.json 4.json -o merged.json

[*] For now, we run it as a module:
When you are in the main /kernel_tuner directory, start a poetry shell (by running `poetry shell`), and run
`python3.X -m kernel_tuner.cache.cli <options>`
"""

# import required files from within kernel tuner
from .cache import Cache
from .cli_fct import cli_inspect, cli_convert, cli_merge

import argparse


def main():
	"""
	The main function performing the argument parsing and following the user-specified actions.
	We can:
	  - `convert`: using the functionality from convert.py one can convert to a specified version.
	  - `merge`: merge several cache files that are of the same json schema and have equivalent 'tune_params_keys' and 'objective'.
	  - `inspect`: one can inspect a specific cache file. Optional arguments are:
	    - `-c, --check`: check if a certain cacheline is in the cachefile. I expect this to be very slow as you might have to search through the whole search
		  space.
		- `-r, --remove`: remove a certain cacheline from the cache file.
		- `-a, --add`: add a list of cachelines to the cachefile, reading it from file specified by `-f/--file`.
	"""


	# Creates the parser and adds subparsers.

	parser = argparse.ArgumentParser(
		prog="cache_cli",
		description="A CLI tool to manipulate kernel tuner cache files.",
		epilog="More help/issues? Visit https://github.com/kernel_tuner/kernel_tuner")

	sp = parser.add_subparsers(required=True, help="Possible subcommands: 'convert', 'merge' and 'inspect'.")

	convert = sp.add_parser("convert", help="Convert a cache file from one version to another.")
	convert.add_argument("-i", "--infile", required=True, help="The input cache file to read from.")
	convert.add_argument("-o", "--output", required=True, help="The output (JSON) file to write to.")
	convert.add_argument("-v", "--version", required=True, help="The destination version. Write 'T4' for conversion to T4 format.")

	convert.set_defaults(func=cli_convert)


	inspect = sp.add_parser("inspect", help="Inspect inside a certain cachefile.")

	inspect.add_argument("-i", "--infile", required=True, help="The input cachefile to read from.")
	inspect.add_argument("-a", "--append", action="store_true", help="Append cache entries to the cachefile, reading from input file specified by -f/--file.")
	inspect.add_argument("-c", "--check", help="Check if a cacheline exists.")
	inspect.add_argument("-f", "--file", help="The file to work with for the appending functionality.")
	inspect.add_argument("-r", "--remove", help="Remove the provided cacheline from the cachefile.")
	inspect.add_argument("-o", "--output", help="The output file to write to, after appending or removing.")

	inspect.set_defaults(func=cli_inspect)


	merge = sp.add_parser("merge", help="Merge two or more cachefiles.")

	merge.add_argument("files", nargs="+", help="The cachefiles to merge (minimum two). They must be of the same version, and contain equivalent metadata.")
	merge.add_argument("-o", "--output", required=True, help="The output file to write the merged cachefiles to.")

	merge.set_defaults(func=cli_merge)


	# Parse input and call the appropiate function.
	res = parser.parse_args()

	res.func(res)

if __name__ == "__main__":
	main()
