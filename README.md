# pfile_modifier
First steps in creating a python tool for reading in, adjusting, and writing Commander3 parameter files.

Commander3 currently relies on default files in order to keep the main parameter file short enough
to be able to readily read. In order for Commander3 to read these default files, the user has to define
a bash environment variable called COMMANDER_PARAMS_DEFAULT.

Ensure that this variable has been defined in your bash environment before attempting to run or write
any code here. We will call this a requirement.