import os

from dataclasses import dataclass, field
from parameters import parameter
from typing import List

#print(os.environ['COMMANDER_PARAMS_DEFAULT'])

def read_in_paramfile(pfile,params):

    with open(pfile) as f:
        for line in f:
            if line.startswith('#') or line.startswith('\n') or line.startswith(' ') or line.startswith('*'):
                continue
            elif line.startswith('@'):
                # We will add some "reading the defaults files" thing here
                continue
            else:
                new_line = parse_line(line)
                line_list = list(new_line.items())

                if len(new_line) > 1:
                    key, val = line_list[0]
                    comment = line_list[1][1]
                    key = key.strip()
                    val = val.strip()
                    params.append(parameter(cpar=key,value=val,comment=comment))

                else:
                    key, val = line_list[0]
                    key = key.strip()
                    val = val.strip()
                    params.append(parameter(cpar=key,value=val))

def parse_line(line):
    line_dict = {}
    comments = ''
    if '#' in line:
        parameters, comments = line.split('#')[:2]
    else:
        parameters = line
    key, val = parameters.split('=')

    line_dict[key] = val
    if comments != '':
        line_dict['comment'] = comments

    return line_dict

def main():
    params: List[parameter] = []
    read_in_paramfile('param_test.txt',params)
    print(type(params[0]))

if __name__ == '__main__':
    main()