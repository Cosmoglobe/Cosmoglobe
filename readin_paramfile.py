import os

from dataclasses import dataclass, field
#from . import parameters

#print(os.environ['COMMANDER_PARAMS_DEFAULT'])

def read_in_paramfile(pfile):

    pfile_dict = {}

    with open(pfile) as f:
        for line in f:
            if line.startswith('#') or line.startswith('\n') or line.startswith(' ') or line.startswith('*'):
                continue
            elif line.startswith('@'):
                # We will add some "reading the defaults files" thing here
                continue
            else:
                print(line)
                new_line = parse_line(line)
                line_list = list(new_line.items())

                if len(new_line) > 1:
                    key, val = line_list[0]
                    pfile_dict[key] = val
                else:
                    key, val = line_list[0]
                    pfile_dict[key] = val

    return pfile_dict


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
    paramfile = read_in_paramfile('param_test.txt')
    for key, val in paramfile:
        print(key,val)

if __name__ == '__main__':
    main()