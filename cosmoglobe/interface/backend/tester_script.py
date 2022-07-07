from parameter_parser import ParameterParser
from parameter_writer import ParameterWriter


def test():
    param_parser = ParameterParser('param_test.txt', 'defaults')
    gen_params = param_parser.classify_params()
    param_writer = ParameterWriter(gen_params)
    param_writer.write_paramfile('param_out.txt')

#    print(gen_params)

if __name__ == '__main__':
    test()
