from parameter_parser import ParameterParser


def test():
    param_parser = ParameterParser('param_test.txt', 'defaults')
    gen_params = param_parser.create_gen_params()
    print(gen_params)

if __name__ == '__main__':
    test()
