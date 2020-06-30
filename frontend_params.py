from decimal import Decimal
form_dictionary = {'prior': {'family': 'lognormal', 'normal': {'param1': Decimal('1.00'), 'param2': Decimal('2.00'), 'csrf_token': 'IjZjOTIzNWU4NTBjMzkwZjI1N2Q0NGExYTU4NzhmNDViNmY0MmMxNTEi.XvrPQw.wU3dUfMQGjswTndeAYoshnYIyGI'}, 'lognormal': {'param1': Decimal('1'), 'param2': Decimal('1'), 'csrf_token': 'IjZjOTIzNWU4NTBjMzkwZjI1N2Q0NGExYTU4NzhmNDViNmY0MmMxNTEi.XvrPQw.wU3dUfMQGjswTndeAYoshnYIyGI'}, 'beta': {'param1': None, 'param2': None, 'csrf_token': 'IjZjOTIzNWU4NTBjMzkwZjI1N2Q0NGExYTU4NzhmNDViNmY0MmMxNTEi.XvrPQw.wU3dUfMQGjswTndeAYoshnYIyGI'}, 'uniform': {'param1': None, 'param2': None, 'csrf_token': 'IjZjOTIzNWU4NTBjMzkwZjI1N2Q0NGExYTU4NzhmNDViNmY0MmMxNTEi.XvrPQw.wU3dUfMQGjswTndeAYoshnYIyGI'}, 'csrf_token': ''}, 'likelihood': {'family': 'normal', 'normal': {'param1': Decimal('2.00'), 'param2': Decimal('3.00'), 'csrf_token': 'IjZjOTIzNWU4NTBjMzkwZjI1N2Q0NGExYTU4NzhmNDViNmY0MmMxNTEi.XvrPQw.wU3dUfMQGjswTndeAYoshnYIyGI'}, 'lognormal': {'param1': None, 'param2': None, 'csrf_token': 'IjZjOTIzNWU4NTBjMzkwZjI1N2Q0NGExYTU4NzhmNDViNmY0MmMxNTEi.XvrPQw.wU3dUfMQGjswTndeAYoshnYIyGI'}, 'beta': {'param1': None, 'param2': None, 'csrf_token': 'IjZjOTIzNWU4NTBjMzkwZjI1N2Q0NGExYTU4NzhmNDViNmY0MmMxNTEi.XvrPQw.wU3dUfMQGjswTndeAYoshnYIyGI'}, 'uniform': {'param1': None, 'param2': None, 'csrf_token': 'IjZjOTIzNWU4NTBjMzkwZjI1N2Q0NGExYTU4NzhmNDViNmY0MmMxNTEi.XvrPQw.wU3dUfMQGjswTndeAYoshnYIyGI'}, 'csrf_token': ''}, 'graphrange': {'param1': None, 'param2': None, 'csrf_token': 'IjZjOTIzNWU4NTBjMzkwZjI1N2Q0NGExYTU4NzhmNDViNmY0MmMxNTEi.XvrPQw.wU3dUfMQGjswTndeAYoshnYIyGI'}, 'csrf_token': ''}

def recursively_remove_csrf(dictionary):
    dictionary.pop('csrf_token')
    for key in dictionary:
        if type(dictionary[key]) is dict:
            recursively_remove_csrf(dictionary[key])

recursively_remove_csrf(form_dictionary)

s = 'localhost:5000/?'
s += 'data=' + str(form_dictionary)

print(s)