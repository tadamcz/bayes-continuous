dict = {}

dict['prior-family'] = 'lognormal'
dict['prior-lognormal-param1']=1
dict['prior-lognormal-param2']=1

dict['likelihood-family'] = 'normal'
dict['likelihood-normal-param1']=10
dict['likelihood-normal-param2']=1


s = 'localhost:5000/?'

for x in dict:
    s += str(x) + '=' + str(dict[x]) + '&'
print(s)
