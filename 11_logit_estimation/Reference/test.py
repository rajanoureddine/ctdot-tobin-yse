dict_X2 = {'broad_ev': ['urban'],
               'prices': ['I(1/income)'],
               '1': ['rc']}

str_x2_formulation = '1'

for x2 in dict_X2:
        if x2 != '1':
            str_x2_formulation = str_x2_formulation + '+' + x2

print(str_x2_formulation)


lst_demogs = list()
for x2 in dict_X2:
        lst_new = dict_X2[x2]
        lst_new = [x for x in lst_new if (x != 'rc') and not( x in lst_demogs)]
        lst_demogs.extend(lst_new)

print(lst_demogs)

lst_demogs = ['1'] + lst_demogs
# add the mid/high if they are interacted with something
if any(item in ['I(mid/income)', 'I(1/income)'] for item in lst_demogs):
    lst_demogs = lst_demogs + ['mid','high']
for d in lst_demogs:
    if d == '1':
        str_d_formulation =  d
    else:
        str_d_formulation = str_d_formulation + '+' + d
# correctly ordered list of demographics
lst_x2_full = str_x2_formulation.split('+')

print(lst_x2_full)


