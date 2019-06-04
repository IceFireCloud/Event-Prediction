import subprocess

def function(job_id, n_epochs, l_r, dropout_rate, num_filters, conv_layers, first_pool, second_pool, l1_reg, train_end):
    command = ['python3', '1D_spearmint.py', str(job_id),
                                            str(n_epochs),
                                            str(l_r),
                                            str(dropout_rate),
                                            str(num_filters),
                                            str(conv_layers),
                                            str(first_pool),
                                            str(second_pool),
                                            str(l1_reg),
                                            str(train_end)]

    output = subprocess.Popen(command, stdout=subprocess.PIPE)
    val = str(output.communicate()[0])
    val = val[val.index('$')+1:val.index('$$')]
    return float(val)

def main(job_id, params):
    print(params)
    return function(job_id,
                    params['n_epochs'],
                    params['l_r'],
                    params['dropout_rate'],
                    params['num_filters'],
                    params['conv_layers'],
                    params['first_pool'],
                    params['second_pool'],
                    params['l1_reg'],
                    params['train_end'])
