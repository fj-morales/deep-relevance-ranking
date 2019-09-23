import subprocess

def eval(trec_eval_command, qrel, qret):
    
    metrics = '-m map -m P.20 -m ndcg_cut.20'
    toolkit_parameters = [
                            trec_eval_command,
                            '-m',
                            'map',
                            '-m',
                            'P.20',
                            '-m',
                            'ndcg_cut.20',
                            qrel,
                            qret]

#     print(toolkit_parameters)

    proc = subprocess.Popen(toolkit_parameters, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
    (out, err) = proc.communicate()
#     print(out.decode("utf-8"))
    print('Run error: ', err)
    if err == None:
        pass
#         print('No errors')
    out_split = out.decode("utf-8").replace('\tall\t','').splitlines()
    out_dict = {item.split()[0]:item.split()[1] for item in out_split}
    return out_dict