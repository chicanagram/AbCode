import os
import platform
opsys = platform.system()
import numpy as np
from project_config.variables import aaList

def sort_list(lst):
    lst.sort()
    return lst

def mkDir(res, output_dir, remove_existing_dir=True):
    import shutil
    # making new directory
    new_dir = (output_dir + res)
    if os.path.exists(new_dir):
        # remove if directory exists, and make new directory
        if remove_existing_dir:
            shutil.rmtree(new_dir)
            os.makedirs(new_dir)
    else:
        os.makedirs(new_dir)
    return new_dir

def findProcess(process_name):
    if opsys=='Windows':
        return [int(item.split()[1]) for item in os.popen('tasklist').read().splitlines()[4:] if process_name in item.split()]
    elif opsys=='Linux' or opsys=='Darwin':
        return [int(pid) for pid in os.popen('pidof '+process_name).read().strip(' \n').split(' ')]

def exit_program(pid):
    import signal
    print("Sending SIGINT to self...")
    os.kill(pid, signal.SIGINT)
    print('Exited program', pid)
def is_float(string):
    if string.replace(".", "").replace("-", "").isnumeric():
        return True
    else:
        return False

def save_dict_as_csv(datadict, cols, log_fpath, csv_suffix ='', multiprocessing_proc_num=None):
    # save results as CSV
    csv_txt = ''
    # get csv_suffix if running multiprocessing
    if multiprocessing_proc_num is not None:
        csv_suffix += '_' + str(multiprocessing_proc_num)

    # check if file exists yet
    log_fpath_full = log_fpath + csv_suffix + '.csv'
    if not os.path.exists(log_fpath_full):
        # if not, start a new file with headers
        write_mode = 'w'
        csv_txt += ','.join(cols) + '\n'
    else:
        write_mode = 'a'

    # convert dict of lists to list of dicts
    if isinstance(datadict[cols[0]], list):
        num_rows = len(datadict[cols[0]])
        datadict_byrow = []
        for row_idx in range(num_rows):
            row = []
            for col in cols:
                row.append(datadict[col][row_idx])
            datadict_byrow.append(row)
    else:
        row = []
        for col in cols:
            row.append(datadict[col])
        datadict_byrow = [row]

    # add data to csv file
    for row in datadict_byrow:
        csv_txt += ','.join([str(el) for el in row])
        csv_txt += '\n'
    # save the changes
    with open(log_fpath_full, write_mode) as f:
        f.write(csv_txt)
    return csv_txt, log_fpath_full, write_mode

def combine_csv_files(log_fpath_list, output_dir, output_fname, remove_combined_files=True):
    # combine files spawned
    txt_all_list = []
    for i, log_fpath in enumerate(log_fpath_list):
        with open(log_fpath, 'r') as f:
            if i==0:
                txt_all_list += f.readlines()
            else:
                txt_all_list += f.readlines()[1:]
    if os.path.exists(output_dir + output_fname + '.csv'):
        write_mode = 'a'
        txt_all_list = txt_all_list[1:]
    else:
        write_mode = 'w'
    # get text string to write
    txt_all = '\n'.join(txt_all_list)
    txt_all = txt_all.replace('\n\n', '\n').replace(',\n', '\n')
    # update or save file
    with open(output_dir + output_fname + '.csv', write_mode) as f:
        f.write(txt_all)
    # remove combined files
    if remove_combined_files:
        for log_fpath in log_fpath_list:
            os.remove(log_fpath)
    return txt_all


def flatten_2D_arr(arr2D, seq, MT_aa=aaList):
    """
    arr2D is a 2-dimensional matrix
        axis 0 (vertical): 20 amino acids along axis 0
        axis 1 (horizontal): sequence positions
    """
    if not isinstance(arr2D, np.ndarray):
        WT_res = [seq[pos-1]+str(pos) for pos in arr2D.columns.tolist()]
        arr2D = arr2D.to_numpy()
    else:
        WT_res = [seq[pos - 1] + str(pos) for pos in list(range(1, arr2D.shape[1] + 1))]

    arr1D = arr2D.flatten('F')
    mutations = [wt + mt for wt in WT_res for mt in MT_aa]
    return arr1D, mutations