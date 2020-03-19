import numpy as np
import argparse
import json
from pynoisy import qmetric_functions as qm
from tqdm import tqdm
from joblib import Parallel, delayed
import glob
import ehtim as eh
import os
import pandas as pd


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

def parse_arguments():
    """Parse the command-line arguments for each run.
    The arguemnts are split into general, envelope and evolution related arguments.

    Returns:
        parser (argparse.parser): the argument parser.
        args (Namespace): an argparse Namspace object with all the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--eht_home',
                        default='/home/aviad/Code/eht-imaging/',
                        help='(default value: %(default)s) ehtim home directory.')
    parser.add_argument('--uvfolder',
                        help='Path to sgr uvfits file. Relative within the eht home directory.')
    parser.add_argument('--date',
                        type=int,
                        default=3599,
                        help='(default value: %(default)s) Day of synthetic measurement.')
    parser.add_argument('--n_jobs',
                        type=int,
                        default=1,
                        help='Number of jobs to parallelize computations.')

    # Q-metric parameters
    qmetric = parser.add_argument_group('Qmetric parameters')
    qmetric.add_argument('--tavg',
                        type=float,
                        default=120.0,
                        help='(default value: %(default)s) Averaging time in seconds')
    qmetric.add_argument('--bintime',
                        type=float,
                        default=0.0,
                        help='(default value: %(default)s) Bin time for unevenly sampled data')
    qmetric.add_argument('--segtime',
                        type=float,
                        default=0.0,
                        help='(default value: %(default)s) Split into segments of segtime before detrending and calculating Q')
    qmetric.add_argument('--diftime',
                        type=float,
                        default=0.0,
                        help='(default value: %(default)s) Differencing time')
    qmetric.add_argument('--detrend_deg',
                        type=int,
                        default=3,
                        help='(default value: %(default)s) Degree of detrending polynomial')

    args = parser.parse_args()
    return parser, args

def params_from_uvfile(uvfile, parameters):
    """Get run parameters from uvfile
    """
    envelopes = np.array([envelope.split('/')[-1][:-5] for envelope in parameters.pop('envelope')])
    params =  {'envelope': envelopes[np.array([uvfile.find(envelope) for envelope in envelopes]) > 0][0]}
    for param in parameters.keys():
        params[param] = uvfile.split('.uvfits')[0].split(param)[1].split('_')[0]
    return params

def zero_baseline_std(obs):
    tmp = obs.unpack(['u', 'v', 'amp'], debias=True)
    amp, u, v = tmp['amp'], tmp['u'], tmp['v']
    return np.std(amp[np.sqrt(u**2 + v**2) < 1e8])

def compute_qmetric_df(uvfile, date, tavg, bintime, segtime, diftime, detrend_deg, parameters):
    """Compute a qmetric dataframe
    """
    obs = eh.obsdata.load_uvfits(uvfile)

    # Average
    obs = obs.avg_coherent(tavg)

    # Calculate closure phases
    obs_cphases = obs.c_phases()

    # List nontrivial triangles
    triangle_list = []
    for i in range(len(obs_cphases)):
        triangle = [obs_cphases['t1'][i], obs_cphases['t2'][i], obs_cphases['t3'][i]]

        if 'AA' in triangle and 'AP' in triangle:
            continue
        if 'JC' in triangle and 'SM' in triangle:
            continue

        if triangle not in triangle_list:
            triangle_list.append(triangle)

    zbl_std = zero_baseline_std(obs)
    qmetric = {'triangle': [], 'date': [date]*len(triangle_list), 'q': [], 'dq': [], 'zbl_std': [zbl_std]*len(triangle_list)}
    params = params_from_uvfile(uvfile, parameters)
    for key, value in params.items():
        qmetric[key] = [value]*len(triangle_list)

    # Calculate Q for measured closure phases
    for triangle in triangle_list:
        this_cphases = obs.cphase_tri(triangle[0], triangle[1], triangle[2])
        time = this_cphases['time']
        cp = this_cphases['cphase']
        err = this_cphases['sigmacp']
        q, dq = qm.qmetric(
            time, cp, err, bintime=bintime, segtime=segtime, diftime=diftime, detrend_deg=detrend_deg
        )
        qmetric['triangle'].append('{}-{}-{}'.format(*triangle))
        qmetric['q'].append(q)
        qmetric['dq'].append(dq)

    return pd.DataFrame(qmetric)

def split_arguments(parser, args):
    """Split arguments into general and qmetric related arguments.

    Returns:
        qmetric_params (dict): dictionary with all the qmetric parameters.
    """
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)

    qmetric_args = arg_groups['Qmetric parameters'].__dict__
    return qmetric_args

if __name__ == "__main__":

    parser, args = parse_arguments()

    uvfolder = os.path.join(args.eht_home, args.uvfolder)
    uvfolder = uvfolder + '/' if uvfolder[-1] != '/' else uvfolder
    uvfiles = glob.glob(uvfolder + '*.uvfits')

    # Load parameters
    with open(os.path.join(uvfolder, 'args.txt'), 'r') as file:
        parameters = json.load(file)
    parameters.pop('fits_path')

    if args.n_jobs == 1:
        qmetric_df = [
            compute_qmetric_df(file, args.date, args.tavg, args.bintime, args.segtime, args.diftime,
                               args.detrend_deg, parameters) for file in uvfiles
        ]

    else:
        qmetric_df = Parallel(n_jobs=args.n_jobs)(delayed(compute_qmetric_df)(
            file, args.date, args.tavg, args.bintime, args.segtime,
            args.diftime, args.detrend_deg, parameters) for file in tqdm(uvfiles))

    qmetric_args = split_arguments(parser, args)
    output_path = os.path.join(uvfolder, 'qmetric_{}'.format(args.date) +
        ''.join(['_{}{}'.format(key, value) for key, value in qmetric_args.items()]) + '.csv')
    qmetric_df = pd.concat(qmetric_df)
    qmetric_df.to_csv(output_path)