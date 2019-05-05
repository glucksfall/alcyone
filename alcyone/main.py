# -*- coding: utf-8 -*-

'''
Project "Bootstrapping of parameter values", Rodrigo Santibáñez, 2019 @ NBL, UMayor
Wrapper around pleione to determine confidence interval of parameter values
Citation:
'''

__author__  = 'Rodrigo Santibáñez'
__license__ = 'gpl-3.0'
__software__ = 'pleione-v1.0+'

import argparse, glob, multiprocessing, os, random, re, shutil, subprocess, sys, time
import pandas, numpy

def safe_checks():
	error_msg = ''
	if shutil.which(opts['python']) is None:
		error_msg += 'python3 (at {:s}) can\'t be called to perform error calculation.\n' \
			'You could use --python {:s}\n'.format(opts['python'], shutil.which('python3'))

	# check for simulators
	if opts['soft'].lower() == 'bng2' and shutil.which(opts['bng2']) is None:
		error_msg += 'BNG2 (at {:s}) can\'t be called to perform simulations.\n' \
			'Check the path to BNG2.\n'.format(opts['bng2'])

	elif opts['soft'].lower() == 'kasim' and shutil.which(opts['kasim']) is None:
		error_msg += 'KaSim (at {:s}) can\'t be called to perform simulations.\n' \
			'Check the path to KaSim.\n'.format(opts['kasim'])

	elif opts['soft'].lower() == 'nfsim' and shutil.which(opts['nfsim']) is None:
		error_msg += 'NFsim (at {:s}) can\'t be called to perform simulations.\n' \
			'Check the path to NFsim.\n'.format(opts['nfsim'])

	elif opts['soft'].lower() == 'piskas' and shutil.which(opts['piskas']) is None:
		error_msg += 'PISKaS (at {:s}) can\'t be called to perform simulations.\n' \
			'Check the path to PISKaS.\n'.format(opts['piskas'])

	# check for slurm
	if opts['slurm'] is not None or opts['slurm'] == '':
		if not sys.platform.startswith('linux'):
			error_msg += 'SLURM do not support WindowsOS and macOS (https://slurm.schedmd.com/platforms.html)\n'
		else:
			if shutil.which('sinfo') is None:
				error_msg += 'You specified a SLURM partition but SLURM isn\'t installed on your system.\n' \
					'Delete --slurm to use the python multiprocessing API or install SLURM (https://pleione.readthedocs.io/en/latest/SLURM.html)\n'
			else:
				cmd = 'sinfo -hp {:s}'.format(opts['slurm'])
				cmd = re.findall(r'(?:[^\s,"]|"+(?:=|\\.|[^"])*"+)+', cmd)
				out, err = subprocess.Popen(cmd, shell = False, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()
				if out == b'':
					error_msg += 'You specified an invalid SLURM partition.\n' \
						'Please, use sinfo to know available $SLURM_JOB_PARTITION to set --slurm or delete --slurm to use the python multiprocessing API.\n'

	# check if model file exists
	if not os.path.isfile(opts['model']):
		error_msg += 'The "{:s}" file cannot be opened.\n' \
			'Please, check the path to the model file.\n'.format(opts['model'])

	# check if data files exist
	if len(opts['data']) == 1: # shlex
		if len(glob.glob(opts['data'][0])) == 0:
			error_msg += 'The path "{:s}" is empty.\n' \
				'Please, check the path to the data files.\n'.format(opts['data'][0])
	else:
		for data in opts['data']: # each file was declared explicitly
			if not os.path.isfile(data):
				error_msg += 'The "{:s}" file cannot be opened.\n' \
					'Please, check the path to the data file.\n'.format(data)

	# check GA options
	if opts['mut_swap'] > 1.0:
		error_msg += 'Parameter swap (or recombination) probability must be a float between zero and one.\n'

	if opts['mut_rate'] > 1.0:
		error_msg += 'Parameter mutation probability must be a float between zero and one.\n'

	# print error
	if error_msg != '':
		print(error_msg)
		raise ValueError(error_msg)

	return 0

def parallelize(cmd):
	proc = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
	out, err = proc.communicate()
	proc.wait()
	return 0

def argsparser():
	parser = argparse.ArgumentParser(description = 'Perform a bootstrapped calibration of a RBM employing a Genetic Algorithm.')

	# required arguments for alcyone
	parser.add_argument('--runs'   , metavar = 'int'  , type = int  , required = True , nargs = 1  , help = 'number of bootstrapping runs')
	parser.add_argument('--nobs'   , metavar = 'int'  , type = int  , required = True , nargs = 1  , help = 'number of replications to draw from original data')
	parser.add_argument('--soft'   , metavar = 'str'  , type = str  , required = True , nargs = 1  , help = 'one of the compatible stochastic software: bng2, kasim4, nfsim, piskas')

	# required arguments for pleione
	parser.add_argument('--model'  , metavar = 'str'  , type = str  , required = True  , nargs = 1  , help = 'RBM with tagged variables to parameterize')
	parser.add_argument('--final'  , metavar = 'float', type = str  , required = True  , nargs = 1  , help = 'limit time to simulate')
	parser.add_argument('--steps'  , metavar = 'float', type = str  , required = True  , nargs = 1  , help = 'time steps to simulate')
	# choose one or more fitness functions
	parser.add_argument('--error'  , metavar = 'str'  , type = str  , required = True  , nargs = '+', help = 'list of supported fit functions')
	parser.add_argument('--data'   , metavar = 'str'  , type = str  , required = True  , nargs = '+', help = 'data files to parameterize')

	# useful paths for simulators
	parser.add_argument('--bng2'   , metavar = 'path' , type = str  , required = False, default = '~/bin/bng2'    , help = 'BioNetGen path, default ~/bin/bng2')
	parser.add_argument('--kasim'  , metavar = 'path' , type = str  , required = False, default = '~/bin/kasim4'  , help = 'KaSim path, default ~/bin/kasim4')
	parser.add_argument('--nfsim'  , metavar = 'path' , type = str  , required = False, default = '~/bin/nfsim'   , help = 'NFsim path, default ~/bin/nfsim')
	parser.add_argument('--piskas' , metavar = 'path' , type = str  , required = False, default = '~/bin/piskas'  , help = 'PISKaS path, default ~/bin/piskas')
	parser.add_argument('--python' , metavar = 'path' , type = str  , required = False, default = '~/bin/python3' , help = 'python path, default ~/bin/python3')

	# distribute computation with SLURM, otherwise with python multiprocessing API
	parser.add_argument('--slurm'  , metavar = 'str'  , type = str  , required = False, default = None            , help = 'SLURM partition to use, default None')

	# general options for pleione
	parser.add_argument('--seeds'  , metavar = 'list' , type = int  , required = False, default = [], nargs = '+' , help = 'random number generator seeds, default empty list')
	parser.add_argument('--iter'   , metavar = 'int'  , type = int  , required = False, default = 100             , help = 'number of iterations, default 100')
	parser.add_argument('--inds'   , metavar = 'int'  , type = int  , required = False, default = 100             , help = 'number of individuals per iteration, default 100')
	parser.add_argument('--sims'   , metavar = 'int'  , type = int  , required = False, default = 10              , help = 'number of simulations per individual, default 100')
	parser.add_argument('--best'   , metavar = 'int'  , type = int  , required = False, default = 10              , help = 'size of elite individuals, default 10.')
	parser.add_argument('--swap'   , metavar = 'float', type = float, required = False, default = 0.50            , help = 'Q1: global parameter swap probability, default 0.5')
	parser.add_argument('--cross'  , metavar = 'str'  , type = str  , required = False, default = 'multiple'      , help = 'Type of crossover: multiple or single point, default multiple')
	parser.add_argument('--rate'   , metavar = 'float', type = float, required = False, default = 0.50            , help = 'Q2: global parameter mutation probability, default 0.5')
	parser.add_argument('--dist'   , metavar = 'str'  , type = str  , required = False, default = 'inverse'       , help = 'parent selection inverse|uniform, default inverse')
	parser.add_argument('--self'   , metavar = 'False', type = str  , required = False, default = False           , help = 'self recombination True|False, default False')
	parser.add_argument('--crit'   , metavar = 'path' , type = str  , required = False, default = None            , help = 'table of Mann-Whitney U-test critical values, default None')
	parser.add_argument('--prec'   , metavar = 'str'  , type = str  , required = False, default = '7g'            , help = 'precision and format of parameter values, default 7g')

	# other options for pleione
	parser.add_argument('--syntax' , metavar = 'str'  , type = str  , required = False, default = '4'             , help = 'KaSim syntax, default 4')
	parser.add_argument('--equil'  , metavar = 'float', type = float, required = False, default = 0               , help = 'equilibrate model before running the simulation, default 0')
	parser.add_argument('--sync'   , metavar = 'float', type = str  , required = False, default = '1.0'           , help = 'time period to syncronize compartments, default 1.0')
	parser.add_argument('--output' , metavar = 'str'  , type = str  , required = False, default = 'outmodels'     , help = 'ranking files prefixes, default outmodels')
	parser.add_argument('--results', metavar = 'str'  , type = str  , required = False, default = 'results'       , help = 'output folder where to move the results, default results')
	parser.add_argument('--parsets', metavar = 'str'  , type = str  , required = False, default = 'individuals'   , help = 'folder to save the generated models, default individuals')
	parser.add_argument('--rawdata', metavar = 'str'  , type = str  , required = False, default = 'simulations'   , help = 'folder to save the simulations, default simulations')
	parser.add_argument('--fitness', metavar = 'str'  , type = str  , required = False, default = 'goodness'      , help = 'folder to save the goodness of fit, default goodness')
	parser.add_argument('--ranking', metavar = 'str'  , type = str  , required = False, default = 'ranking'       , help = 'folder to save the ranking summaries, default ranking')

	# TO BE DEPRECATED, only with publishing purposes.
	# the random standard library does not have a random.choice with an optional probability list, therefore, Pleione uses numpy.random.choice
	parser.add_argument('--legacy' , metavar = 'True' , type = str  , required = False, default = False           , help = 'use True: random.random instead of False: numpy.random')
	# If the user wants to know the behavior of other functions, the option --dev should be maintained
	parser.add_argument('--dev'    , metavar = 'True' , type = str  , required = False, default = False           , help = 'calculate all fitness functions, default False')

	args = parser.parse_args()

	if args.crit is None:
		if set(args.error).issuperset(set(['MWUT'])):
			parser.error('--error MWUT requires --crit file')
		if args.dev:
			parser.error('--dev requires --crit file')
		args.crit = 'dummy-file.txt' # the file is not read by the error calculation script

	if len(args.seeds) < args.runs[0]:
		if sys.platform.startswith('linux'):
			for idx in range(len(args.seeds), args.runs[0]):
				args.seeds.append(int.from_bytes(os.urandom(4), byteorder = 'big'))
		else:
			parser.error('pleione requires --seed list of integers equal to the number of runs')

	if args.legacy and args.dist == 'inverse':
		parser.error('legacy uses the random standard library that don\'t support a non-uniform random choice.\n' \
			'Please delete legacy or set to False.')

	return args

def ga_opts():
	return {
		# alcyone
		'runs'      : args.runs[0],
		'nobs'      : args.nobs[0],
		'soft'      : args.soft[0],
		# pleione
		# user defined options
		'model'     : args.model[0],
		'final'     : args.final[0], # not bng2
		'steps'     : args.steps[0], # not bng2
		'error'     : args.error,
		'data'      : args.data,
		'bng2'      : os.path.expanduser(args.bng2), # bng2, nfsim only
		'kasim'     : os.path.expanduser(args.kasim), # kasim4 only
		'piskas'    : os.path.expanduser(args.piskas), # piskas only
		'nfsim'     : os.path.expanduser(args.nfsim), # nfsim only
		'python'    : os.path.expanduser(args.python),
		'slurm'     : args.slurm,
		'rng_seed'  : args.seeds,
		'num_iter'  : args.iter,
		'pop_size'  : args.inds,
		'num_sims'  : args.sims,
		'pop_best'  : args.best,
		'mut_swap'  : args.swap,
		'mut_rate'  : args.rate,
		'dist_type' : args.dist,
		'self_rec'  : args.self,
		'xpoints'   : args.cross,
		'crit_vals' : args.crit,
		'par_fmt'   : args.prec,
		'syntax'    : args.syntax, # kasim4 only
		'equil'     : args.equil, # nfsim only
		'sync'      : args.sync, # piskas only
		'outfile'   : args.output,
		'results'   : args.results,
		'parsets'   : args.parsets,
		'rawdata'   : args.rawdata,
		'fitness'   : args.fitness,
		'ranking'   : args.ranking,
		# non-user defined options
		'home'      : os.getcwd(),
		'null'      : '/dev/null',
		'systime'   : str(time.time()).split('.')[0],
		}

def bootstrapper():
	# read input data ...
	data = []
	for infile in opts['data']:
		with open(infile, 'r') as infile:
			if opts['soft'] == 'kasim':
				data.append(pandas.read_csv(infile, delimiter = ',', header = 0, engine = 'python').set_index('[T]', drop = False).rename_axis(None, axis = 0).drop('[T]', axis = 1))
	# and concatenate data in a single dataframe
	data = pandas.concat(data, keys = range(len(data))).reorder_levels([1,0])

	# save new experimental data to subdirectories
	for idx1 in range(opts['runs']):
		try:
			os.mkdir('bootstrapping_run{:02d}'.format(idx1))
		except:
			continue

		# select observations with replacement ...
		subsamples = []
		for idx2 in list(data.index.levels[0]):
			subsamples.append(data.loc[idx2].sample(n = opts['nobs'], replace = True).reset_index(drop = True))
		# and reconcatenate in a single dataframe
		samples = pandas.concat(subsamples, keys = data.index.levels[0]).reorder_levels([1,0])

		for idx2 in list(samples.index.levels[0]):
			with open('./bootstrapping_run{:02d}/subsample_{:02d}.txt'.format(idx1, idx2), 'w+') as outfile:
				tmp = samples.loc[idx2]

				if opts['soft'] == 'kasim':
					tmp.index.name = '[T]'
					tmp.to_csv(outfile, sep = ',')

	return data, samples

def callibration():
	job_desc = {
		'nodes'     : 1,
		'ntasks'    : 1,
		'ncpus'     : 1,
		'null'      : opts['null'],
		'partition' : opts['slurm'],
		'job_name'  : 'child_{:s}'.format(opts['systime']),
		'stdout'    : 'stdout_{:s}.txt'.format(opts['systime']),
		'stderr'    : 'stderr_{:s}.txt'.format(opts['systime']),
		}

	# submit simulations to the queue
	squeue = []
	for run in range(opts['runs']):
		opts['tmp_run'] = run
		opts['tmp_seed'] = opts['rng_seed'][run]
		opts['tmp_data'] = ' '.join(glob.glob('bootstrapping_run{tmp_run:02d}/subsample*'.format(**opts)))
		opts['tmp_error'] = ' '.join(opts['error'])
		opts['legacy'] = args.legacy

		job_desc['exec_pleione'] = '{python} -m pleione.{soft} --output {outfile} \
			--model {model} --final {final} --steps {steps} --error {tmp_error} --data {tmp_data} \
			SIMULATOR --python {python} --slurm {slurm} \
			--iter {num_iter} --inds {pop_size} --sims {num_sims} --best {pop_best} \
			--seed {tmp_seed} --swap {mut_swap} --rate {mut_rate} --cross {xpoints} --dist {dist_type} --self {self_rec} \
			--results bootstrapping_run{tmp_run:02d}/{results} --parsets {parsets} --rawdata {rawdata} --fitness {fitness} --ranking {ranking} \
			--crit {crit_vals} --prec {par_fmt} --syntax {syntax} --legacy {legacy}'.format(**opts).replace('\t', '')

		# remove --slurm if not setted by the user
		if opts['slurm'] is None:
			job_desc['exec_pleione'] = job_desc['exec_pleione'].replace('--slurm {slurm} '.format(**opts), '')

		# edit SIMULATOR to match user selection
		if opts['soft'].lower() == 'bng2':
			job_desc['exec_pleione'] = job_desc['exec_pleione'].replace('SIMULATOR', '--bng2 {bng2}'.format(**opts))
		elif opts['soft'].lower() == 'kasim':
			job_desc['exec_pleione'] = job_desc['exec_pleione'].replace('SIMULATOR', '--kasim {kasim}'.format(**opts))
		elif opts['soft'].lower() == 'nfsim':
			job_desc['exec_pleione'] = job_desc['exec_pleione'].replace('SIMULATOR', '--nfsim {nfsim}'.format(**opts))
		elif opts['soft'].lower() == 'piskas':
			job_desc['exec_pleione'] = job_desc['exec_pleione'].replace('SIMULATOR', '--piskas {piskas}'.format(**opts))
		print(job_desc['exec_pleione'])

		# use SLURM Workload Manager
		if opts['slurm'] is not None:
			cmd = os.path.expanduser('sbatch --no-requeue -p {partition} -N {nodes} -c {ncpus} -n {ntasks} -o {null} -e {null} -J {job_name} \
				--wrap ""{exec_pleione}"" --hold'.format(**job_desc)) # job is submitted in hold state
			cmd = re.findall(r'(?:[^\s,"]|"+(?:=|\\.|[^"])*"+)+', cmd)
			out, err = subprocess.Popen(cmd, shell = False, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()
			while err == sbatch_error:
				out, err = subprocess.Popen(cmd, shell = False, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()
			squeue.append(out.decode('utf-8')[20:-1])

		# use multiprocessing.Pool
		else:
			cmd = os.path.expanduser(job_desc['exec_pleione'])
			cmd = re.findall(r'(?:[^\s,"]|"+(?:=|\\.|[^"])*"+)+', cmd)
			squeue.append(cmd)

	if opts['slurm'] is not None:
		# edit dependencies to make pleione runs in series
		for job_id in range(1, len(squeue)): # job 0 must have no dependencies
			cmd = os.path.expanduser('scontrol update jobid={:d} dependency=afterok:{:d}'.format(squeue[job_id], squeue[job_id-1]))
			cmd = re.findall(r'(?:[^\s,"]|"+(?:=|\\.|[^"])*"+)+', cmd)
			out, err = subprocess.Popen(cmd, shell = False, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()

		# release jobs
		for job_id in range(len(squeue)):
			cmd = os.path.expanduser('scontrol release jobid={:d}'.format(squeue[job_id]))
			cmd = re.findall(r'(?:[^\s,"]|"+(?:=|\\.|[^"])*"+)+', cmd)
			out, err = subprocess.Popen(cmd, shell = False, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()

		# check if squeued jobs have finished
		for job_id in range(len(squeue)):
			cmd = 'squeue --noheader -j{:s}'.format(squeue[job_id])
			cmd = re.findall(r'(?:[^\s,"]|"+(?:=|\\.|[^"])*"+)+', cmd)
			out, err = subprocess.Popen(cmd, shell = False, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()
			while out.count(b'child') > 0 or err == squeue_error:
				time.sleep(1)
				out, err = subprocess.Popen(cmd, shell = False, stdout = subprocess.PIPE, stderr = subprocess.PIPE).communicate()

	# calibrate with multiprocessing.Pool
	else:
		with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
			pool.map(parallelize, sorted(squeue), chunksize = opts['runs']) # chunksize controls serialization of subprocesses

	return 0

def read_reports():
	reports = []
	for idx1 in range(opts['runs']):
		reports.append(sorted(glob.glob('bootstrapping_run{:02d}/{:s}*'.format(idx1, opts['results'])))[-1])
	print(reports)

	for folder in reports:
		last_outmodels = sorted(glob.glob(folder + ' '))[-1]
		with open(last_outmodels, 'r') as infile:
			data = pandas.read_csv(infile, delimiter = '\t', skiprows = 4, header = 0, engine = 'python')
		print(data.iloc[0,1])

	return 0

if __name__ == '__main__':
	sbatch_error = b'sbatch: error: slurm_receive_msg: Socket timed out on send/recv operation\n' \
		b'sbatch: error: Batch job submission failed: Socket timed out on send/recv operation'
	squeue_error = b'squeue: error: slurm_receive_msg: Socket timed out on send/recv operation\n' \
		b'slurm_load_jobs error: Socket timed out on send/recv operation'
	#sbatch_error = b'sbatch: error: Slurm temporarily unable to accept job, sleeping and retrying.'
	#sbatch_error = b'sbatch: error: Batch job submission failed: Resource temporarily unavailable'

	# general options
	args = argsparser()
	opts = ga_opts()

	# perform safe checks prior to any calculation
	safe_checks()

	# write bootstrapped obvervations
	#data, samples = bootstrapper()

	# call pleione N times
	#callibration()

	# read reports
	read_reports()
