#!/usr/bin/env python3

import os, shlex, subprocess, sys

config = {
	'subdirs': [ name for name in os.listdir(os.getcwd()) if os.path.isdir(os.path.join(os.getcwd(), name)) ],
	'rules': ['all', 'debug', 'clean'],
	'default_rule': 'all',
	'sources': []
}

def build(rule, subdirs):
	procs = []
	pwd = os.getcwd()

	for subdir in subdirs:
		os.chdir(subdir)
		p = subprocess.Popen(['make', rule])
		procs.append((subdir, p))
		os.chdir(pwd)
	
	return procs

def main():
	try:
		rule = sys.argv[1]
		if rule not in config['rules']:
			print_usage()
			sys.exit(1)
	except IndexError:
		rule = config['default_rule']

	if len(sys.argv) > 2:
		subdirs = shlex.split(' '.join(sys.argv[2:]))
		for subdir in subdirs:
			if subdir not in config['subdirs']:
				print('Could not find directory: ', subdir)
				return
	else:
		subdirs = config['subdirs']

	procs = build(rule,subdirs)
	failed = False

	for subdir, p in procs:
		rc = p.wait()
		if rc != 0:
			print('build: ' + subdir + ' failed.')
			failed = True

	if failed:
		sys.exit(1)

def print_usage():
	print('Usage: ' + sys.argv[0] + ' <rule>')
	print('Valid rules: ' + ' '.join(config['rules']))

if __name__ == '__main__':
	main()