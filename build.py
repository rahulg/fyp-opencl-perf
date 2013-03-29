#!/usr/bin/env python3

import os, shlex, subprocess, sys

config = {
	'subdirs': ['derpcl','tests'],
	'rules': ['all', 'debug', 'clean'],
	'default_rule': 'all',
	'sources': []
}

def build(rule, subdirs):
	rcs = []
	pwd = os.getcwd()

	for subdir in subdirs:
		os.chdir(subdir)
		if subdir == 'derpcl':
			rc = subprocess.call(['make', rule])
			rcs.append((subdir, rc))
		elif subdir == 'tests':
			rc = subprocess.call(['./build.py', rule])
			rcs.append((subdir, rc))
		os.chdir(pwd)

	return rcs

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

	rcs = build(rule,subdirs)
	failed = False

	for subdir, rc in rcs:
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