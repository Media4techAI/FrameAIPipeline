#!/usr/bin/env python3.7

import os
import sys
import argparse
import signal
import time
from datetime import datetime
from dotenv import load_dotenv
from lib import (
  Pipeline,
  MatlabFilter,
  FixGroundtruthFilter,
  FixImageBackgroundsFilter,
  ReconstructFilter,
  BoostFilter
)
import tensorflow as tf

# Global flag to handle graceful shutdown
interrupted = False


def signal_handler(signum, frame):
	"""Handle SIGINT (Ctrl+C) signal"""
	global interrupted
	interrupted = True
	print("\nInterrupt received, finishing current step before exiting...")
	# Set a new handler for immediate termination on second Ctrl+C
	signal.signal(signal.SIGINT, lambda s, f: sys.exit(1))


def load_environment_variables(script_dir):
	"""Load environment variables from .env file using dotenv"""
	env_file = os.path.join(script_dir, ".env")
	if os.path.exists(env_file):
		print(f"Loading environment variables from {env_file}...")
		load_dotenv(env_file)
		return True
	else:
		print("No .env file found, using system environment variables only")
		return False


def main(_):
	"""Main pipeline execution function"""
	# Setup signal handler for graceful shutdown
	signal.signal(signal.SIGINT, signal_handler)
	
	parser = argparse.ArgumentParser(
			description="Frame pipeline execution script",
			formatter_class=argparse.RawDescriptionHelpFormatter,
			epilog="""
Examples:
python3.7 run_pipeline.py ./masks/ ./orig.jpg "1:9"
python3.7 run_pipeline.py ./masks/ ./orig.jpg "1:9" ./custom_config.json
python3.7 run_pipeline.py -j /path/to/existing/job_folder

Environment variables can be set in .env file.
If no config_file is provided, default parameters will be used.
Use -j to resume processing from an existing job folder.
			"""
	)
    
	parser.add_argument("masks_folder", nargs='?', help="Path to masks folder (not needed if using -j)")
	parser.add_argument("original_image", nargs='?', help="Path to original image (not needed if using -j)")
	parser.add_argument("parts", nargs='?', help="Parts specification (e.g., '1:9') (not needed if using -j)")
	parser.add_argument("config_file", nargs='?', help="Optional custom config file")
	parser.add_argument("-j", "--job-folder", help="Resume processing from existing job folder")
	
	args = parser.parse_args()
	
	# Load environment variables
	script_dir = os.path.dirname(os.path.abspath(__file__))
	load_environment_variables(script_dir)
	
	# Check if resuming from existing job folder
	if args.job_folder:
		if not os.path.exists(args.job_folder):
			print(f"Error: Job folder {args.job_folder} does not exist")
			sys.exit(1)
		
		# For resuming jobs, we don't need to validate other arguments
		print(f"Resuming job from folder: {args.job_folder}")
		job_folder = args.job_folder
		job_id = os.path.basename(job_folder)
		shared_dir = os.path.dirname(job_folder)
		
		# Try to load job parameters from existing job
		params_file = os.path.join(job_folder, "input", "params.json")
		if os.path.exists(params_file):
			import json
			with open(params_file, 'r') as f:
				job_params = json.load(f)
				
			# Extract parameters from job config
			masks_folder = job_params.get('masks_folder', None)
			original_image = job_params.get('original_image', None)
			parts = job_params.get('parts', None)
			config_file = job_params.get('config_file', None)
			
			print(f"Loaded job parameters from {params_file}")
			print(f"  masks_folder: {masks_folder}")
			print(f"  original_image: {original_image}")
			print(f"  parts: {parts}")
		else:
			print(f"Warning: Could not find params.json in {params_file}")
			print("Job will be resumed without recreating input files")
			masks_folder = None
			original_image = None
			parts = None
			config_file = None
	else:
		# Validate input arguments for new job
		if not args.masks_folder or not args.original_image or not args.parts:
			print("Error: masks_folder, original_image, and parts are required when not using -j")
			parser.print_help()
			sys.exit(1)
			
		if not os.path.exists(args.masks_folder):
			print(f"Error: Masks folder {args.masks_folder} does not exist")
			sys.exit(1)
	
		if not os.path.exists(args.original_image):
			print(f"Error: Original image {args.original_image} does not exist")
			sys.exit(1)
			
		masks_folder = args.masks_folder
		original_image = args.original_image
		parts = args.parts
	
	# Determine configuration file (only for new jobs)
	if not args.job_folder:
		config_file = None
		if args.config_file and os.path.exists(args.config_file):
				config_file = args.config_file
				print(f"Using custom config file: {config_file}")
		elif os.environ.get('PIPELINE_DEFAULT_CONFIG'):
				default_config = os.environ.get('PIPELINE_DEFAULT_CONFIG')
				if os.path.exists(default_config):
						config_file = default_config
						print(f"Using default config file from environment: {config_file}")
		
		if not config_file:
				print("Using default configuration")
		
		# Setup new job folder
		shared_dir = os.environ.get('PIPELINE_JOBS_DIR', 
																												'/mnt/c/Users/user/Documenti/Reassembly2d_Sources/jobs')
		
		job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
		job_folder = os.path.join(shared_dir, job_id)
		
		print(f"Creating job folder at {job_folder}")
	else:
		print(f"Resuming job: {job_id}")
	
	pipeline = Pipeline(
		shared_dir=shared_dir,
		mask_folder=masks_folder,
		original_image=original_image,
		config_file=config_file,
		parts=parts,
		job_folder=job_folder
	)

	pipeline.add_filter(MatlabFilter)
	pipeline.add_filter(FixGroundtruthFilter)
	pipeline.add_filter(FixImageBackgroundsFilter)
	time.sleep(1)  # Simulate some delay
	pipeline.add_filter(BoostFilter)
	time.sleep(1)  # Simulate some delay
	pipeline.add_filter(ReconstructFilter)

	pipeline.prepare()
	if interrupted:
		print("Pipeline interrupted before pipeline preparation.")
		sys.exit(130)  # Standard exit code for SIGINT
	
	# Wait for MATLAB processing
	pipeline_completed = pipeline.process()
	
	if interrupted:
		print(f"\nPipeline interrupted during pipeline processing.")
		print(f"Job folder partially completed: {job_folder}")
		sys.exit(130)  # Standard exit code for SIGINT
	elif pipeline_completed:
		print(f"Pipeline execution completed for job: {job_id}")
		print(f"Job folder: {job_folder}")
	else:
		print(f"Error during pipeline execution.")
		sys.exit(1)

if __name__ == "__main__":
	tf.compat.v1.app.run()