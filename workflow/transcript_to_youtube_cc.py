#
# Transform transcript to sentences only for youtube cc upload
# -------------------------------
# This script takes out all our additional Markdown formatting and titles, 
# so that we receive a clean list of sentences per input file (written to a specified output folder).
# This output can be copy + pasted into the Youtube UI for closed captions (in the Youtube creator studio)
#

import argparse
import os
import sys
sys.path.append(os.getcwd())
import time

import blingfire
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--in-files', action='store', dest='in_files', help='glob of markdown transcript files', required=True)
parser.add_argument('--out-folder', action='store', dest='out_folder', help='folder to put in cc files', required=True)
args = parser.parse_args()

start_time = time.time()


files = glob.glob(args.in_files)

ignore_line_starts = ("#","*Automatic closed captions","Average talking")
ignore_line_ends = ("seconds*")

if not os.path.exists(args.out_folder):
    os.makedirs(args.out_folder)

for file in files:
    print(file)
    with open(file,"r",encoding="utf8") as in_file:
        with open(os.path.join(args.out_folder,os.path.splitext(os.path.basename(file))[0]+".txt"),"w",encoding="utf8") as out_file:
            for l in in_file:
                if not l.startswith(ignore_line_starts) and not l.strip().endswith(ignore_line_ends):
                    sentences = blingfire.text_to_sentences(l).splitlines()
                    for s in sentences:
                        out_file.write(s+"\n")
