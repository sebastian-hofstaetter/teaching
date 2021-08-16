#
# transfor transcipt to sentences only for youtube cc upload
# -------------------------------
#

import argparse
import os
import sys
sys.path.append(os.getcwd())
import copy
import time
import datetime

import blingfire
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--in-files', action='store', dest='in_files', help='glob of markdown transcript files', required=True)
parser.add_argument('--out-folder', action='store', dest='out_folder', help='fodler to put in cc files', required=True)
args = parser.parse_args()

start_time = time.time()


files = glob.glob(args.in_files)

ignore_line_starts = ("#","*Automatic closed captions","Average talking")
ignore_line_ends = ("seconds*")


for file in files:
    print(file)
    with open(file,"r",encoding="utf8") as in_file:
        with open(os.path.join(args.out_folder,os.path.splitext(os.path.basename(file))[0]+".txt"),"w",encoding="utf8") as out_file:
            for l in in_file:
                if not l.startswith(ignore_line_starts) and not l.strip().endswith(ignore_line_ends):
                    sentences = blingfire.text_to_sentences(l).splitlines()
                    for s in sentences:
                        out_file.write(s+"\n")
