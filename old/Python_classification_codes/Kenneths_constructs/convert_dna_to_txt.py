# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:21:54 2019

@author: willi
"""

from snapgene_reader import snapgene_file_to_dict, snapgene_file_to_seqrecord
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import Bio.SeqIO
import os
converted_files = []
files = list(os.walk('.'))
for file in files[0][2]:
    if file[-4:] == '.dna':
        name = file[:-4]
        seq_record = snapgene_file_to_seqrecord(file)
        converted_files.append(seq_record)
        with open((name+'.txt'),'w') as newfile:
            newfile.write(seq_record.seq.tostring())
        
        
        