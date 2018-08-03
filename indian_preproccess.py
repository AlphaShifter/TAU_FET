# import pandas as pd
import os.path
import os
import numpy as np
import glob

working_dir = "data/indian_expression_db/IMFDB_final/*/*/*.txt"
emotion_list= ["ANGER", "HAPPINESS", "SADNESS", "SURPRISE", "FEAR", "DISGUST", "NEUTRAL"]
num_emotion=len(emotion_list)
file_list =  glob.glob(working_dir, recursive=True)

filenames_and_labels = {}

problem_line = 0
	#variable for counting how many lines (ergo how many files and labels) were problematic and will be unused
problem_file = 0
file_list.sort()

for labelfile in file_list:
	#iterate through each labelfile in the collection of label files
	with open(labelfile, 'r') as searchfile:
		#open labelfile
		for line in searchfile:
			#iterate through each line in the labelfile
			split = line.split()
				#split line by spaces (including tabs) into list of each item in line
			if len(split) <= 2 or split[2].endswith(".jpg") == False or "_" not in split[2]:
				#if there are less than three items in the line (ergo item[2], filename, is missing) or it isn't a JPG or it doesn't have _
				problem_line = problem_line + 1
					#add 1 to problem line count
				continue
					#skip line (do not include in dictionary)
			removedpath = labelfile.rsplit("/", 1)[0]
				#remove first element from the right after "/" from filepath of .txt file
			imagefilepath = removedpath + "/images/" + split[2]
				#add to path "images" folder and filename from line in txt file
			filepath = os.path.abspath(imagefilepath)
				#get path of image
			if not os.path.isfile(filepath):
				problem_file += 1
				continue
				#if path doesn't exist, skip file and add to problem file counter
			if len(split) <= 11 or split[11] not in emotion_list:
				#if there are less than twelve items in the line (ergo item[11], expression label, is missing) or if the element does not match one of the seven labels
				problem_line = problem_line + 1
				#add 1 to problem line count
				continue
					#skip line (do not include in dictionary)
			label = split[11]
				#label is the twelfth element in the split line
			filenames_and_labels[filepath] = label
					#add filename and label as key-value pair to dictionary

# for file in file_list:
# 	df_list = [pd.read_table(file) for file in file_list]
# 	if df_list:
# 		final_df = pd.concat(df_list)
# 		final_df.to_csv(os.path.join(working_dir, "Final.csv"))

psudo_onehot_dict={j:[1 if i == j else -1 for i in emotion_list] for j in emotion_list}
print(psudo_onehot_dict)
for filename in filenames_and_labels.keys():
	label=filenames_and_labels[filename]
	filenames_and_labels[filename]=psudo_onehot_dict[label]
# print(filenames_and_labels,len(filenames_and_labels))
# print(filenames_and_labels,len(filenames_and_labels))
for filename,label in filenames_and_labels.items():
	print(filename+" "+" ".join(str(x) for x in label))
