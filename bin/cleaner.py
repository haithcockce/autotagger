import sys
import csv
import bs4 as BeautifulSoup
import string
import copy

# DONT JUDGE ME I DONT KNOW HOW TO PYTHON AND I GOOGLED ALMOST ALL THIS

#SETUP 
records = []
code_only_records = []
plain_text_records = []
tags= []
transtable = {ord(c): ' ' for c in string.punctuation}
PROJECT_PATH = '/home/dev/School/CSc-522/Project/autotagger/'
stripped = ''

csv_reader = csv.reader(open(PROJECT_PATH + 'data/original/TEST-full.csv', 'r', newline=''))
for row in csv_reader: 
    records.append(row)

code_only_records.append(copy.deepcopy(records[0]))
plain_text_records.append(copy.deepcopy(records[0]))

tag_str = ''
for i in range(1, len(records)):
    stripped = ''
    plain_text_records.append(copy.deepcopy(records[i]))
    body = BeautifulSoup.BeautifulSoup(plain_text_records[i][8], 'html.parser')
    if len(body.find_all('code')) != 0:
        # Gather all the records that have code blocks and unwrap the contents
        #code_only_records.append(copy.deepcopy(records[i]))

        # Remove all code blocks 
        while len(body.find_all('code')) != 0:
            body.code.decompose()  

    # Strip away all punctuation and random escape sequences   
    for dom_element in body.find_all(True):
        stripped += dom_element.get_text().strip().translate(transtable) + ' '
        stripped = stripped.replace('\n', ' ').replace('\\n', ' ').replace('\t', ' ').replace('\r', ' ')
    plain_text_records[i][8] = stripped

    # Grab the tags
#    if len(records[i][16]) != 0:
#        tag_strings = records[i][16].replace('<', '').replace('>', ' ').strip().split(' ')
#        for tag in tag_strings:
#            if tag not in 

# Flatten code blocks in code only
#stripped = ''
#for i in range(1, len(code_only_records)):
#    body = BeautifulSoup.BeautifulSoup(code_only_records[i][8], 'html.parser')
#    code_contents = body.code.contents
#    for i in range(0,len(code_contents)):
#       stripped += code_contents[i] + ' '
#    code_only_records[i][8] = stripped

# Flatten code blocks 

csv_out = open(PROJECT_PATH + 'data/plain-text/TEST-full.csv', 'w', newline='')
csv_writer = csv.writer(csv_out, dialect='unix')
for row in plain_text_records:
    csv_writer.writerow(row)


