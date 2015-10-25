import sys
import csv
import bs4 as BeautifulSoup

#DATA_DIRECTORY = '/home/dev/School/CSc-522/Project/autotagger/data/original'

#SETUP 
records = []
body = []
code_only_records = []
plain_text_records = []
text_and_html_records = []
no_html_records = []


csv_reader = csv.reader(open('/home/dev/School/CSc-522/Project/autotagger/data/original/TEST-full.csv', newline=''))
for row in csv_reader: 
    records.append(row)

for i in range(1, len(records)):
    body = BeautifulSoup.BeautifulSoup(records[i][8], 'html.parser')
    # CHECKING FOR CODE BLOCKS 
    if len(body.find_all('code')) != 0:
        code_only_records.append(records[1])
        # Concatenate code blocks
#    if records[i]

