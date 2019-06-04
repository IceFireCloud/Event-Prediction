#!/usr/bin/python3

# print first lines of results from Model_Results.py

name = 'Model_Results.csv'
num_lines = 10  

with open(name, 'r') as f:
    lines = f.readlines()
    lines = lines[-num_lines:]
    for line in lines:
        line_list = line.split()
        for col in range(len(line_list)):
            if col > 2 and col < 9:
                print(line_list[col], end=' ')
            elif col > 8 and col < 12:
                print(line_list[col][:6], end=' ')
        print()

        


