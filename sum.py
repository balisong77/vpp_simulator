import csv


file1_name = 'result_64_1.csv'
file2_name = 'result_32_1.csv'
file3_name = 'result_expert_1.csv'


# 读取并计算第一个CSV文件的第二列总和
sum_column1 = 0
with open(file1_name, 'r') as file1:
    csv_reader1 = csv.reader(file1)
    next(csv_reader1)
    for row in csv_reader1:
        sum_column1 += float(row[1])

# 读取并计算第二个CSV文件的第二列总和
sum_column2 = 0
with open(file2_name, 'r') as file2:
    csv_reader2 = csv.reader(file2)
    next(csv_reader2)
    for row in csv_reader2:
        sum_column2 += float(row[1])
        
# 读取并计算第二个CSV文件的第二列总和
sum_column3 = 0
with open(file3_name, 'r') as file3:
    csv_reader3 = csv.reader(file3)
    next(csv_reader3)
    for row in csv_reader3:
        sum_column3 += float(row[1])

# 输出两个文件第二列数相加总和
print(f"{file1_name} reward sum: ", sum_column1)
print(f"{file2_name} reward sum: ", sum_column2)
print(f"{file3_name} reward sum: ", sum_column3)