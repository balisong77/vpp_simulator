import csv
import random

# Define the file path
file_path = 'trace_burst.csv'

# Define the number of rows to generate
num_rows = 10000

# Open the file in write mode
with open(file_path, 'w', newline='') as file:
    # Create a CSV writer object
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(['Packet Number', 'IPSec Ratio'])

    # Generate random values and write them to the file
    # for i in range(num_rows):
    #     packet_num = random.randint(0, 5)
    #     ratio = random.uniform(0, 1)
    #     writer.writerow([packet_num, ratio])
    
    for period in range(1, 5):
        # 每段一万行（tick），20step
        # 第一段，高流量
        for i in range(num_rows):
            packet_num = random.randint(5, 12)
            ratio = random.uniform(0, 1)
            # if i % 20 == 0:
            #     packet_num = random.randint(20,10)
            #     ratio = random.uniform(0, 1)
            writer.writerow([packet_num, ratio])
        # 第一段，低流量
        for i in range(num_rows):
            packet_num = random.randint(0, 5)
            ratio = random.uniform(0, 1)
            writer.writerow([packet_num, ratio])
        # 第三段，burst高流量
        for i in range(num_rows):
            packet_num = random.randint(0, 3)
            ratio = random.uniform(0, 1)
            if i % 8 == 0:
                packet_num = random.randint(20,30)
                ratio = random.uniform(0, 1)
            writer.writerow([packet_num, ratio])
        # 第四段，低流量
        for i in range(num_rows):
            packet_num = random.randint(0, 4)
            ratio = random.uniform(0, 1)
            writer.writerow([packet_num, ratio])

print(f"CSV file generated at {file_path}")