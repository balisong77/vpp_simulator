import csv
import random

# Define the file path
file_path = 'trace.csv'

# Define the number of rows to generate
num_rows = 1000

# Open the file in write mode
with open(file_path, 'w', newline='') as file:
    # Create a CSV writer object
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(['Packet Number', 'IPSec Ratio'])

    # Generate random values and write them to the file
    for i in range(num_rows):
        packet_num = random.randint(0, 512)
        ratio = random.uniform(0, 1)
        writer.writerow([packet_num, ratio])

print(f"CSV file generated at {file_path}")