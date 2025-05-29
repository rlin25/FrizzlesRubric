import csv
import sys

if len(sys.argv) != 3:
    print("Usage: python fix_csv_quotes.py <input_csv> <output_csv>")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8", newline="") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)
    for row in reader:
        # Strip whitespace and ensure only two columns
        row = [col.strip() for col in row]
        if len(row) > 2:
            # Join all but last as prompt, last as label
            prompt = ",".join(row[:-1])
            label = row[-1]
            writer.writerow([prompt, label])
        else:
            writer.writerow(row)

print(f"Cleaned CSV written to {output_path}") 