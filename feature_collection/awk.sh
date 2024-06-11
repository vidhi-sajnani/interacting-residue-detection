#!/bin/bash

# Create the output_files directory if it doesn't exist

# Loop through each text file in the current directory
for file in *.txt; do
    # Skip the script itself and output_files directory
    if [[ "$file" != "awk1.sh" && "$file" != "output_files" ]]; then
        # Step 1: Extract lines containing '<-->' from input.txt and save to output.txt
        awk '/<-->/' "$file" > output.txt

        # Step 2: Extract specific columns from output.txt and save to output1.txt
        awk '{print $5,$6,$11,$12}' output.txt > output1.txt

        # Step 3: Separate the extracted columns into two separate lines and save to output2.txt
        awk '{a[NR]=$1 " " $2; b[NR]=$3 " " $4} END {for(i=1;i<=NR;i++) print a[i]; for(i=1;i<=NR;i++) print b[i]}' output1.txt > output2.txt

        # Step 4: Remove duplicate lines and save to 1a4y.txt
        awk '!seen[$1,$2]++' output2.txt > "$file"_temp.txt

        # Step 5: Add filename as third field and print with column formatting, save to output file
        awk '
          NR==1{ sub(/\.txt$/, "", FILENAME) } # remove .txt suffix from FILENAME
          NR>0{ $3=FILENAME }                  # replace the first field with filename
          1                                    # print record
        ' "$file"_temp.txt | column -t > "output_files/${file%.*}_output.txt"

        # Cleanup temporary files
        rm output.txt output1.txt output2.txt "$file"_temp.txt
    fi
done

echo "All processing completed. Output files saved in output_files directory."
