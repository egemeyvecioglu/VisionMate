# Define the input and output file names
input_file_name = 'metadata.txt'
output_file_name = 'updated_metadata.txt'

# Open the input file in read mode and the output file in write mode
with open(input_file_name, 'r') as infile, open(output_file_name, 'w') as outfile:
    # Read all lines from the input file
    lines = infile.readlines()

    # Write the first line as it is to the output file
    outfile.write(lines[0])

    # Process each subsequent line
    for line in lines[1:]:  # Skip the first line
        # Remove any trailing whitespace including newlines
        stripped_line = line.strip()
        
        attributes = stripped_line.split()
        midpoint_3d = [float(item) for item in attributes[-1].split(',')]
        camera_to_marker_vector = [float(item) for item in attributes[2].split(',')]
        
        gaze_direction = [camera_to_marker_vector[i] - midpoint_3d[i] for i in range(3)]
        
        print(midpoint_3d, camera_to_marker_vector)
        
        # Append "0,0,0" to the line
        new_line = f"{stripped_line} {gaze_direction[0]},{gaze_direction[1]},{gaze_direction[2]}\n"
        # Write the modified line to the output file
        outfile.write(new_line)

print(f"File '{input_file_name}' has been processed. Updated lines are written to '{output_file_name}'.")
