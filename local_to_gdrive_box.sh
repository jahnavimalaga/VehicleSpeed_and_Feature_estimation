#!/bin/bash
export RCLONE_CONFIG=~/.config/rclone/rclone.conf

#Author: Jahnavi Malagavalli
<<EOF
This bash script is used to upload a file to "safestreetonline@gmail.com" Google Drive using rclone

Follow the commands in this to set up the rclone
https://www.endpointdev.com/blog/2020/09/rclone-upload-to-cloud-from-cli/
...

EOF

# Set paths for local file and Google Drive
local_file_path="Complete_Pipeline/Results/preprocessed_outputs/output.xlsx"
local_file_path_1="Complete_Pipeline/runs/track/"

gdrive_path="gdrive:Results"
box_path="utbox:Good Systems, automated speed enforcement work w Jahnavi, Lamha, Tong & Eric/Jahnavi's Fall23_Work/safestreet.online data/Processed_Results"

box_path_1="utbox:Good Systems, automated speed enforcement work w Jahnavi, Lamha, Tong & Eric/Jahnavi's Fall23_Work/safestreet.online data/Processed_Results/Crude_Results"

# Set log file path
log_file="file_upload_log.txt"

# Function to log messages with timestamp
log_message() {
    echo "$(date): $1" >> "$log_file"
}

# Run rclone to upload the file to Google Drive and UT Box
~/bin/rclone copy "$local_file_path" "$gdrive_path"
~/bin/rclone copy "$local_file_path" "$box_path"

~/bin/rclone move "$local_file_path_1" "$box_path_1" --delete-empty-src-dirs


# Verify the upload was successful
if [ $? -eq 0 ]; then
    log_message "File uploaded to Google Drive and UT Box successfully"
    echo "File uploaded to Google Drive and UT Box successfully at $(date)"
else
    log_message "Error: File upload to Google Drive and UT Box failed"
    echo "Error: File upload to Google Drive and UT Box failed at $(date)"
fi
