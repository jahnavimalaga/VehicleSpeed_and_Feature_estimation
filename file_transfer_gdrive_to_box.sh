#!/bin/bash
#Author: Jahnavi Malagavalli

<<EOF
This bash script is used to tranfer files from "safestreetonline@gmail.com" Google drive to UT box folder
The script is currently run can be run in mac, windows or linux machine.
Before running the the script please ensure the following
1. Install the rclone
2. Set up the remote apps using <<rclone config>> and give them names
2. Set a CRON scheduling task so you do not have to run the script by yourself.

And lastly, command to run the script in terminal
<<sh FileTransfer_From_GdriveToBox.sh>>
EOF

#Set paths for Google Drive and Box, whose names are given as 'googledrive' and 'box,' respectively
gdrive_path1="googledrive:BeingWatched_Project_Video_Submissions (File responses)/Upload files (File responses)"
box_path1="box:Good Systems, automated speed enforcement work w Jahnavi, Lamha, Tong & Eric/Jahnavi's Fall23_Work/safestreet.online data/Vidoes and Photos"
gdrive_path2="googledrive:BeingWatched_Project_Video_Submissions (File responses)/BeingWatched Project Video Submissions (Responses).xlsx"
box_path2="box:Good Systems, automated speed enforcement work w Jahnavi, Lamha, Tong & Eric/Jahnavi's Fall23_Work/safestreet.online data"


# Run rclone to move files from Google Drive to Box
rclone move "$gdrive_path1" "$box_path1" #--delete-empty-src-dirs
# Verify the first move was successful
if [ $? -eq 0 ]; then
    # Run rclone to copy files from Google Drive to Box - Second transfer
    rclone copy "$gdrive_path2" "$box_path2"

    # Verify the second copy was successful
    if [ $? -eq 0 ]; then
        echo "Files transferred from Google Drive to Box and videos are removed from Google Drive at $(date)"
    else
        echo "Error: Second file transfer from Google Drive to Box failed at $(date)"
    fi
else
    echo "Error: First file transfer from Google Drive to Box failed at $(date)"
fi