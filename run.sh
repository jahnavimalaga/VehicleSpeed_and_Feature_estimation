#!/bin/bash

#Author: Jahnavi Malagavalli, Dec 2023

#!/bin/bash
export RCLONE_CONFIG=~/.config/rclone/rclone.conf

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

# Please change this path accordingly
current_directory=$(pwd)
cd "$current_directory" #/work2/06519/jahnavi/frontera/GoodSystems/for_repo


#...................................................................................
# Set paths for Google Drive and Box, whose names are given as 'googledrive' and 'box,' respectively
gdrive_path1="gdrive:BeingWatched_Project_Video_Submissions (File responses)/Upload files (File responses)"
box_path1="utbox:Good Systems, automated speed enforcement work w Jahnavi, Lamha, Tong & Eric/Jahnavi's Fall23_Work/safestreet.online data/Vidoes and Photos"
gdrive_path2="gdrive:BeingWatched_Project_Video_Submissions (File responses)/BeingWatched Project Video Submissions (Responses).xlsx"
box_path2="utbox:Good Systems, automated speed enforcement work w Jahnavi, Lamha, Tong & Eric/Jahnavi's Fall23_Work/safestreet.online data"


# Run rclone to move files from Google Drive to Box
~/bin/rclone move "$gdrive_path1" "$box_path1" #--delete-empty-src-dirs
# Verify the first move was successful
if [ $? -eq 0 ]; then
    # Run rclone to copy files from Google Drive to Box - Second transfer
    ~/bin/rclone copy "$gdrive_path2" "$box_path2"

    # Verify the second copy was successful
    if [ $? -eq 0 ]; then
        echo "Files transferred from Google Drive to Box and videos are removed from Google Drive at $(date)"
    else
        echo "Error: Second file transfer from Google Drive to Box failed at $(date)"
    fi
else
    echo "Error: First file transfer from Google Drive to Box failed at $(date)"
fi

#.....................................................................................

# Set paths for Google Drive and Box, whose names are given as 'googledrive' and 'box,' respectively
local_path="Complete_Pipeline/test"
box_path1="utbox:Good Systems, automated speed enforcement work w Jahnavi, Lamha, Tong & Eric/Jahnavi's Fall23_Work/safestreet.online data/Vidoes and Photos"

box_path2="utbox:Good Systems, automated speed enforcement work w Jahnavi, Lamha, Tong & Eric/Jahnavi's Fall23_Work/safestreet.online data/Videos_processed"

# Run rclone to cp files from Box to local
~/bin/rclone copy "$box_path1" "$local_path" --include "*.{mp4,mov}"
echo "Files transferred from box to local"
 # Verify the first move was successful
if [ $? -eq 0 ]; then
    sleep 60

    # Run rclone to move files from box to box - Second transfer
    ~/bin/rclone move "$box_path1" "$box_path2" --include "*.{mp4,mov}"

    # Change this to your conda env location
    source /work2/06519/jahnavi/frontera/miniconda3/bin/activate 
    
    file_path="CompletePipleline.out"

    #echo "$(stat -c %s "$file_path")"
    # Verify the second copy was successful
    if [ $? -eq 0 ]; then
        echo "Files transferred from box to Box"
        conda activate test_vvd #good_sys
        echo "Executing script.slurm"
        sbatch script.slurm
        sleep 120
       

        # Set the maximum sleep time (in seconds)
        max_sleep_time=600  # 10 minutes

        # Initialize total sleep time
        total_sleep_time=0

        sleep 60


        initial_size=$(stat -c %s "$file_path")

        sleep 300

        current_size=$(stat -c %s "$file_path")

        # Infinite loop
        while true; do
            if [ "$initial_size" -eq "$current_size" ]; then
                sleep 120
                echo "The file size did not change."
                # Continue with the rest of the script

                echo "Executing post_processing.py"
                python Complete_Pipeline/post_processing.py --path Complete_Pipeline/runs/track/
                echo "post_processing.py completed"
                sleep 120

                if [ $? -eq 0 ]; then
                    echo "Executing local_to_gdrive_box.sh"
                    source local_to_gdrive_box.sh
                    echo "local_to_gdrive_box.sh completed"
                    sleep 30
                    rm -r Complete_Pipeline/test/*.{mp4,mov}
                    echo "Deletion of Complete_Pipeline/test/*.{mp4,mov} completed"
                else
                    echo "Error in post_processing.py at $(date)"
                fi

                break  # Exit the loop if the sentence is found
            else
                initial_size=$(stat -c %s "$file_path")
                echo "File is still being written."
                sleep 60  # Adjust the sleep duration as needed
                ((total_sleep_time++))
                current_size=$(stat -c %s "$file_path")

                # Check if the total sleep time exceeds the threshold
                if [ "$total_sleep_time" -gt "$max_sleep_time" ]; then
                    echo "Exceeded maximum sleep time. Exiting the loop."
                    break
                fi
            fi
        done

    else
        echo "Error: Second file transfer box to box failed at $(date)"
    fi
    conda deactivate

else
    echo "Error: First file transfer from Box to local failed at $(date)"
fi
