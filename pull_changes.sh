#!/bin/bash

# Get the current directory
current_dir=$(pwd)

# Loop over all the folders in the current directory
for folder in *; do

  # If the folder is a directory, apply git pull
  if [ -d "$folder" ]; then
    cd "$folder"
    git pull
    cd "$current_dir"
  fi

done
