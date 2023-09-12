import os
import shutil
import openai
import moviepy.editor as mp
import pandas as pd
import chardet
import csv
from config import *

# Function to get the names of subfolders in a given folder
def get_subfolder_names(folder_path):
    subfolder_names = []

    # Traverse each item in the folder
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        # Check if it is a directory (subfolder)
        if os.path.isdir(item_path):
            subfolder_names.append(item)

    return subfolder_names


# Function to check if a folder has folders
def check_subfolder(folder_path):
    subfolder_names = get_subfolder_names(folder_path)

    if not subfolder_names:
        return False
    else:
        return True


# Function to get unique file extensions in a folder
def get_extension(folder_path):
    extensions = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_extension = file.split('.')[-1]
            if file_extension not in extensions:
                extensions.append(file_extension)
    return extensions


# Function to create folders based on file extensions
def make_folders(ext_list, parent_folders):
    for folder_name in ext_list:
        # Join the parent directory path and folder name to create the complete path
        folder_path = os.path.join(parent_folders, folder_name + "_files")

        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"{folder_name} created successfully!")
        else:
            print(f"{folder_name} already exists!")


# Function to copy files from a folder to respective extension folders
def copy_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_extension = file.split('.')[-1]
            src_file = os.path.join(root, file)
            shutil.copy2(src_file, "Database/" + file_extension + "_files")


# Function to get the script from a video file using MoviePy and OpenAI
def get_script(filename):
    video = mp.VideoFileClip(filename)
    audio_file = video.audio
    audio_file.write_audiofile("temp.mp3")
    audio_file = open("temp.mp3", "rb")
    transcript = openai.Audio.translate("whisper-1", audio_file)
    return transcript["text"]


# Function to create text files with scripts from video files
def make_txt(src_folder, tgt_folder):
    filenames = os.listdir(src_folder)
    for filename in filenames:
        video_file = os.path.join(src_folder, filename)
        filename_without_extension = os.path.splitext(filename)[0]
        txt = get_script(video_file)
        txt_path = os.path.join(tgt_folder, filename_without_extension + ".txt")
        with open(txt_path, "w") as file:
            file.write(txt)
            file.close()
        print(txt)


# Function to convert XLSX files to CSV files
def convert_csv(src_folder, tgt_folder):
    filenames = os.listdir(src_folder)
    for filename in filenames:
        xlsx_file = os.path.join(src_folder, filename)
        filename_without_extension = os.path.splitext(filename)[0]
        xlsx = pd.read_excel(xlsx_file, sheet_name=None)
        for sheet_name, data in xlsx.items():
            csv_path = os.path.join(tgt_folder, filename_without_extension + "_" + sheet_name + ".csv")
            data.to_csv(csv_path, index=False)


def rewrite_csv_as_utf8(file_path):
    with open(file_path, 'rb') as file:
        detection_result = chardet.detect(file.read())
        encoding = detection_result['encoding']

    with open(file_path, 'r', encoding=encoding) as file:
        csv_data = file.read()

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(csv_data)


def rewrite_csv_folder(folder_path):
    csv_list = os.listdir(folder_path)
    for csv_path in csv_list:
        csv_file_path = os.path.join(folder_path, csv_path)
        print("__________________")
        print(csv_file_path)
        rewrite_csv_as_utf8(csv_file_path)


def delete_lines(csv_path):
    with open(csv_path, 'r', encoding='latin') as file:
        lines = list(csv.reader(file))
    new_lines = [lines[0]]
    for line in lines:
        if line[0] != 'post_id':
            new_lines.append(line)
    with open(csv_path, 'w', newline='', encoding='latin') as file:
        writer = csv.writer(file)
        writer.writerows(new_lines)


def process_text_db(tgt_folder, csv_path):
    with open(csv_path, 'r', encoding='latin') as file:
        lines = list(csv.reader(file))
    post = ""
    post_id = ""
    for line in lines:
        if line[0] != 'post_id':
            post_id = line[0]
            if post == "":
                post = post + line[3] + "\n\n"
            post = post + line[10] + "\n"
        else:
            txt_path = os.path.join(tgt_folder, post_id + ".txt")
            with open(txt_path, "w", encoding='latin') as file:
                print(post)
                file.write(post)
                file.close()
            post = ""
            post_id = ""


# OpenAI key
openai.api_key = openai_key

# 1. Make Folders
# extensions = get_extension(path)
# make_folders(extensions, parent_folder)

# 2. Classify files
# copy_files(path)

# 3. Make scripts from mp4 files
# os.makedirs(txt_folder)
# make_txt(video_folder, txt_folder)
# os.remove('temp.mp3')

# 4. Convert XLSX files to CSV files
# os.makedirs(csv_folder)
# convert_csv(xlsx_folder, csv_folder)
# rewrite_csv_folder(csv_folder)

# 5. Delete unuseful lines in csv
# delete_lines(csv_file)

# 6. Process csv files
process_text_db(txt_folder, csv_file)