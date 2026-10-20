import os


def get_subject_ids(audio_folder):

    subject_ids = []

    for file in os.listdir(audio_folder):

        if file.endswith(".wav"):

            subject_id = file.replace(".wav", "")

            subject_ids.append(subject_id)

    subject_ids.sort(key=int)

    return subject_ids