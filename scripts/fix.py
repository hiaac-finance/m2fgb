import os
import glob
import shutil

folders = glob.glob("../results/group_experiment/*/*")
for folder in folders:
    if "XtremeFair_LGBM" in folder:
        new_name = folder
        new_name = new_name.replace("XtremeFair_LGBM", "MMBFair")

        print(folder, new_name)
        # create new folder if does not exist
        if not os.path.exists(new_name):
            os.mkdir(new_name)

        files_to_copy = glob.glob(folder + "/*")
        for file in files_to_copy:
            # copy content from old folder to new_folder
            shutil.copy(file, new_name)
