def save_files_unparsed(files):
    import shutil, os
    saved_paths = []
    for file in files:
        filename = os.path.basename(file.name)
        dest_path = os.path.join("user_resumes_unparsed", filename)
        try:
            shutil.copy(file.name, dest_path)
        except:
            os.remove(dest_path)
            shutil(file.name, dest_path)
        saved_paths.append(dest_path)
    return f"Saved files:\n" + "\n".join(saved_paths)