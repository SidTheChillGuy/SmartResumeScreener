def save_files(files):
    import shutil, os
    saved_paths = []
    for file in files:
        filename = os.path.basename(file.name)
        dest_path = os.path.join("user_resume_unparsed", filename)
        shutil.copy(file.name, dest_path)
        saved_paths.append(dest_path)
    return f"Saved files:\n" + "\n".join(saved_paths)