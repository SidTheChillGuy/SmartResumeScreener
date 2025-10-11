def save_files(files):
    import shutil, os
    saved_paths = []
    for file in files:
        filename = os.path.basename(file.name)
        dest_path = os.path.join("resume_uploads", filename)
        shutil.copy(file.name, dest_path)
        saved_paths.append(dest_path)
    return f"Saved files:\n" + "\n".join(saved_paths)