import os

def filesNdirec(direcory_path,ext):
    files_list,names,roots =[],[],[]
    for root, dirs, files in os.walk(direcory_path,True):
        for name in files:
            if name.__contains__(ext):
                files_list.append(os.path.join(root, name))
                names.append(name)
                roots.append(root)
    return files_list,names,roots
