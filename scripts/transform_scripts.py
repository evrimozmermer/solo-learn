import os
import yaml
import glob

def save_yaml(data, path):
    with open(path, 'w') as outfile:
        yaml.dump(data, outfile)

folders = os.listdir("pretrain")
for folder in folders:
    filepaths = glob.glob(os.path.join("pretrain", folder, "*.sh"))
    for filepath in filepaths:
        with open(filepath, "r") as fp:
            pre_text = fp.readlines()
        post_text = {}
        for i, line in enumerate(pre_text):
            if i == 0:
                pass
            else:
                arg = line.split("--")[-1].split(" ")[0]
                arg = arg.replace("\n","").replace("\\","")
                try:
                    val = line.split("--")[-1].split(" ")[1].replace("\n","").replace("\\","")
                except:
                    val = line.split("--")[-1].replace("\n","").replace("\\","")
                val = True if val=="" else val
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                post_text[arg] = val

        out_path = os.path.join("yaml_configs", filepath)
        out_dir = os.path.join(*out_path.split(os.sep)[0:-1])
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = out_path.replace(".sh", ".yaml")
        save_yaml(post_text, out_path)


