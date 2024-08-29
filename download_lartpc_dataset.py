import os

def download_lartpc_dataset(exp: str):
    os.makedirs(os.path.join(exp, "datasets", "lartpc"), exist_ok=True)
    os.system(f"wget https://zenodo.org/records/8300355/files/training_data.tar.gz -P {exp}/datasets/lartpc")
    os.system(f"tar -xvzf {exp}/datasets/lartpc/training_data.tar.gz -C {exp}/datasets/lartpc")
    os.system(f"cp {exp}/datasets/lartpc/training_data/larcv_png_*.npy {exp}/datasets/lartpc")
    os.system(f"rmz -v {exp}/datasets/lartpc/training_data.tar.gz")
    os.system(f"rm -rv {exp}/datasets/lartpc/training_data")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str)
    args = parser.parse_args()

    exp = args.exp
    download_lartpc_dataset(exp)

if __name__ == "__main__":
    main()