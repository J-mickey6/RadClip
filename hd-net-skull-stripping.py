import subprocess, glob, os

base_dir = r"C:\visual code\MRI\dataset\PDCADxFoundation_Val_image_mask_20250801\val"
folders = glob.glob(os.path.join(base_dir, "RJPD_*"))

for folder in folders:
    inp  = os.path.join(folder, "T1.nii.gz")
    outp = os.path.join(folder, "T1_BET.nii.gz")
    cmd = ["hd-bet", "-i", inp, "-o", outp, "-device", "cpu"] # CPU 모드
    # cmd = ["hd-bet", "-i", inp, "-o", outp, "-device", "cuda:0"] # GPU 모드
    subprocess.run(cmd, check=True)
    print(folder + "의 HD-BET 처리 완료!")

print("전체 훈련 샘플 HD-BET 처리 완료!") 