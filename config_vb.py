import os

# front-end parameter settings
win_size = 320
fft_num = 320
win_shift = 160
chunk_length = 3*16000
feat_type = 'sqrt'   # normal, sqrt, cubic, log_1x
is_conti = False
conti_path = './CP_dir/checkpoint_early_exit_16th.pth.tar'
is_pesq = True

# server parameter settings
json_dir = '/home/yuguochen/CYCLEGAN-ATT-UNET/data/Json'
file_path = '/home/yuguochen/vbdataset'
loss_dir = './LOSS/aia_internew_vb.mat'
batch_size = 2
epochs = 80
lr = 5e-4
model_best_path = './BEST_MODEL/aia_internew_vb.pth.tar'
check_point_path = './CP_dir/aia_internew_vb'

os.makedirs('./BEST_MODEL', exist_ok=True)
os.makedirs('./LOSS', exist_ok=True)
os.makedirs(check_point_path, exist_ok=True)