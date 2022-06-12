import os

# front-end parameter settings
win_size = 320
fft_num = 320
win_shift = 160
chunk_length = 3*16000
feat_type = 'sqrt'   # normal, sqrt, cubic, log_1x
is_conti = True  
conti_path = './CP_dir/aia_merge_dns300_conti/checkpoint_early_exit_24th.pth.tar'
is_pesq = False

# server parameter settings/home/yuguochen/dual_path_AIA_Trans/interaction_ab/CP_dir/aia_merge_dns300/
#json_dir = '/home/yuguochen/CYCLEGAN-ATT-UNET/toy_wsj_si84/Json'
#file_path = '/home/yuguochen/CYCLEGAN-ATT-UNET/toy_wsj_si84'
json_dir = '/home/yuguochen/DNS_dataset_300h/Json'
file_path = '/home/yuguochen/DNS_dataset_300h/'
loss_dir = './LOSS/aia_merge_dns300_conti2.mat'
batch_size = 2
epochs = 50
lr = 5e-4
model_best_path = './BEST_MODEL/aia_merge_dns300_conti.pth.tar'
check_point_path = './CP_dir/aia_merge_dns300_conti'

os.makedirs('./BEST_MODEL', exist_ok=True)
os.makedirs('./LOSS', exist_ok=True)
os.makedirs(check_point_path, exist_ok=True)