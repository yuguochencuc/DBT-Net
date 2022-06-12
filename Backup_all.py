"""
This script is the backup function used to support backup support for the SE system
Author: Andong Li
Time: 2019/05
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import librosa
import pickle
import json
import os
import h5py
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pystoi.stoi import stoi
import sys
from functools import reduce
from torch.nn.modules.module import _addindent
from config_vb import *

EPSILON = 1e-10

def calc_sp(mix, clean, data_type, Win_length, Offset_length):

    n_window= Win_length
    n_overlap = n_window- Offset_length
    c = np.sqrt(np.sum((mix ** 2) )/ len(mix))
    mix = mix / c
    clean = clean / c

    mix_x = librosa.stft(mix,
                     n_fft = n_window,
                     hop_length= n_overlap,
                     win_length= n_window,
                     window= 'hamming').T
    clean_x = librosa.stft(clean,
                           n_fft = n_window,
                           hop_length= n_overlap,
                           win_length= n_window,
                           window= 'hamming').T

    mix_angle = np.angle(mix_x)
    clean_angle = np.angle(clean_x)
    mix_x = np.abs(mix_x)
    clean_x = np.abs(clean_x)

    return data_pack(mix_x, mix_angle), data_pack(clean_x, clean_angle)

class data_pack(object):
    def __init__(self, mag, angle):
        self.mag = mag
        self.angle = angle

def contexual_frame_add(data, n_concate):
    """
    This func is to make contexual frame concatenation
    :param data: input feature matrix, size: frame x feature
    :param n_concate: =(2 * n_hop + 1)
    :return:  n_concate x feature
    """
    data = pad_with_border(data,n_concate)
    frame_num, feature_num = data.shape
    out = []
    ct = 0
    while (ct+ n_concate <= frame_num):
        out.append(data[ct: ct + n_concate])
        ct += 1
    out = np.concatenate(out , axis = 0)
    out = np.reshape(out, (-1, n_concate, feature_num))
    out = np.reshape(out, (out.shape[0], -1))
    return  out

def pad_with_border(x,n_concate):
    x_pad_list = [x[0:1]] * (n_concate-1) + [x]
    x_pad_list = np.concatenate(x_pad_list, axis= 0)
    return x_pad_list

def batch_cal_max_frame(file_infos):
    max_frame = 0
    for utter_infos in zip(file_infos):
        file_path = utter_infos[0]
        # read mat file
        mat_file = h5py.File(file_path[0])
        mix_feat = np.transpose(mat_file['mix_feat'])
        max_frame = np.max([max_frame, mix_feat.shape[0]])
    return max_frame

def de_pad(pack):
    """
    clear the zero value in each batch tensor
    Note: return is a numpy format instead of Tensor
    :return:
    """
    mix = pack.mix[0:pack.frame_list,:]
    esti = pack.esti[0:pack.frame_list,:]
    speech = pack.speech[0:pack.frame_list,:]
    return mix, esti, speech


class decode_pack(object):
    def __init__(self, mix, esti, speech, frame_list, c_list):
        self.mix = mix
        self.esti = esti
        self.speech = speech
        self.frame_list = frame_list.astype(np.int32)
        self.c_list = c_list


def recover_audio(batch_info, model, args):
    """
    This func is to recover the audio by iSTFT and overlap-add
    :param pack:  pack is a class, consisting of four components
    :param args:
    :return:
    """
    esti = model(batch_info.feats)
    real_esti, imag_esti = esti[0].cpu().numpy(), esti[1].cpu().numpy()
    real_mix, imag_mix = batch_info.feats[0].cpu().numpy(), batch_info.feats[1].cpu().numpy()
    real_speech, imag_speech = batch_info.labels[0], batch_info.labels[1]
    frame_list = batch_info.frame_list
    c_list = batch_info.c_list
    info_list = batch_info.info_list


    # The path to write audio
    if args.seen_flag == 1:
        write_out_dir = os.path.join(args.recover_space, 'seen recover')
    else:
        write_out_dir = os.path.join(args.recover_space, 'unseen recover')
    os.makedirs(write_out_dir, exist_ok = True)
    clean_write_dir = os.path.join(write_out_dir, 'clean')
    os.makedirs(clean_write_dir, exist_ok= True)
    esti_write_dir = os.path.join(write_out_dir, 'esti')
    os.makedirs(esti_write_dir, exist_ok= True)
    mix_write_dir = os.path.join(write_out_dir, 'mix')
    os.makedirs(mix_write_dir, exist_ok=True)

    for i in range(len(frame_list)):
        de_frame = frame_list[i]
        de_mix_real, de_mix_imag = real_mix[i, 0: de_frame, :], imag_mix[i, 0: de_frame, :]
        de_esti_real, de_esti_imag = real_esti[i, 0: de_frame, :], imag_esti[i, 0: de_frame, :]
        de_speech_real, de_speech_imag = real_speech[i], imag_speech[i]
        de_c = c_list[i]

        de_mix = de_mix_real + 1j* de_mix_imag
        de_speech = de_speech_real + 1j* de_speech_imag
        de_esti = de_esti_real + 1j* de_esti_imag

        mix_utt = librosa.istft((de_mix).T, hop_length=(win_size - win_shift),
                                 win_length= win_size, window='hamming').astype(np.float32)
        esti_utt = librosa.istft((de_esti).T, hop_length= (win_size- win_shift),
                                               win_length= win_size, window= 'hamming').astype(np.float32)
        clean_utt = librosa.istft((de_speech).T, hop_length= (win_size- win_shift),
                                  win_length= win_size, window= 'hamming').astype(np.float32)
        mix_utt = mix_utt * de_c
        esti_utt = esti_utt * de_c
        clean_utt = clean_utt * de_c


        filename = info_list[i]
        clean_filename = '%s_%s' % (filename.split('_', -1)[0], filename.split('_', -1)[1])
        os.makedirs(os.path.join(esti_write_dir), exist_ok=True)
        os.makedirs(os.path.join(mix_write_dir), exist_ok=True)
        os.makedirs(os.path.join(clean_write_dir), exist_ok=True)
        librosa.output.write_wav(os.path.join(esti_write_dir, '%s_enhanced.wav' % filename),
                                 esti_utt, args.fs)
        librosa.output.write_wav(os.path.join(clean_write_dir, '%s_clean.wav' % filename), clean_utt,
                                 args.fs)
        librosa.output.write_wav(os.path.join(mix_write_dir, '%s_mix.wav' % filename), mix_utt,
                                 args.fs)

def mse_loss(esti, label, nframes):
    with torch.no_grad():
        mask_for_loss_list = []
        for frame_num in nframes:
            mask_for_loss_list.append(torch.ones(frame_num, label[0].size()[2], dtype= torch.float32 ))
        mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss_list, batch_first= True).to(esti[0].device)
    esti_real = esti[0] * mask_for_loss
    esti_imag = esti[1] * mask_for_loss
    label_real = label[0] * mask_for_loss
    label_imag = label[1] * mask_for_loss
    loss = 0.5* ((esti_real - label_real) ** 2 + (esti_imag - label_imag) ** 2).sum() / mask_for_loss.sum() + EPSILON
    return loss


def summary(model, file=sys.stderr):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        if file is sys.stderr:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        print(string, file)
    return count




class train_data_clustered(object):
    def __init__(self, mix_mag, clean_mag, frame_list):
        self.mix_mag = mix_mag
        self.clean_mag = clean_mag
        self.frame_list = frame_list


class cv_data_clustered(object):
    def __init__(self, mix_spec, clean_utts, frame_list):
        self.mix_spec = mix_spec
        self.clean_utts = clean_utts
        self.frame_list = frame_list








