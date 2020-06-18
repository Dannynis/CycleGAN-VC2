# -*- coding: utf-8 -*-

import os
import numpy as np
import argparse
import time
import librosa
import tqdm 

from preprocess import *
from model import CycleGAN
from data_save import world_encode_data_toLoad, world_encode_data_toSave


def remove_radical_pitch_samples(f0s,mceps,log_f0s_mean,log_f0s_std):
    print ("running radical pitch clearing on {} mceps".format(len(mceps)))
    filtered_mceps = []
    filtered_f0s = []
    filtered_out_count = 0
    total_count = 0
    for i,(f0,mcep) in enumerate(zip(f0s,mceps)):
        try:
            mask = ((np.ma.log(f0) - log_f0s_mean) ** 2 < log_f0s_std * 0.5).data
            filtered_mceps.append(mcep[mask])
            filtered_f0s.append(f0[mask])
            filtered_out_count += len(mcep)- np.sum(mask)
            total_count += len(mcep)
        except:
            print (f0.shape)
            print (traceback.format_exc())
            print (i)
    print ("filtered {} out of {}".format(filtered_out_count,total_count))
    return filtered_mceps, filtered_f0s

def load_speaker_features(file_path):

    mcep_params = np.load(file_path, allow_pickle=True)

    coded_sps = mcep_params['coded_sps']
    return coded_sps


def train(train_A_dir, train_B_dir, model_dir, model_name, random_seed, validation_A_dir, validation_B_dir, output_dir,
          tensorboard_log_dir, gen_model, MCEPs_dim, lambda_list,processed_data_dir):

    gen_loss_thres = 100.0
    np.random.seed(random_seed)
    num_epochs = 5000
    mini_batch_size = 1
    generator_learning_rate = 0.00002
    generator_learning_rate_decay = generator_learning_rate / 200000
    discriminator_learning_rate = 0.000005
    discriminator_learning_rate_decay = discriminator_learning_rate / 200000
    sampling_rate = 22050
    num_mcep = MCEPs_dim
    frame_period = 5.0
    n_frames = 128
    lambda_cycle = lambda_list[0]
    lambda_identity = lambda_list[1]

    Speaker_A_features = os.path.join(processed_data_dir, 'wav_A.npz')
    Speaker_B_features = os.path.join(processed_data_dir, 'wav_B.npz')
    start_time = time.time()
    print ('lookiong for preprocessed data in:{}'.format(processed_data_dir))
    if os.path.exists(Speaker_A_features) and os.path.exists(Speaker_B_features):
        print ('#### loading processed data #######')
        coded_sps_A = load_speaker_features(Speaker_A_features)
        coded_sps_B = load_speaker_features(Speaker_B_features)
    else:
        print('Preprocessing Data...')
        if not os.path.exists(processed_data_dir):
            os.makedirs(processed_data_dir)


        wavs_A = load_wavs(wav_dir=train_A_dir, sr=sampling_rate)

        f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = world_encode_data(wavs=wavs_A, fs=sampling_rate,
                                                                         frame_period=frame_period, coded_dim=num_mcep)
        np.savez(Speaker_A_features, coded_sps=coded_sps_A)

        del wavs_A

        wavs_B = load_wavs(wav_dir=train_B_dir, sr=sampling_rate)

        f0s_B, timeaxes_B, sps_B, aps_B, coded_sps_B = world_encode_data(wavs=wavs_B, fs=sampling_rate,
                                                                         frame_period=frame_period, coded_dim=num_mcep)
        np.savez(Speaker_B_features,coded_sps=coded_sps_B)

        del wavs_B

        print('Data preprocessing finished !')

    # log_f0s_mean_A, log_f0s_std_A = logf0_statistics(f0s_A)
    # log_f0s_mean_B, log_f0s_std_B = logf0_statistics(f0s_B)
    #
    # print('Log Pitch A')
    # print('Mean: %f, Std: %f' %(log_f0s_mean_A, log_f0s_std_A))
    # print('Log Pitch B')
    # print('Mean: %f, Std: %f' %(log_f0s_mean_B, log_f0s_std_B))
    #
    # coded_sps_A,f0s_A = remove_radical_pitch_samples(f0s_A, coded_sps_A, log_f0s_mean_A, log_f0s_std_A)
    # coded_sps_B,f0s_B = remove_radical_pitch_samples(f0s_B, coded_sps_B, log_f0s_mean_B, log_f0s_std_B)
    #
    # print('recalculating mean and std of radical cleared f0s')
    # log_f0s_mean_A, log_f0s_std_A = logf0_statistics(f0s_A)
    # log_f0s_mean_B, log_f0s_std_B = logf0_statistics(f0s_B)


    coded_sps_A_transposed = transpose_in_list(lst = coded_sps_A)
    coded_sps_B_transposed = transpose_in_list(lst = coded_sps_B)



    print("Input data fixed.")
    coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = coded_sps_normalization_fit_transoform(
        coded_sps=coded_sps_A_transposed)
    coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std = coded_sps_normalization_fit_transoform(
        coded_sps=coded_sps_B_transposed)


    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # np.savez(os.path.join(model_dir, 'logf0s_normalization.npz'), mean_A=log_f0s_mean_A, std_A=log_f0s_std_A,
    #          mean_B=log_f0s_mean_B, std_B=log_f0s_std_B)
    np.savez(os.path.join(model_dir, 'mcep_normalization.npz'), mean_A=coded_sps_A_mean, std_A=coded_sps_A_std,
             mean_B=coded_sps_B_mean, std_B=coded_sps_B_std)

    if validation_A_dir is not None:
        validation_A_output_dir = os.path.join(output_dir, 'converted_A')
        if not os.path.exists(validation_A_output_dir):
            os.makedirs(validation_A_output_dir)

    if validation_B_dir is not None:
        validation_B_output_dir = os.path.join(output_dir, 'converted_B')
        if not os.path.exists(validation_B_output_dir):
            os.makedirs(validation_B_output_dir)

    end_time = time.time()
    time_elapsed = end_time - start_time
    print('Preprocessing Done.')
    print('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % (
    time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))
    # ---------------------------------------------- Data preprocessing ---------------------------------------------- #

    # Model define
    model = CycleGAN(num_features = num_mcep, log_dir=tensorboard_log_dir, model_name=model_name, gen_model=gen_model)
    # load model
    if os.path.exists(os.path.join(model_dir, (model_name+".index"))) == True:
        model.load(filepath=os.path.join(model_dir, model_name))

    # =================================================== Training =================================================== #
    for epoch in range(num_epochs):
        print('Epoch: %d' % epoch)

        start_time_epoch = time.time()

        dataset_A, dataset_B = sample_train_data(dataset_A = coded_sps_A_norm, dataset_B = coded_sps_B_norm, n_frames = n_frames)

        n_samples = dataset_A.shape[0]
        # -------------------------------------------- one epoch learning -------------------------------------------- #
        for i in tqdm.tqdm(range(n_samples // mini_batch_size)):

            num_iterations = n_samples // mini_batch_size * epoch + i

            if num_iterations > 10000:
                lambda_identity = 0
            if num_iterations > 200000:
                generator_learning_rate = max(0, generator_learning_rate - generator_learning_rate_decay)
                discriminator_learning_rate = max(0, discriminator_learning_rate - discriminator_learning_rate_decay)

            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size

            generator_loss, discriminator_loss, generator_loss_A2B = model.train\
                (input_A = dataset_A[start:end], input_B = dataset_B[start:end],
                 lambda_cycle = lambda_cycle, lambda_identity = lambda_identity,
                 generator_learning_rate = generator_learning_rate, discriminator_learning_rate = discriminator_learning_rate)
# issue #4,
#            model.summary()

            # Minimum AtoB loss model save
            # if gen_loss_thres > generator_loss_A2B:
            #     gen_loss_thres = generator_loss_A2B
            #     best_model_name = 'Bestmodel' + model_name
            #     model.save(directory=model_dir, filename=best_model_name)
            #     print("generator loss / generator A2B loss ", generator_loss, generator_loss_A2B)

            if i % 50 == 0:
                print('Iteration: {:07d}, Generator Learning Rate: {:.7f}, Discriminator Learning Rate: {:.7f}, Generator Loss : {:.3f}, Discriminator Loss : {:.3f}'.format(num_iterations, generator_learning_rate, discriminator_learning_rate, generator_loss, discriminator_loss))

        # Last model save
        if epoch % 10 == 0:
            model.save(directory = model_dir, filename = model_name)

        end_time_epoch = time.time()
        time_elapsed_epoch = end_time_epoch - start_time_epoch

        print('Time Elapsed for This Epoch: %02d:%02d:%02d' % (time_elapsed_epoch // 3600, (time_elapsed_epoch % 3600 // 60), (time_elapsed_epoch % 60 // 1)))
        # -------------------------------------------- one epoch learning -------------------------------------------- #
        # ------------------------------------------- validation inference ------------------------------------------- #
        if validation_A_dir is not None:
            # if epoch % 50 == 0:
            if epoch % 10 == 0:
                print('Generating Validation Data B from A...')
                for file in os.listdir(validation_A_dir):
                    filepath = os.path.join(validation_A_dir, file)
                    wav, _ = librosa.load(filepath, sr = sampling_rate, mono = True)
                    wav = wav_padding(wav = wav, sr = sampling_rate, frame_period = frame_period, multiple = 4)
                    mel = encode_wav(wav=torch.Tensor(wav))
                    coded_sp_converted_norm = model.test(inputs = np.array([mel]), direction = 'A2B')[0]
                    wav_transformed = WV.mel2wav(coded_sp_converted_norm)
                    librosa.output.write_wav(os.path.join(validation_A_output_dir, os.path.basename(file)), wav_transformed, sampling_rate)
                    # break

        if validation_B_dir is not None:
            # if epoch % 50 == 0:
            if epoch % 10 == 0:
                print('Generating Validation Data A from B...')
                for file in os.listdir(validation_B_dir):
                    filepath = os.path.join(validation_B_dir, file)
                    wav, _ = librosa.load(filepath, sr=sampling_rate, mono=True)
                    wav = wav_padding(wav=wav, sr=sampling_rate, frame_period=frame_period, multiple=4)
                    mel = encode_wav(wav=torch.Tensor(wav))
                    coded_sp_converted_norm = model.test(inputs=np.array([mel]), direction='A2B')[0]
                    wav_transformed = WV.mel2wav(coded_sp_converted_norm)
                    librosa.output.write_wav(os.path.join(validation_B_output_dir, os.path.basename(file)), wav_transformed, sampling_rate)
                    # break
        # ------------------------------------------- validation inference ------------------------------------------- #
    # =================================================== Training =================================================== #


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train CycleGAN-VC2 model')

    train_A_dir_default = '/media/dan/Disk/ml+dl+dsp/Pytorch-CycleGAN-VC2/train_44k/podcast'
    train_B_dir_default = '/media/dan/Disk/ml+dl+dsp/Pytorch-CycleGAN-VC2/train_44k/nas'
    model_dir_default = './model/sf1_tf2'
    model_name_default = 'sf1_tf2.ckpt'
    random_seed_default = 0
    validation_A_dir_default = '/root/onejin/test/ma'
    validation_B_dir_default = '/root/onejin/test/fe'
    output_dir_default = './validation_output'
    tensorboard_log_dir_default = './log'
    generator_model_default = 'CycleGAN-VC2'
    MCEPs_dim_default = 80
    lambda_cycle_defalut = 10.0
    lambda_identity_defalut = 5.0
    processed_data_dir = './processed_data'

    parser.add_argument('--train_A_dir', type = str, help = 'Directory for A.', default = train_A_dir_default)
    parser.add_argument('--train_B_dir', type = str, help = 'Directory for B.', default = train_B_dir_default)
    parser.add_argument('--model_dir', type = str, help = 'Directory for saving models.', default = model_dir_default)
    parser.add_argument('--model_name', type = str, help = 'File name for saving model.', default = model_name_default)
    parser.add_argument('--random_seed', type = int, help = 'Random seed for model training.', default = random_seed_default)
    parser.add_argument('--validation_A_dir', type=str,
                        help='Convert validation A after each training epoch. If set none, no conversion would be done during the training.',
                        default=validation_A_dir_default)
    parser.add_argument('--validation_B_dir', type=str,
                        help='Convert validation B after each training epoch. If set none, no conversion would be done during the training.',
                        default=validation_B_dir_default)
    parser.add_argument('--output_dir', type = str, help = 'Output directory for converted validation voices.', default = output_dir_default)
    parser.add_argument('--tensorboard_log_dir', type = str, help = 'TensorBoard log directory.', default = tensorboard_log_dir_default)
    parser.add_argument('--gen_model', type=str, help='generator_gatedcnn / generator_gatedcnn_SAGAN', default=generator_model_default)
    parser.add_argument('--MCEPs_dim', type=int, help='input dimension', default=MCEPs_dim_default)
    parser.add_argument('--lambda_cycle', type=float, help='lambda cycle', default=lambda_cycle_defalut)
    parser.add_argument('--lambda_identity', type=float, help='lambda identity', default=lambda_identity_defalut)
    parser.add_argument('--processed_data_dir', type=str, help='processed_data_dir', default=processed_data_dir)

    argv = parser.parse_args()

    train_A_dir = argv.train_A_dir
    train_B_dir = argv.train_B_dir
    model_dir = argv.model_dir
    model_name = argv.model_name
    random_seed = argv.random_seed
    validation_A_dir = None if argv.validation_A_dir == 'None' or argv.validation_A_dir == 'none' else argv.validation_A_dir
    validation_B_dir = None if argv.validation_B_dir == 'None' or argv.validation_B_dir == 'none' else argv.validation_B_dir
    output_dir = argv.output_dir
    tensorboard_log_dir = argv.tensorboard_log_dir
    generator_model = argv.gen_model
    MCEPs_dim = argv.MCEPs_dim
    lambda_cycle = argv.lambda_cycle
    lambda_identity = argv.lambda_identity
    processed_data_dir = argv.processed_data_dir

    train(train_A_dir=train_A_dir, train_B_dir=train_B_dir, model_dir=model_dir, model_name=model_name,
          random_seed=random_seed, validation_A_dir=validation_A_dir, validation_B_dir=validation_B_dir,
          output_dir=output_dir, tensorboard_log_dir=tensorboard_log_dir, gen_model=generator_model,
          MCEPs_dim=MCEPs_dim, lambda_list=[lambda_cycle, lambda_identity],processed_data_dir=processed_data_dir)
