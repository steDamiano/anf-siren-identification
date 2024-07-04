import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import soundfile
from scipy import signal
from pathlib import Path
from anfsd.anf.kalmanf import kalmanf
import torchaudio
        
class SirenDatasetANF(Dataset):
    def __init__(
            self,
            root: str,
            index: str,
            sample_rate: int = 16000,
            rho: float = 0.95,
            q: float = 5e-5,
            r: float = 5e-1,
            q_down: int = 20,
            segment_duration: float = 2.0,
            num_channels: int = 1,
            split: int | None = None,
            fold: int = 0
    ):

        super().__init__()

        
        self._root = Path(root)
        
        if split is not None:
            self.df = pd.read_csv(index[:index.rfind('/')+1] + 'splits/' + str(split) + '/' + index[index.rfind('/')+1:-4]+'_fold_'+str(fold)+'.csv')
        else:
            self.df = pd.read_csv(index)
        self.sample_rate = sample_rate
        self.rho = rho
        self.q = q
        self.r = r
        self.q_down = q_down
        self.class_labels = ['noise', 'siren']
        self.segment_duration = segment_duration
        self.num_channels = num_channels

    
    def __len__(self):
        return len(self.df)

    
    def __getitem__(self, index: int):
        
        cur_sample = self.df.iloc[index]
        
        f_inst = self._get_instfreq(cur_sample)
        f_inst = torch.as_tensor(f_inst, dtype=torch.float32)

        label = torch.as_tensor(self.class_labels.index(cur_sample['file_class']), dtype=torch.long)
        
        return {"audio": f_inst, "label": label}
    
    def _get_instfreq(
        self, 
        current_sample: pd.Series
    ):
        file_path = current_sample['file_path']

        # Load audio
        audio, _ = soundfile.read(self._root.joinpath(file_path))

        # Cut out segment
        if 'audio_start' in self.df.keys():
            onset = int(current_sample['audio_start'] * self.sample_rate)
            offset = int(current_sample['audio_end'] * self.sample_rate)
            audio = audio[onset:offset]
        else:
            while len(audio) < self.segment_duration * self.sample_rate:
                audio = np.tile(audio, 2)
            audio_start = np.random.randint(0, round(len(audio) - self.segment_duration * self.sample_rate))
            audio = audio[round(audio_start):round(audio_start + self.segment_duration * self.sample_rate)]

        # Apply normalization based on power
        norm_factor =  np.sqrt(np.mean(audio**2))       
        if norm_factor > 0:
            audio /= norm_factor

        if self.num_channels == 1:
            ########## 1 CHANNEL, MODULATED ##########

            # Apply the KalmANF algorithm
            f_kal, a_kal, e_kal, _ = kalmanf(audio, self.sample_rate, self.rho, q=self.q, r=self.r, num_channels=self.num_channels) 
            
            # Downsample
            f_kal = signal.resample_poly(f_kal, 1, self.q_down)
            
            # Convert to row vector
            f_kal = f_kal[None, :]

            return f_kal
        elif self.num_channels == 2:
            ###### 2 CHANNELS ##########

            # Apply the KalmANF algorithm
            f_kal, a_kal, e_kal, pow_ratio = kalmanf(audio, self.sample_rate, self.rho, q=self.q, r=self.r, num_channels=self.num_channels) 
        
            # Downsample
            f_kal = signal.resample_poly(f_kal, 1, self.q_down)
            pow_ratio = signal.resample_poly(pow_ratio, 1, self.q_down)

            # Normalization
            if max(abs(f_kal)) > 0:
                f_kal /= (self.sample_rate / 2)
            
            # Convert to row vector
            f_kal = f_kal[None, :]
            pow_ratio = pow_ratio[None, :]

            # Stack both channels
            out = np.squeeze(np.stack((f_kal, pow_ratio), axis=0))
            
            return out

        else:
            raise ValueError('num_channels must be either 1 or 2')

class SirenDataset(Dataset):
    def __init__(
            self,
            root: str,
            index: str,
            sample_rate: int = 16000,
            transforms = None,
            segment_duration: float = 2.0,
            split: int | None = None,
            fold = 0
        ):

        super().__init__()

        self._root = Path(root)
        if split is not None:
            self.df = pd.read_csv(index[:index.rfind('/')+1] + 'splits/' + str(split) + '/' + index[index.rfind('/')+1:-4]+'_fold_'+str(fold)+'.csv')
        else:
            self.df = pd.read_csv(index)

        self.sample_rate = sample_rate
        self.transforms = transforms

        self.class_labels = ['noise', 'siren']
        self.segment_duration = segment_duration

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate = sample_rate,
            n_fft = int(0.064 * sample_rate),
            win_length = int(0.064 * sample_rate),
            hop_length = int(0.032 * sample_rate),
            f_min = 20,
            f_max = sample_rate // 2,
            n_mels = 128
        )

        # conversion to dB
        self.db_scale = torchaudio.transforms.AmplitudeToDB(top_db = 80)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index: int):
        cur_sample = self.df.iloc[index]
        

        audio = self._load_audio(self._root.joinpath(cur_sample['file_path']))

        if 'audio_start' in self.df.keys():
            onset = int(self.df.loc[index, 'audio_start'] * self.sample_rate)
            offset = int(self.df.loc[index, 'audio_end'] * self.sample_rate)
            audio = audio[:,onset:offset]
        else:
            if audio.shape[1] < self.segment_duration:
                repetitions = int(self.segment_duration / audio.shape[1] + 1)
                audio = audio.repeat(1, repetitions)
            while True:
                onset = int(float(torch.rand(1,)) * (audio.shape[1] - self.segment_duration * self.sample_rate))
                offset = int(onset + self.segment_duration * self.sample_rate)
            
                if torch.count_nonzero(audio[:,onset:offset]) > 0:
                    audio = audio[:,onset:offset]
                    break
        
        label = torch.as_tensor(self.class_labels.index(cur_sample['file_class']), dtype=torch.long)

        # Apply normalization based on power
        norm_factor =  torch.sqrt(torch.mean(audio**2))       
        if norm_factor > 0:
            audio /= norm_factor


        if self.transforms is not None:
            audio = self.transforms(audio)
        audio = self.mel_spectrogram(audio)
        audio = self.db_scale(audio)
        if torch.max(torch.abs(audio)) > 0:
            audio /= torch.max(torch.abs(audio))
        
        return {'audio': audio, 'label': label}
    
    def _load_audio(self, file_path: str):
        audio, sample_rate = torchaudio.load(file_path)

        # Convert to mono if needed
        if len(audio) > 1:
            audio = audio[0]
            audio = audio[None, :]
        while len(audio[0]) < self.segment_duration * self.sample_rate:
            audio = torch.tile(audio, (1,2))
        # Resample to target sample rate
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            audio = resampler(audio)
        
        return audio