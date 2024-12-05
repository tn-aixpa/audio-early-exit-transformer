import torch
import torchaudio
import torchaudio.transforms as T

class DHDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        super(DHDataset).__init__()
        print(f"DHDataset:{paths}")
        self.paths = paths
        self.data = []
        for num, path in enumerate(paths):
            waveform, sample_rate = torchaudio.load(path)
            label = ""
            spk_id = num
            ut_id = num
            tuple = (waveform, sample_rate, label, spk_id, ut_id)
            self.data.append(tuple)
    
    def __len__(self):             
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

class DHCollateInferFn(object):

    def spec_transform(self, waveform):
        spec_t = T.Spectrogram(n_fft=self.args.n_fft * 2,
                            hop_length=self.args.hop_length,
                            win_length=self.args.win_length)
        return spec_t(waveform)

    def melspec_transform(self, waveform):
        melspec_t = T.MelScale(sample_rate=self.args.sample_rate,
                            n_mels=self.args.n_mels,
                            n_stft=self.args.n_fft+1)
        return melspec_t(waveform)

    def pad_sequence(self, batch, padvalue):
        # Make all tensor in a batch the same length by padding with zeros
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(
            batch, batch_first=True, padding_value=padvalue)
        return batch.permute(0, 2, 1)

    def __init__(self, args):
        self.args = args
        print(f"DHCollateInferFn:{args}")

    def __call__(self, batch,
                 SOS_token=None, EOS_token=None, PAD_token=None):
        if SOS_token == None:
            SOS_token = self.args.trg_sos_idx
        if EOS_token == None:
            EOS_token = self.args.trg_eos_idx
        if PAD_token == None:
            PAD_token = self.args.trg_pad_idx

        tensors, t_source = [], []

        # Gather in lists, and encode labels as indices
        for waveform, smp_freq, label, spk_id, ut_id, *_ in batch:
            print(f"waveform[{spk_id}]:{list(waveform.size())}")
            spec = self.spec_transform(waveform)  # .to(self.args.device)
            spec = self.melspec_transform(spec).to(self.args.device)
            t_source += [spec.size(2)]

            npads = 1000
            if spec.size(2) > 1000:
                npads = 500

            tensors += spec
            
            del spec            
            del waveform
            del label

        if tensors:
            tensors = self.pad_sequence(tensors, 0)
            return tensors.squeeze(1), torch.tensor(t_source)
        else:
            return None
        

def get_infer_data_loader(args, paths):

    try:
        train_dataset = DHDataset(paths)

        collate_infer_fn = DHCollateInferFn(args=args)
        data_loader = torch.utils.data.DataLoader(train_dataset,
                                                pin_memory=False,
                                                batch_size=args.batch_size,
                                                shuffle=args.shuffle,
                                                collate_fn=collate_infer_fn,
                                                num_workers=args.n_workers)
        return data_loader

    except Exception as e:
        exit(f"Invalid data split:{e}")