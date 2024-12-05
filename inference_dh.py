import os
import re
from torchaudio.utils import download_asset

from data_dh import get_infer_data_loader
from models.model.early_exit import Early_conformer
from util.beam_infer import BeamInference
from util.conf import get_args
from util.model_utils import *
from util.tokenizer import *


def evaluate_batch_ctc(args, model, batch, valid_len, inf, vocab):
    encoder = model(batch[0].to(args.device), valid_len)
    #print(f"encoder:{encoder}")
    best_combined = inf.ctc_cuda_predict(encoder[0], args.tokens)
    #print(f"best_combined:{best_combined}")
    if args.bpe == True:
        transcript = apply_lex(args.sp.decode(best_combined[0][0].tokens).lower(), vocab)
    else:
        apply_lex(re.sub(r"[#^$]+", "", best_combined[0][0].lower()), vocab)
    return transcript


def run(args, model, data_loader, inf, vocab):
    result = []
    for batch in data_loader:

        valid_len = batch[1]

        result.append(evaluate_batch_ctc(args, model, batch,
                            valid_len,  inf, vocab))

    return result


def main():
    #
    #   CONFIG
    #

    # Parse config from command line arguments
    args = get_args()
    args.batch_size = 1
    args.n_workers = 1
    args.shuffle = False

    #
    #   MODEL
    #

    model = Early_conformer(src_pad_idx=args.src_pad_idx,
                            n_enc_exits=args.n_enc_exits,
                            d_model=args.d_model,
                            enc_voc_size=args.enc_voc_size,
                            dec_voc_size=args.dec_voc_size,
                            max_len=args.max_len,
                            d_feed_forward=args.d_feed_forward,
                            n_head=args.n_heads,
                            n_enc_layers=args.n_enc_layers_per_exit,
                            features_length=args.n_mels,
                            drop_prob=args.drop_prob,
                            depthwise_kernel_size=args.depthwise_kernel_size,
                            device=args.device).to(args.device)

    # If model checkpoint path is provided, load it.
    # (Overrides --load_model-dir)
    if args.load_model_path != None:
        path = os.getcwd() + '/' + args.load_model_path
        model.load_state_dict(torch.load(
            path, map_location=args.device))

    # If model checkpoint dir is provided, check that
    # the epochs to begin and end averaging are also
    # provided. If so, average the specified models.
    elif None not in (args.load_model_dir, args.avg_model_start, args.avg_model_end):
        model = avg_models(args, model, args.load_model_dir,
                           args.avg_model_start, args.avg_model_end)

    # If neither option has been provided, then raise error.
    else:
        raise ValueError(
            "Invalid model loading config. Use either --load_model_path for a single model or --load_model_dir/--avg_model_start/--avg_model_end for an average of models.")

    model.eval()

    print(f'The model has {count_parameters(model):,} trainable parameters')
    # print("batch_size:", batch_size, " num_heads:", n_heads, " num_encoder_layers:",
    #     n_enc_layers, "vocab_size:", dec_voc_size, "DEVICE:", device)

    torch.multiprocessing.set_start_method('spawn')
    torch.set_num_threads(args.n_threads)

    # Used to access various inference functions, see util/beam_infer
    inf = BeamInference(args=args)

    file_dict = 'librispeech.lex'
    vocab = load_dict(file_dict)

    # Load data 
    SPEECH_FILE = download_asset("tutorial-assets/ctc-decoding/1688-142285-0007.wav")
    print(SPEECH_FILE)
    paths = [SPEECH_FILE]
    data_loader = get_infer_data_loader(args=args, paths=paths)

    result = run(model=model, args=args, data_loader=data_loader,
        inf=inf, vocab=vocab)
    
    for num, transcript in enumerate(result):
        print(f"Transcript[{num}]:{transcript}")


if __name__ == '__main__':
    main()
