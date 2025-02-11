import os
import sys
import re
import string
import random
import json 
from multipart import parse_form_data, is_form_request
from wsgiref.simple_server import make_server
from data_dh import get_infer_data_loader
from models.model.early_exit import Early_conformer
from util.beam_infer import BeamInference
from util.conf import get_args
from util.model_utils import *
from util.tokenizer import *
import digitalhub as dh


context_dict = {}

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


def init(context, model_name="early-exit-eng-model", sp_model="bpe-256.model", sp_lexicon="bpe-256.lex", sp_tokens="bpe-256.tok"):
    try:
        os.mkdir("/data/upload")
        context.logger.info("create dir data/upload")
    except OSError as error:
        context.logger.warn(f"create dir data/upload error:{error}")
    try:
        os.mkdir("/data/trained_model")
        context.logger.info("create dir data/trained_model")
    except OSError as error:
        context.logger.warn(f"create dir data/trained_model error:{error}")
    try:
        os.mkdir("/data/sentencepiece")
        context.logger.info("create dir data/sentencepiece")
    except OSError as error:
        context.logger.warn(f"create dir data/sentencepiece error:{error}")

    #project = dh.get_or_create_project(os.getenv("PROJECT_NAME"))
    project = context.project

    sp_model_artifact = context.project.get_artifact(sp_model)    
    sp_model_path = sp_model_artifact.download(destination="/data/sentencepiece", overwrite=True)

    sp_lexicon_artifact = context.project.get_artifact(sp_lexicon)    
    sp_lexicon_path = sp_lexicon_artifact.download(destination="/data/sentencepiece", overwrite=True)

    sp_tokens_artifact = context.project.get_artifact(sp_tokens)    
    sp_tokens_path = sp_tokens_artifact.download(destination="/data/sentencepiece", overwrite=True)

    args = get_args(initial_args=[], sp_model=sp_model_path, sp_lexicon=sp_lexicon_path, sp_tokens=sp_tokens_path)
    args.batch_size = 1
    args.n_workers = 1
    args.shuffle = False
    args.decoder_mode = 'ctc'
    args.model_type == 'early_conformer'

    context_dict['args'] = args

    model = project.get_model(model_name)
    path = model.download(destination="/data/trained_model", overwrite=True)
    model = load_model(path, args)
    context_dict['model'] = model

    file_dict = 'librispeech.lex'
    vocab = load_dict(file_dict)
    context_dict['vocab'] = vocab

    print(f"app context:{len(context_dict)}")


def load_model(model_path, args):
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
    
    model.load_state_dict(torch.load(model_path, map_location=args.device))

    model.eval()

    print(f'The model has {count_parameters(model):,} trainable parameters')
    # print("batch_size:", batch_size, " num_heads:", n_heads, " num_encoder_layers:",
    #     n_enc_layers, "vocab_size:", dec_voc_size, "DEVICE:", device)

    torch.multiprocessing.set_start_method('spawn')
    torch.set_num_threads(args.n_threads)

    return model


def serve_local(path):
    # Used to access various inference functions, see util/beam_infer
    inf = BeamInference(args=context_dict['args'])

    paths = []
    paths.append(path)
    data_loader = get_infer_data_loader(args=context_dict['args'], paths=paths)

    result = run(model=context_dict['model'], args=context_dict['args'], data_loader=data_loader,
        inf=inf, vocab=context_dict['vocab'])
    
    for num, transcript in enumerate(result):
        print(f"Transcript[{num}]:{transcript}")

    if len(result) >= 1: 
        return result[0]
    else:
        return ""


def serve(context, event):
    context.logger.info(f"Received event: {event.body}")
    artifact_name = event.body["name"]
    artifact = context.project.get_artifact(artifact_name)    
    path = artifact.download(destination="/data/upload", overwrite=True)
    
    transcript = serve_local(path)
    context.logger.warn(f"Transcript for file {path}:{transcript}")

    results = []
    info = {}
    info['filename'] = artifact_name
    info['transcript'] = transcript
    results.append(info)

    return results
       


def id_generator(size=8, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def simple_app(environ, start_response):
    results = []
    if is_form_request(environ):
        forms, files = parse_form_data(environ)
        for filed_name in files:
            try:
                file_details = files[filed_name]
                print(f"process file:{file_details.filename}")
                filename = "/data/upload/" + id_generator() + "_" + file_details.filename
                file_details.save_as(filename) 

                trasncript = serve_local(filename)  
                info = {}
                info['filename'] = filename
                info['trasncript'] = trasncript
                results.append(info)

                if os.path.exists(filename):
                    os.remove(filename)
            except Exception as e:
                print(e)

    status = '200 OK'
    headers = [('Content-type', 'application/json; charset=utf-8')]
    content = json.dumps(results)
    content = [content.encode('utf-8')]
    start_response(status, headers)
    return content


def main():
    args = sys.argv[1:] 
    if len(args) > 0:
        init(model_name=args[0])
    else:
        init()    

    with make_server('', 8051, simple_app) as httpd:
        print("Serving on port 8051...")
        httpd.serve_forever()


if __name__ == "__main__":
    main()