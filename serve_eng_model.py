import os
import re
import string
import random
import json 
import multipart
from wsgiref.simple_server import make_server
from data_dh import get_infer_data_loader
from models.model.early_exit import Early_conformer
from util.beam_infer import BeamInference
from util.conf import get_args
from util.model_utils import *
from util.tokenizer import *
import digitalhub as dh


context = {}

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


def init():
    args = get_args()
    args.batch_size = 1
    args.n_workers = 1
    args.shuffle = False
    args.decoder_mode = 'ctc'
    context.args = args
    #setattr(context, "args", args)

    #project = dh.get_or_create_project("demo-early-exit-eng")
    #model = project.get_model("early-exit-eng-model")
    #path = model.download()

    path = "trained_model/mod032-transformer"
    model = load_model(path, args)
    context.model = model
    #setattr(context, "model", model)

    file_dict = 'librispeech.lex'
    vocab = load_dict(file_dict)
    context.vocab = vocab
    #setattr(context, "vocab", vocab)



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


def serve(path):
    # Used to access various inference functions, see util/beam_infer
    inf = BeamInference(args=context.args)

    paths = []
    paths.append(path)
    data_loader = get_infer_data_loader(args=context.args, paths=paths)

    result = run(model=context.model, args=context.args, data_loader=data_loader,
        inf=inf, vocab=context.vocab)
    
    for num, transcript in enumerate(result):
        print(f"Transcript[{num}]:{transcript}")

    if len(result) >= 1: 
        return result[0]
    else:
        return ""

def id_generator(size=8, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def simple_app(environ, start_response):
    results = []
    fields = {}
    files = {}
    def on_field(field):
        fields[field.field_name] = field.value
    def on_file(file):
        files[file.field_name] = {'name': file.file_name.decode("utf-8"), 'file_object': file.file_object}

    if environ['REQUEST_METHOD'] == 'POST':
        multipart_headers = {'Content-Type': environ['CONTENT_TYPE']}
        multipart_headers['Content-Length'] = environ['CONTENT_LENGTH']
        multipart.parse_form(multipart_headers, environ['wsgi.input'], on_field, on_file)
        for filed_name, each_file_details in files.items():
            filename = "upload/" + id_generator() + "_" + each_file_details['name']
            with open(filename, 'wb') as f:
                uploaded_file = each_file_details['file_object']
                uploaded_file.seek(0)
                f.write(uploaded_file.read())            
            
        trasncript = serve(filename)  
        info = {}
        info.filename = filename
        info.trasncript = trasncript
        results.append(info)
        
        if os.path.exists(filename):
            os.remove(filename)

    status = '200 OK'
    headers = [('Content-type', 'application/json; charset=utf-8')]
    content = json.dumps(results)
    content = [content.encode('utf-8')]
    start_response(status, headers)
    return content


init()
with make_server('', 8051, serve) as httpd:
    print("Serving on port 8051...")
    httpd.serve_forever()