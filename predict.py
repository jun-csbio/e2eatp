import argparse
import os
import random
import time

import numpy as np
import torch
from torch import nn

import esm


def exists(fileOrFolderPath):
    return os.path.exists(fileOrFolderPath)


def set_seed(seed=-1):
    if seed == -1:
        seed = random.randint(1, 10000)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def createFolder(folder):
    if not exists(folder):
        os.makedirs(folder)


def print_namespace(anamespace, ignore_none=True):
    for key in anamespace.__dict__:
        if ignore_none and None is anamespace.__dict__[key]:
            continue
        print("{}: {}".format(key, anamespace.__dict__[key]))


def parsePredProbs(outs):
    """
    :param outs [Tensor]: [*, 2 or 1]
    :return pred_probs: [*], tgts: [*]
    """

    # 1 : one probability of each sample
    # 2 : two probabilities of each sample
    __type = 1
    if outs.size(-1) == 2:
        __type = 2
        outs = outs.view(-1, 2)
    else:
        outs = outs.view(-1, 1)

    sam_num = outs.size(0)

    outs = outs.tolist()

    pred_probs = []
    for j in range(sam_num):
        out = outs[j]
        if 2 == __type:
            prob_posi = out[1]
            prob_nega = out[0]
        else:
            prob_posi = out[0]
            prob_nega = 1.0 - prob_posi

        sum = prob_posi + prob_nega

        if sum < 1e-99:
            pred_probs.append(0.)
        else:
            pred_probs.append(prob_posi / sum)

    return pred_probs


class JSeq2ESM2:
    def __init__(self, esm2_model_path, device='cpu'):
        with torch.no_grad():
            self.esm2, alphabet = esm.pretrained.load_model_and_alphabet_local(esm2_model_path)
            self.tokenizer = alphabet.get_batch_converter()
            del alphabet

            self.esm2 = self.esm2.to(device)
            for param in self.esm2.parameters():
                param.requires_grad = False
            self.esm2.eval()

        self.device = device
        self.emb_dim = self.esm2.embed_dim
        self.layer_num = self.esm2.num_layers

    def tokenize(self, seq):
        """
        :param tuple_list: e.g., [('seq1', 'FFFFF'), ('seq2', 'AAASDA')]
        """
        tuple_list = [("seq", "{}".format(seq))]
        with torch.no_grad():
            _, _, tokens = self.tokenizer(tuple_list)
            return tokens.to(self.device)

    def embed(self, seq):
        with torch.no_grad():
            if len(seq) < 5000:
                # [B, L_rec, D]
                return self.esm2(self.tokenize(seq),
                                 repr_layers=[self.layer_num])["representations"][self.layer_num][..., 1:-1, :]
            else:
                embs = None
                for ind in range(0, len(seq), 5000):
                    sind = ind
                    eind = min(ind+5000, len(seq))
                    sub_seq = seq[sind:eind]
                    print(len(sub_seq), len(seq))
                    sub_emb = self.esm2(self.tokenize(sub_seq),
                                        repr_layers=[self.layer_num])["representations"][self.layer_num][..., 1:-1, :]
                    if None is embs:
                        embs = sub_emb
                    else:
                        embs = torch.cat([embs, sub_emb], dim=1)
                print(embs.size())
                return embs


class CoreModel(nn.Module):
    def __init__(
            self,
            in_dim=20,
            out_dim=2,
            body_num=10,
            dr=0.1):
        super(CoreModel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, 128, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(128)
        )

        self.body_num = body_num
        self.body1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, padding=0),
                nn.GELU(),
                nn.BatchNorm1d(128),
            ) for _ in range(body_num)
        ])
        self.bn_list = nn.ModuleList([
            nn.BatchNorm1d(128) for _ in range(body_num)
        ])
        self.dropout = nn.Dropout(p=dr)

        self.tail = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.Conv1d(32, out_dim, kernel_size=1)
        )

    def forward(self, x):
        """
        :param x: [B, L, C]
        :return: [B, L, O]
        """
        x = x.transpose(-1, -2).contiguous()
        x = self.conv(x)
        for bind in range(self.body_num):
            x1 = self.body1[bind](x)
            x = self.bn_list[bind](x + x1)
        x = self.dropout(x)
        x = self.tail(x).transpose(-1, -2).contiguous()
        return torch.softmax(x, dim=-1)


class JModel(nn.Module):
    def __init__(self, model):
        super(JModel, self).__init__()
        self.model = model
        self.useless = nn.Parameter(torch.zeros(1))

    def save_son_model(self, savepath):
        checkpoint = {'model': self.model.state_dict()}
        torch.save(checkpoint, savepath)

    def forward(self, x):
        x = x.to(self.useless.device)
        model = self.model.to(self.useless.device)
        return model(x)


def load_model(emb_dim=1280, body_num=5):
    model = CoreModel(
        in_dim=emb_dim,
        out_dim=2,
        body_num=body_num,
        dr=0.5
    )

    return model


def loadFasta(fasta):
    with open(fasta, 'r') as f:
        lines = f.readlines()
    ans = {}
    name = ''
    seq_list = []
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            if 1 < len(name):
                ans[name] = "".join(seq_list)
            name = line[1:]
            seq_list = []
        else:
            seq_list.append(line)
    if 0 < seq_list.__len__():
        ans[name] = "".join(seq_list)
    return ans


def dateTag():
    time_tuple = time.localtime(time.time())
    yy = time_tuple.tm_year
    mm = "{}".format(time_tuple.tm_mon)
    dd = "{}".format(time_tuple.tm_mday)
    if len(mm) < 2:
        mm = "0" + mm
    if len(dd) < 2:
        dd = "0" + dd

    date_tag = "{}{}{}".format(yy, mm, dd)
    return date_tag


def timeTag():
    time_tuple = time.localtime(time.time())
    hour = "{}".format(time_tuple.tm_hour)
    minuse = "{}".format(time_tuple.tm_min)
    second = "{}".format(time_tuple.tm_sec)
    if len(hour) < 2:
        hour = "0" + hour
    if len(minuse) < 2:
        minuse = "0" + minuse
    if len(second) < 2:
        second = "0" + second

    time_tag = "{}:{}:{}".format(hour, minuse, second)
    return time_tag


def timeRecord(time_log, content):
    date_tag = dateTag()
    time_tag = timeTag()
    with open(time_log, 'a') as file_object:
        file_object.write("{} {} says: {}\n".format(date_tag, time_tag, content))


if __name__ == '__main__':

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-sf", "--savefolder")
    parser.add_argument('-esm2m', "--esm2m")
    parser.add_argument('-e2eatpm', "--e2eatpm")
    parser.add_argument("-seq_fa", "--seq_fa")
    parser.add_argument("-sind", "--start_index", type=int, default=0)
    parser.add_argument("-eind", "--end_index", type=int, default=-1)
    parser.add_argument("-cutoff", "--prob_cutoff", type=float, default=0.48)
    parser.add_argument("-dv", "--device", default='cuda:0')
    args = parser.parse_args()

    if args.savefolder is None or args.seq_fa is None:
        parser.print_help()
        exit("PLEASE INPUT YOUR PARAMETERS CORRECTLY")

    print_namespace(args)
    set_seed(2023)
    seq_fa = args.seq_fa
    if args.esm2m is None:
        esm2m = "{}/esm2m/esm2_t33_650M_UR50D.pt".format(os.path.abspath('.'))
    else:
        esm2m = args.esm2m

    if args.e2eatpm is None:
        e2eatpm = "{}/e2eatpm/e2eatpm.pkl".format(os.path.abspath('.'))
    else:
        e2eatpm = args.e2eatpm
    savefolder = args.savefolder
    createFolder(savefolder)
    cutoff = args.prob_cutoff

    device = args.device if torch.cuda.is_available() else 'cpu'

    timeRecord("{}/run.time".format(savefolder), "Start")

    feaer = JSeq2ESM2(esm2_model_path=esm2m, device=device)
    core_model = load_model()
    if os.path.exists(e2eatpm):
        checkpoint = torch.load(e2eatpm, map_location=device)
        state_dict = checkpoint['model']

        # due to the module of body2, body3, and weights is not used in the function of forward, we reduce them
        for key in list(state_dict.keys()):
            if key.startswith('body2') or key.startswith('body3') or key.startswith('weights'):
                del state_dict[key]

        core_model.load_state_dict(state_dict)
    model = JModel(core_model).to(device)

    seq_dict = loadFasta(seq_fa)

    start_index = args.start_index
    end_index = args.end_index
    if end_index <= start_index:
        end_index = len(seq_dict)

    keys = []
    for key in seq_dict:
        keys.append(key)

    tot_seq_num = len(seq_dict)
    for ind in range(tot_seq_num):
        if ind < start_index or ind >= end_index:
            continue
        key = keys[ind]
        seq = seq_dict[key]

        if ind % 1 == 0:
            print("The {}/{}-th {}({}) is predicting...".format(ind, tot_seq_num, key, len(seq)))

        emb = feaer.embed(seq)
        out = model(emb)

        probs = parsePredProbs(out)
        filepath = "{}/{}.pred".format(savefolder, key)
        with open(filepath, 'w') as file_object:
            length = len(probs)
            file_object.write("Index    AA    Prob.    State\n")
            for i in range(length):
                aa = seq[i]
                prob = probs[i]
                if prob > cutoff:
                    file_object.write("{:5d}     {}    {:.3f}    B\n".format(i, aa, probs[i]))
                else:
                    file_object.write("{:5d}     {}    {:.3f}    N\n".format(i, aa, probs[i]))
            file_object.close()

    timeRecord("{}/run.time".format(savefolder), "End")
    print("Hope the predicted results could help you!")
