import torch
from torch.utils.data import DataLoader
import csv
from dateutil import parser
import numpy as np
import time
import random
import os
#
import json



def parse_PDB_biounits(x, atoms=['N','CA','C'], chain=None):
  '''
  input:  x = PDB filename
          atoms = atoms to extract (optional)
  output: (length, atoms, coords=(x,y,z)), sequence
  '''

  alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
  states = len(alpha_1)
  alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
             'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']
  
  aa_1_N = {a:n for n,a in enumerate(alpha_1)}
  aa_3_N = {a:n for n,a in enumerate(alpha_3)}
  aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
  aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
  aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}
  
  def AA_to_N(x):
    # ["ARND"] -> [[0,1,2,3]]
    x = np.array(x);
    if x.ndim == 0: x = x[None]
    return [[aa_1_N.get(a, states-1) for a in y] for y in x]
  
  def N_to_AA(x):
    # [[0,1,2,3]] -> ["ARND"]
    x = np.array(x);
    if x.ndim == 1: x = x[None]
    return ["".join([aa_N_1.get(a,"-") for a in y]) for y in x]

  xyz,seq,min_resn,max_resn = {},{},1e6,-1e6
  for line in open(x,"rb"):
    line = line.decode("utf-8","ignore").rstrip()

    if line[:6] == "HETATM" and line[17:17+3] == "MSE":
      line = line.replace("HETATM","ATOM  ")
      line = line.replace("MSE","MET")

    if line[:4] == "ATOM":
      ch = line[21:22]
      if ch == chain or chain is None:
        atom = line[12:12+4].strip()
        resi = line[17:17+3]
        resn = line[22:22+5].strip()
        x,y,z = [float(line[i:(i+8)]) for i in [30,38,46]]

        if resn[-1].isalpha(): 
            resa,resn = resn[-1],int(resn[:-1])-1
        else: 
            resa,resn = "",int(resn)-1
#         resn = int(resn)
        if resn < min_resn: 
            min_resn = resn
        if resn > max_resn: 
            max_resn = resn
        if resn not in xyz: 
            xyz[resn] = {}
        if resa not in xyz[resn]: 
            xyz[resn][resa] = {}
        if resn not in seq: 
            seq[resn] = {}
        if resa not in seq[resn]: 
            seq[resn][resa] = resi

        if atom not in xyz[resn][resa]:
          xyz[resn][resa][atom] = np.array([x,y,z])

  # convert to numpy arrays, fill in missing values
  seq_,xyz_ = [],[]
  try:
      for resn in range(min_resn,max_resn+1):
        if resn in seq:
          for k in sorted(seq[resn]): seq_.append(aa_3_N.get(seq[resn][k],20))
        else: seq_.append(20)
        if resn in xyz:
          for k in sorted(xyz[resn]):
            for atom in atoms:
              if atom in xyz[resn][k]: xyz_.append(xyz[resn][k][atom])
              else: xyz_.append(np.full(3,np.nan))
        else:
          for atom in atoms: xyz_.append(np.full(3,np.nan))
      return np.array(xyz_).reshape(-1,len(atoms),3), N_to_AA(np.array(seq_))
  except TypeError:
      return 'no_chain', 'no_chain'

def parse_PDB(path_to_pdb, input_chain_list=None, ca_only=False):
    c=0
    pdb_dict_list = []
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet
     
    if input_chain_list:
        chain_alphabet = input_chain_list  
 

    biounit_names = [path_to_pdb]
    for biounit in biounit_names:
        my_dict = {}
        s = 0
        concat_seq = ''
        concat_N = []
        concat_CA = []
        concat_C = []
        concat_O = []
        concat_mask = []
        coords_dict = {}
        for letter in chain_alphabet:
            if ca_only:
                sidechain_atoms = ['CA']
            else:
                sidechain_atoms = ['N', 'CA', 'C', 'O']
            xyz, seq = parse_PDB_biounits(biounit, atoms=sidechain_atoms, chain=letter)
            if type(xyz) != str:
                concat_seq += seq[0]
                my_dict['seq_chain_'+letter]=seq[0]
                coords_dict_chain = {}
                if ca_only:
                    coords_dict_chain['CA_chain_'+letter]=xyz.tolist()
                else:
                    coords_dict_chain['N_chain_' + letter] = xyz[:, 0, :].tolist()
                    coords_dict_chain['CA_chain_' + letter] = xyz[:, 1, :].tolist()
                    coords_dict_chain['C_chain_' + letter] = xyz[:, 2, :].tolist()
                    coords_dict_chain['O_chain_' + letter] = xyz[:, 3, :].tolist()
                my_dict['coords_chain_'+letter]=coords_dict_chain
                s += 1
        fi = biounit.rfind("/")
        my_dict['name']=biounit[(fi+1):-4]
        my_dict['num_of_chains'] = s
        my_dict['seq'] = concat_seq
        if s <= len(chain_alphabet):
            pdb_dict_list.append(my_dict)
            c+=1
    return pdb_dict_list

class StructureDataset():
    # def __init__(self, pdb_dict_list, verbose=True, truncate=None, max_length=100,
    #     alphabet='ACDEFGHIKLMNPQRSTVWYX'):
    def __init__(self, jsonl_file, verbose=True, truncate=None, max_length=100,
                        alphabet='ACDEFGHIKLMNPQRSTVWYX-'):
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }
        
        with open(jsonl_file) as f:#
            self.data = []

            lines = f.readlines()#

            start = time.time()
            for i, line in enumerate(lines):#
            # for i, entry in enumerate(pdb_dict_list):
                entry = json.loads(line)#
                seq = entry['seq']
                name = entry['name']

                bad_chars = set([s for s in seq]).difference(alphabet_set)
                if len(bad_chars) == 0:
                    if len(entry['seq']) <= max_length:
                        self.data.append(entry)
                    else:
                        # discard_count['too_long'] += 1
                        discard_count['bad_seq_length'] += 1
                else:
                    if verbose:#
                        print(name, bad_chars, entry['seq'])#
                    #print(name, bad_chars, entry['seq'])
                    discard_count['bad_chars'] += 1

                # Truncate early
                if truncate is not None and len(self.data) == truncate:
                    return

                if verbose and (i + 1) % 1000 == 0:
                    elapsed = time.time() - start
                    #print('{} entries ({} loaded) in {:.1f} s'.format(len(self.data), i+1, elapsed))
            #
            # if verbose:#
            #     print('discarded', discard_count)#

                #print('Discarded', discard_count)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class StructureLoader():
    def __init__(self, dataset, batch_size=100, shuffle=True,
        collate_fn=lambda x:x, drop_last=False):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.batch_size = batch_size
        sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
                batch_max = size
            else:
                clusters.append(batch)
                batch, batch_max = [], 0
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch


def worker_init_fn(worker_id):
    np.random.seed()

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

def get_std_opt(parameters, d_model, step):
    return NoamOpt(
        d_model, 2, 4000, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step
    )




def get_pdbs(data_loader, repeat=1, max_length=10000, num_units=1000000):
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet
    c = 0
    c1 = 0
    pdb_dict_list = []
    t0 = time.time()
    for _ in range(repeat):
        for step,t in enumerate(data_loader):
            t = {k:v[0] for k,v in t.items()}
            c1 += 1
            if 'label' in list(t):
                my_dict = {}
                s = 0
                concat_seq = ''
                concat_N = []
                concat_CA = []
                concat_C = []
                concat_O = []
                concat_mask = []
                coords_dict = {}
                mask_list = []
                visible_list = []
                if len(list(np.unique(t['idx']))) < 352:
                    for idx in list(np.unique(t['idx'])):
                        letter = chain_alphabet[idx]
                        res = np.argwhere(t['idx']==idx)
                        initial_sequence= "".join(list(np.array(list(t['seq']))[res][0,]))
                        if initial_sequence[-6:] == "HHHHHH":
                            res = res[:,:-6]
                        if initial_sequence[0:6] == "HHHHHH":
                            res = res[:,6:]
                        if initial_sequence[-7:-1] == "HHHHHH":
                           res = res[:,:-7]
                        if initial_sequence[-8:-2] == "HHHHHH":
                           res = res[:,:-8]
                        if initial_sequence[-9:-3] == "HHHHHH":
                           res = res[:,:-9]
                        if initial_sequence[-10:-4] == "HHHHHH":
                           res = res[:,:-10]
                        if initial_sequence[1:7] == "HHHHHH":
                            res = res[:,7:]
                        if initial_sequence[2:8] == "HHHHHH":
                            res = res[:,8:]
                        if initial_sequence[3:9] == "HHHHHH":
                            res = res[:,9:]
                        if initial_sequence[4:10] == "HHHHHH":
                            res = res[:,10:]
                        if res.shape[1] < 4:
                            pass
                        else:
                            my_dict['seq_chain_'+letter]= "".join(list(np.array(list(t['seq']))[res][0,]))
                            concat_seq += my_dict['seq_chain_'+letter]
                            if idx in t['masked']:
                                mask_list.append(letter)
                            else:
                                visible_list.append(letter)
                            coords_dict_chain = {}
                            all_atoms = np.array(t['xyz'][res,])[0,] #[L, 14, 3]
                            coords_dict_chain['N_chain_'+letter]=all_atoms[:,0,:].tolist()
                            coords_dict_chain['CA_chain_'+letter]=all_atoms[:,1,:].tolist()
                            coords_dict_chain['C_chain_'+letter]=all_atoms[:,2,:].tolist()
                            coords_dict_chain['O_chain_'+letter]=all_atoms[:,3,:].tolist()
                            my_dict['coords_chain_'+letter]=coords_dict_chain
                    my_dict['name']= t['label']
                    my_dict['masked_list']= mask_list
                    my_dict['visible_list']= visible_list
                    my_dict['num_of_chains'] = len(mask_list) + len(visible_list)
                    my_dict['seq'] = concat_seq
                    if len(concat_seq) <= max_length:
                        pdb_dict_list.append(my_dict)
                    if len(pdb_dict_list) >= num_units:
                        break
    return pdb_dict_list



class PDB_dataset(torch.utils.data.Dataset):
    def __init__(self, IDs, loader, train_dict, params):
        self.IDs = IDs
        self.train_dict = train_dict
        self.loader = loader
        self.params = params

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        sel_idx = np.random.randint(0, len(self.train_dict[ID]))
        out = self.loader(self.train_dict[ID][sel_idx], self.params)
        return out



def loader_pdb(item,params):

    pdbid,chid = item[0].split('_')
    PREFIX = "%s/pdb/%s/%s"%(params['DIR'],pdbid[1:3],pdbid)
    
    # load metadata
    if not os.path.isfile(PREFIX+".pt"):
        return {'seq': np.zeros(5)}
    meta = torch.load(PREFIX+".pt")
    asmb_ids = meta['asmb_ids']
    asmb_chains = meta['asmb_chains']
    chids = np.array(meta['chains'])

    # find candidate assemblies which contain chid chain
    asmb_candidates = set([a for a,b in zip(asmb_ids,asmb_chains)
                           if chid in b.split(',')])

    # if the chains is missing is missing from all the assemblies
    # then return this chain alone
    if len(asmb_candidates)<1:
        chain = torch.load("%s_%s.pt"%(PREFIX,chid))
        L = len(chain['seq'])
        return {'seq'    : chain['seq'],
                'xyz'    : chain['xyz'],
                'idx'    : torch.zeros(L).int(),
                'masked' : torch.Tensor([0]).int(),
                'label'  : item[0]}

    # randomly pick one assembly from candidates
    asmb_i = random.sample(list(asmb_candidates), 1)

    # indices of selected transforms
    idx = np.where(np.array(asmb_ids)==asmb_i)[0]

    # load relevant chains
    chains = {c:torch.load("%s_%s.pt"%(PREFIX,c))
              for i in idx for c in asmb_chains[i]
              if c in meta['chains']}

    # generate assembly
    asmb = {}
    for k in idx:

        # pick k-th xform
        xform = meta['asmb_xform%d'%k]
        u = xform[:,:3,:3]
        r = xform[:,:3,3]

        # select chains which k-th xform should be applied to
        s1 = set(meta['chains'])
        s2 = set(asmb_chains[k].split(','))
        chains_k = s1&s2

        # transform selected chains 
        for c in chains_k:
            try:
                xyz = chains[c]['xyz']
                xyz_ru = torch.einsum('bij,raj->brai', u, xyz) + r[:,None,None,:]
                asmb.update({(c,k,i):xyz_i for i,xyz_i in enumerate(xyz_ru)})
            except KeyError:
                return {'seq': np.zeros(5)}

    # select chains which share considerable similarity to chid
    seqid = meta['tm'][chids==chid][0,:,1]
    homo = set([ch_j for seqid_j,ch_j in zip(seqid,chids)
                if seqid_j>params['HOMO']])
    # stack all chains in the assembly together
    seq,xyz,idx,masked = "",[],[],[]
    seq_list = []
    for counter,(k,v) in enumerate(asmb.items()):
        seq += chains[k[0]]['seq']
        seq_list.append(chains[k[0]]['seq'])
        xyz.append(v)
        idx.append(torch.full((v.shape[0],),counter))
        if k[0] in homo:
            masked.append(counter)

    return {'seq'    : seq,
            'xyz'    : torch.cat(xyz,dim=0),
            'idx'    : torch.cat(idx,dim=0),
            'masked' : torch.Tensor(masked).int(),
            'label'  : item[0]}




def build_training_clusters(params, debug):
    val_ids = set([int(l) for l in open(params['VAL']).readlines()])
    test_ids = set([int(l) for l in open(params['TEST']).readlines()])
   
    if debug:
        val_ids = []
        test_ids = []
 
    # read & clean list.csv
    with open(params['LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        rows = [[r[0],r[3],int(r[4])] for r in reader
                if float(r[2])<=params['RESCUT'] and
                parser.parse(r[1])<=parser.parse(params['DATCUT'])]
    
    # compile training and validation sets
    train = {}
    valid = {}
    test = {}

    if debug:
        rows = rows[:20]
    for r in rows:
        if r[2] in val_ids:
            if r[2] in valid.keys():
                valid[r[2]].append(r[:2])
            else:
                valid[r[2]] = [r[:2]]
        elif r[2] in test_ids:
            if r[2] in test.keys():
                test[r[2]].append(r[:2])
            else:
                test[r[2]] = [r[:2]]
        else:
            if r[2] in train.keys():
                train[r[2]].append(r[:2])
            else:
                train[r[2]] = [r[:2]]
    if debug:
        valid=train       
    return train, valid, test
