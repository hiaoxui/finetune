import torch


def collate_fn(batch):
    def pad_seqs(seqs):
        lengths = torch.tensor(list(map(len, seqs)))
        ml = lengths.max().item()
        mask = torch.arange(ml).unsqueeze(0).expand(len(seqs), -1) < lengths.unsqueeze(1)
        ids = [seq + [0] * (ml-len(seq)) for seq in seqs]
        return torch.tensor(ids), mask

    ret = dict()
    ret['input_ids'], ret['attention_mask'] = pad_seqs([item['tgt_input_ids'] for item in batch])
    if 'skip' in batch[0]:
        ret['skip'] = torch.tensor([inp['skip'] for inp in batch])
    if 'meta' in batch[0]:
        ret['meta'] = [inp['meta'] for inp in batch]
    return ret
