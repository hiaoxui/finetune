from typing import Tuple, Dict, Any

from torch.utils import data

from lmft.utils.lazy_dataset import LazyDataset
from lmft.utils.lazy_tokenizer import LazyTokenizer
from lmft.utils.chat import ChatFactory, ChatPart, Chat, ChatInput
from .utils import collate_fn


class ClercDataset(LazyDataset, LazyTokenizer):
    def __init__(
            self, pretrained: str, split: str, max_length: int, use_ref: bool = True,
            max_size: int = 999999999999,
    ):
        super().__init__(('jhu-clsp/CLERC',), {'data_dir': 'generation'}, split=split)
        self.pretrained, self.max_length, self.use_ref = pretrained, max_length, use_ref
        self.tokenizer_kwargs = {'truncation': True, 'padding': 'left', 'legacy': False}
        self.chat_factory = ChatFactory(self.pretrained, max_tokens=max_length)
        self.use_ref = use_ref
        self.max_size = max_size

    def __len__(self):
        return min(self.max_size, super().__len__())

    def example_text(self, idx: int) -> Tuple[ChatInput, Dict[str, Any]]:
        # it constructs the prompt for text continuation
        # it returns `ref_start` and `ref_end`, which indicates the char idx range of the references.
        # If we need to truncate the text, we consider truncating the reference parat.
        ex = self.data[idx]
        prev, refs = ex['previous_text'], ex['short_citations']
        if not self.use_ref:
            user_chat = Chat(role='user', parts=[
                ChatPart('Below is a legal case that I have written so far:\n'),
                ChatPart(prev, True, 'left', 512, 5),
                ChatPart(
                    'Continue to write it following the style of my writeup. Your answer contains 100 to 400 words. ' +
                    'Wrap your answer with <answer></answer>. Make your answer concise and avoid redundant languages.'
                ),
            ])
        else:
            ref_texts = []
            ref_ids = []
            for i, ref in enumerate(refs):
                id_idx = ref.index('\n')
                cid, ref_text = ref[:id_idx].strip(), ref[id_idx:].strip()
                ref_texts.append(f'# Reference case {cid}\n{ref_text}\n')
                ref_ids.append(cid)
            ref_text = '\n'.join(ref_texts)

            user_chat = Chat(role='user', parts=[
                ChatPart('Below are some reference articles for legal cases:\n'),
                ChatPart(ref_text, True, 'right', 0, 4),
                ChatPart('\nHere is the case that I have written so far: \n'),
                ChatPart(prev + '\n', True, 'left', 512, 5),
                ChatPart(
                    'Continue to write it following the style of my writeup. Your answer contains 100 to 400 words. ' +
                    'You must explicitly use the reference cases and mention their reference ids, ' +
                    'i.e. ' + ', '.join(ref_ids) + '. ' +
                    'Wrap your answer with <answer></answer>. Make your answer concise and avoid redundant languages.'
                ),
            ])

        assistant_chat = Chat(
            role='assistant', parts=[
                ChatPart('<answer>' + ex['gold_text'] + '</answer>', prefix_split=True)
            ]
        )
        chats = ChatInput([user_chat, assistant_chat])
        meta = {'gold_text': ex['gold_text'], 'docid': ex['docid'], 'index': (self._split, idx)}
        return chats, meta

    def __getitem__(self, idx: int):
        chats, meta = self.example_text(idx)
        processed = self.chat_factory.process(chats, return_text=False)
        return {'input_ids': processed.input_ids, 'skip': processed.prefix, 'meta': meta}


def load_clerc_data(
        bsz: int, pretrained: str, max_length: int, use_ref: bool, n_val: int
) -> Tuple[data.DataLoader, data.DataLoader]:
    # training
    train_data = ClercDataset(
        pretrained=pretrained, max_length=max_length, use_ref=use_ref, split='train', max_size=99999999999999
    )
    train_loader = data.DataLoader(
        train_data, batch_size=bsz, shuffle=False, num_workers=4, collate_fn=collate_fn, prefetch_factor=32
    )
    # test
    test_data = ClercDataset(
        pretrained=pretrained, max_length=max_length, use_ref=use_ref, split='test', max_size=n_val
    )
    test_loader = data.DataLoader(
        test_data, batch_size=bsz, shuffle=False, num_workers=4, collate_fn=collate_fn, prefetch_factor=32
    )
    return train_loader, test_loader
