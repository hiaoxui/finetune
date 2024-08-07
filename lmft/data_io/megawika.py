from typing import Tuple, List

from torch.utils import data

from ..utils.lazy_tokenizer import LazyTokenizer
from ..utils.lazy_dataset import LazyDataset
from ..utils.chat import ChatFactory, Chat, ChatInput, ChatPart, ChatReturn
from .utils import collate_fn


class MegaWika2Dataset(LazyDataset, LazyTokenizer):
    def __init__(self, pretrained: str, use_ref: bool, max_length: int, split: str, max_size: int = 99999999):
        super().__init__(('hiaoxui/megawika2_gen',), {'data_dir': 'short'}, split=split)
        self.pretrained, self.use_ref = pretrained, use_ref
        self.max_size, self.max_length = max_size, max_length
        self.chat_factory = ChatFactory(self.pretrained, max_tokens=max_length)

    def __len__(self):
        return min(super().__len__(), self.max_size)

    def example_text(self, dikt: dict) -> ChatInput:
        # it constructs the prompt for text continuation
        # it returns `ref_start` and `ref_end`, which indicates the char idx range of the references.
        # If we need to truncate the text, we consider truncating the reference parat.
        prev, refs = dikt['previous_text'] + '\n\n', dikt['citations']
        if not self.use_ref:
            user_chat = Chat(role='user', parts=[
                ChatPart(prev, True, 'left', min_tokens=200, truncate_priority=5),
                ChatPart(
                    f'Above is a text about "{dikt["article_title"]}". ' +
                    'Continue to write one more sentence following the style of my writeup. ' +
                    'You need to wrap your answer with <answer></answer>. '
                ),
            ])
        else:
            ref_texts = []
            for ref_idx, ref in enumerate(refs):
                ref_texts.append(f'# Reference case {ref_idx+1}\n{ref["content"]}\n')
            ref_text = '\n'.join(ref_texts)

            user_chat = Chat(role='user', parts=[
                ChatPart(prev, True, 'left', min_tokens=200, truncate_priority=5),
                ChatPart(
                    f'Above is a text about "{dikt["article_title"]}". ' +
                    'Continue to write one more sentence following the style of my writeup. ' +
                    'You need to wrap your answer with <answer></answer>. '
                    'You may use the reference information provided below: \n\n'
                ),
                ChatPart(ref_text, True, 'right', min_tokens=200, truncate_priority=2),
            ])

        gold_answer = f'<answer>{dikt["target_sentence"]}</answer>'
        assistant_chat = Chat(role='assistant', parts=[ChatPart(gold_answer, prefix_split=True)])
        chats = ChatInput([user_chat, assistant_chat])
        return chats

    def __getitem__(self, idx: int):
        datum = self.data[idx]
        meta = {'title': datum['article_title'], 'gold_text': datum['target_sentence'], 'id': datum['id']}
        chats = self.example_text(datum)
        processed = self.chat_factory.process(chats, return_text=False)
        return {'input_ids': processed.input_ids, 'skip': processed.prefix, 'meta': meta}


def load_megawika_data(
        bsz: int, pretrained: str, max_length: int, use_ref: bool, n_val: int,
        test: bool
) -> Tuple[data.DataLoader, data.DataLoader]:
    # training
    train_data = MegaWika2Dataset(
        pretrained=pretrained, max_length=max_length, use_ref=use_ref, split='train', max_size=99999999
    )
    train_loader = data.DataLoader(
        train_data, batch_size=bsz, shuffle=False, num_workers=4, collate_fn=collate_fn, prefetch_factor=32
    )
    # test
    split = 'validation' if not test else 'test'
    test_data = MegaWika2Dataset(
        pretrained=pretrained, max_length=max_length, use_ref=use_ref, split=split, max_size=n_val
    )
    test_loader = data.DataLoader(
        test_data, batch_size=bsz, shuffle=False, num_workers=16, collate_fn=collate_fn, prefetch_factor=32
    )
    return train_loader, test_loader
