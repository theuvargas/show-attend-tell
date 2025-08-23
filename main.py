import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    from datasets import load_dataset
    from collections import Counter
    from tqdm import tqdm
    import re
    from torch.utils.data import Dataset, DataLoader
    from torch.nn.utils.rnn import pad_sequence
    from torchvision import transforms
    return (
        Counter,
        DataLoader,
        Dataset,
        load_dataset,
        pad_sequence,
        re,
        torch,
        tqdm,
        transforms,
    )


@app.cell
def _(load_dataset):
    ds = load_dataset("jxie/flickr8k")
    return (ds,)


@app.cell
def _(Counter, re, torch, tqdm):
    class Vocabulary:
        def __init__(self, min_freq):
            self.min_freq = min_freq
            self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
            self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
            self.word_freq = Counter()

        def add_word(self, word):
            self.word_freq[word] += 1

        def build_vocabulary(self):
            idx = 4

            words = sorted(
                self.word_freq.keys(),
                key=lambda word: self.word_freq[word],
                reverse=True,
            )

            for word in words:
                if self.word_freq[word] >= self.min_freq:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1

        def __len__(self):
            return len(self.word2idx)

        def encode(self, text):
            tokenized_text = re.sub(r"[^\w\s]", "", text.lower()).split()
            return (
                [self.word2idx["<start>"]]
                + [
                    self.word2idx.get(token, self.word2idx["<unk>"])
                    for token in tokenized_text
                ]
                + [self.word2idx["<end>"]]
            )

        def decode(self, seq):
            if isinstance(seq, torch.Tensor):
                seq = seq.cpu().tolist()
    
            return " ".join(
                [self.idx2word.get(idx, "<unk>") for idx in seq if idx >= 3]
            )


    def build_vocab_from_dataset(dataset, min_freq=5):
        vocab = Vocabulary(min_freq)

        for item in tqdm(dataset):
            for i in range(5):
                caption_key = f"caption_{i}"
                caption = item[caption_key]

                # remove pontuação
                caption = re.sub(r"[^\w\s]", "", caption)

                for token in caption.lower().split():
                    vocab.add_word(token)

        vocab.build_vocabulary()

        return vocab
    return (build_vocab_from_dataset,)


@app.cell
def _(build_vocab_from_dataset, ds):
    vocab = build_vocab_from_dataset(ds["train"], 5)
    return (vocab,)


@app.cell
def _(Dataset, torch, vocab):
    class FlickrDataset(Dataset):
        def __init__(self, dataset, vocab, transform=None):
            self.dataset = dataset
            self.vocab = vocab
            self.transform = transform

        def __len__(self):
            return len(self.dataset) * 5

        def __getitem__(self, idx):
            image_idx = idx // 5
            caption_idx = idx % 5

            item = self.dataset[image_idx]
            image = item["image"].convert("RGB")

            if self.transform:
                image = self.transform(image)

            caption_key = f"caption_{caption_idx}"
            caption_str = item[caption_key]

            tokens = vocab.encode(caption_str)

            caption_tensor = torch.tensor(tokens)

            return image, caption_tensor
    return (FlickrDataset,)


@app.cell
def _(pad_sequence, torch):
    class CaptionCollate:
        def __call__(self, batch):
            images = [item[0].unsqueeze(0) for item in batch]
            captions = [item[1] for item in batch]

            images = torch.cat(images, dim=0)

            padded_captions = pad_sequence(
                captions, 
                batch_first=True,
                padding_value=0
            )

            return images, padded_captions
    return (CaptionCollate,)


@app.cell
def _(CaptionCollate, FlickrDataset, ds, transforms, vocab):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    flickr_train = FlickrDataset(
        dataset=ds["train"],
        vocab=vocab,
        transform=transform
    )

    collate_fn = CaptionCollate()
    return collate_fn, flickr_train


@app.cell
def _(DataLoader, collate_fn, flickr_train):
    batch_size = 32
    train_loader = DataLoader(
        dataset=flickr_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )
    return (train_loader,)


@app.cell
def _(train_loader):
    images, captions = next(iter(train_loader))
    return (captions,)


@app.cell
def _(captions, vocab):
    print(captions[2])
    print(vocab.decode(captions[2]))
    return


if __name__ == "__main__":
    app.run()
