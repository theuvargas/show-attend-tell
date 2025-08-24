import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import numpy as np
    from datasets import load_dataset
    from collections import Counter
    from tqdm import tqdm
    import re
    from torch.utils.data import Dataset, DataLoader
    from torch.nn.utils.rnn import pad_sequence
    from torchvision import transforms
    from torchvision.models import regnet_y_1_6gf, RegNet_Y_1_6GF_Weights
    return (
        Counter,
        DataLoader,
        Dataset,
        RegNet_Y_1_6GF_Weights,
        load_dataset,
        mo,
        nn,
        pad_sequence,
        re,
        regnet_y_1_6gf,
        torch,
        tqdm,
        transforms,
    )


@app.cell
def _(load_dataset):
    ds = load_dataset("jxie/flickr8k")
    return (ds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Vocabulário""")
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Data loading""")
    return


@app.cell
def _(Dataset, torch):
    class FlickrDataset(Dataset):
        def __init__(self, dataset, vocab, transform):
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

            image = self.transform(image)

            caption_key = f"caption_{caption_idx}"
            caption_str = item[caption_key]

            tokens = self.vocab.encode(caption_str)

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
def _(
    CaptionCollate,
    FlickrDataset,
    RegNet_Y_1_6GF_Weights,
    build_vocab_from_dataset,
    ds,
):
    vocab = build_vocab_from_dataset(ds["train"], 5)

    weights = RegNet_Y_1_6GF_Weights.DEFAULT

    transform = weights.transforms()

    flickr_train = FlickrDataset(
        dataset=ds["train"],
        vocab=vocab,
        transform=transform
    )

    collate_fn = CaptionCollate()
    return collate_fn, flickr_train, vocab


@app.cell
def _(DataLoader, collate_fn, flickr_train):
    batch_size = 256
    train_loader = DataLoader(
        dataset=flickr_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        collate_fn=collate_fn
    )
    return (train_loader,)


@app.cell
def _(train_loader):
    images, captions = next(iter(train_loader))
    return captions, images


@app.cell
def _(captions, vocab):
    print(vocab.decode(captions[0]))
    return


@app.cell
def _(images):
    images[0].shape
    return


@app.cell
def _(transforms):
    def tensor2pil(tensor):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
        tensor_copy = tensor.clone() 
        for t, m, s in zip(tensor_copy, mean, std):
            t.mul_(s).add_(m) # t = t * s + m
    
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(tensor_copy)
    
        return pil_image
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Encoder""")
    return


@app.cell
def _(regnet_y_1_6gf):
    regnet_y_1_6gf()
    return


@app.cell
def _(RegNet_Y_1_6GF_Weights, nn, regnet_y_1_6gf):
    class Encoder(nn.Module):
        def __init__(self, is_trainable=False):
            super(Encoder, self).__init__()

            weights = RegNet_Y_1_6GF_Weights.DEFAULT
            regnet = regnet_y_1_6gf(weights=weights)

            self.stem = regnet.stem
            self.trunk = regnet.trunk_output

            self.output_dim = regnet.fc.in_features # 888

            if not is_trainable:
                for param in self.parameters():
                    param.requires_grad = False
                self.eval()

        def forward(self, images):
            # passa as imagens pela regnet
            features = self.stem(images)
            features = self.trunk(features)

            # manipulação do shape
            # objetivo: transformar de (batch, dimensões, altura, largura)
            # para (batch, altura x largura, dimensões)
            features = features.flatten(start_dim=2)
            features = features.permute(0, 2, 1)

            return features
    return (Encoder,)


@app.cell
def _(Encoder):
    encoder = Encoder(is_trainable=False)
    return (encoder,)


@app.cell
def _(encoder, torch):
    dummy_images = torch.randn(4, 3, 224, 224)
    features = encoder(dummy_images)
    return (features,)


@app.cell
def _(features):
    features.shape
    return


@app.cell
def _(encoder):
    sum(param.numel() for param in encoder.parameters())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Atenção""")
    return


@app.cell
def _(nn):
    class Attention(nn.Module):
        def __init__(self, encoder_dim, decoder_dim, attention_dim):
            super(Attention, self).__init__()

            self.encoder_projection = nn.Linear(encoder_dim, attention_dim)
            self.decoder_projection = nn.Linear(decoder_dim, attention_dim)

            self.attn_scores_layer = nn.Linear(attention_dim, 1)

            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(dim=1)

        def forward(self, encoder_features, decoder_hidden):
            """
            Args:
                encoder_features: dims (B, 49, 888)
                decoder_features: dims (B, 512)
            """
        
            # projeta os vetores do encoder e o do estado do decoder
            # para o mesmo espaço
            encoder_vec = self.encoder_projection(encoder_features)
            decoder_vec = self.decoder_projection(decoder_hidden)

            combined = self.relu(encoder_vec + decoder_vec.unsqueeze(1))

            # computa os scores de atenção para cada local da imagem
            attn_scores = self.attn_scores_layer(combined)
            attn_scores = attn_scores.squeeze(2)

            # normaliza os scores
            alpha = self.softmax(attn_scores)

            # vetor de contexto: soma ponderada (pela atenção) das features da imagem
            context_vec = (encoder_features * alpha.unsqueeze(2)).sum(dim=1)

            return context_vec, alpha
    return (Attention,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Decoder""")
    return


@app.cell
def _(Attention, nn, torch):
    class Decoder(nn.Module):
        def __init__(
            self,
            embed_dim: int,
            encoder_dim: int,
            decoder_dim: int,
            attention_dim: int,
            vocab_size: int,
            dropout=0.5,
        ):
            super(Decoder, self).__init__()

            self.embed_dim = embed_dim
            self.encoder_dim = encoder_dim
            self.decoder_dim = decoder_dim
            self.attention_dim = attention_dim
            self.vocab_size = vocab_size

            self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.dropout = nn.Dropout(p=dropout)
            # input é um token + vetor de contexto
            self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)

            # gate do contexto
            self.f_beta = nn.Linear(decoder_dim, encoder_dim)
            self.sigmoid = nn.Sigmoid()

            self.init_h = nn.Linear(encoder_dim, decoder_dim)
            self.init_c = nn.Linear(encoder_dim, decoder_dim)

            # mapeia o output da RNN para o vocabulário
            self.fc = nn.Linear(decoder_dim, vocab_size)

            self.init_weights()

        def init_weights(self):
            self.embedding.weight.data.uniform_(-0.1, 0.1)
            self.fc.bias.data.fill_(0)
            self.fc.weight.data.uniform_(-0.1, 0.1)

        def init_hidden_state(self, encoder_features):
            mean_encoder_features = encoder_features.mean(dim=1)
            h = self.init_h(mean_encoder_features)
            c = self.init_c(mean_encoder_features)
            return h, c

        def forward(self, encoder_features, encoded_captions, caption_lengths):
            batch_size = encoder_features.size(0)
        
            h, c = self.init_hidden_state(encoder_features)

            embeddings = self.embedding(encoded_captions)
        
            decode_length = max(caption_lengths) - 1

            # decode_length previsões de tamanho vocab_size
            predictions = torch.zeros(batch_size, decode_length, self.vocab_size).to(encoder_features.device)
            # um alpha para cada região da imagem, para cada palavra
            alphas = torch.zeros(batch_size, decode_length, encoder_features.size(1)).to(encoder_features.device)

            for t in range(decode_length):
                word_embedding_t = embeddings[:, t, :]

                # vetor de contexto através da atenção entre as features extraídas
                # pelo encoder e o estado oculto do decoder
                context_vector, alpha = self.attention(encoder_features, h)

                # 'knowing when to look'
                gate = self.sigmoid(self.f_beta(h))
                gated_context = gate * context_vector

                # o input para a LSTM é a concatenação do embedding
                # e com o contexto
                lstm_input = torch.cat((word_embedding_t, gated_context), dim=1)

                # um passo da célula LSTM atualiza o hidden state
                # depois, o hidden state é projetado no espaço do vocabulário por uma camada fully connected
                h, c = self.decode_step(lstm_input, (h, c))
                preds_t = self.fc(self.dropout(h))

                predictions[:, t, :] = preds_t
                alphas[:, t, :] = alpha
            
            return predictions, encoded_captions, caption_lengths, alphas
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
