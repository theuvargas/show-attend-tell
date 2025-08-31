import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import numpy as np
    from datasets import load_dataset
    from collections import Counter
    from tqdm.notebook import tqdm
    import re
    from torch.utils.data import Dataset, DataLoader
    from torch.nn.utils.rnn import pad_sequence
    from torchvision import transforms
    from torchvision.models import regnet_y_1_6gf, RegNet_Y_1_6GF_Weights
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from contextlib import nullcontext
    import wandb
    from typing import Literal
    import torch.nn.functional as F
    from PIL import Image
    import matplotlib.pyplot as plt
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    return (
        Bleu,
        Cider,
        Counter,
        DataLoader,
        Dataset,
        F,
        LRScheduler,
        Literal,
        Meteor,
        Optimizer,
        RegNet_Y_1_6GF_Weights,
        Rouge,
        load_dataset,
        mo,
        nn,
        np,
        nullcontext,
        pad_sequence,
        plt,
        re,
        regnet_y_1_6gf,
        torch,
        tqdm,
        transforms,
        wandb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Vocabulário""")
    return


@app.cell
def _(Counter, re, torch):
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

            if isinstance(seq, int):
                seq = [seq]

            return " ".join(
                [self.idx2word.get(idx, "<unk>") for idx in seq if idx >= 3]
            )


    def build_vocab_from_dataset(dataset, min_freq=5):
        vocab = Vocabulary(min_freq)

        for item in dataset:
            for i in range(5):
                caption_key = f"caption_{i}"
                caption = item[caption_key]

                # remove pontuação
                caption = re.sub(r"[^\w\s]", "", caption)

                for token in caption.lower().split():
                    vocab.add_word(token)

        vocab.build_vocabulary()

        return vocab
    return Vocabulary, build_vocab_from_dataset


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
        def __init__(self, pad_idx=0):
            self.pad_idx = pad_idx

        def __call__(self, batch):
            images = [item[0].unsqueeze(0) for item in batch]
            captions = [item[1] for item in batch]

            caption_lengths = [len(cap) for cap in captions]

            padded_captions = pad_sequence(
                captions, batch_first=True, padding_value=self.pad_idx
            )

            images = torch.cat(images, dim=0)

            return images, padded_captions, caption_lengths
    return (CaptionCollate,)


@app.cell
def _(transforms):
    def tensor2pil(tensor):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        tensor_copy = tensor.clone().squeeze()
        for t, m, s in zip(tensor_copy, mean, std):
            t.mul_(s).add_(m)  # t = t * s + m

        to_pil = transforms.ToPILImage()
        pil_image = to_pil(tensor_copy)

        return pil_image
    return (tensor2pil,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Encoder""")
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
def _(Attention, Literal, nn, torch):
    class Decoder(nn.Module):
        def __init__(
            self,
            decoder_type: Literal["rnn", "lstm", "gru"],
            embed_dim: int,
            encoder_dim: int,
            decoder_dim: int,
            attention_dim: int,
            vocab_size: int,
            dropout: float,
            use_gate: bool,
        ):
            super(Decoder, self).__init__()

            self.decoder_type = decoder_type
            self.embed_dim = embed_dim
            self.encoder_dim = encoder_dim
            self.decoder_dim = decoder_dim
            self.attention_dim = attention_dim
            self.vocab_size = vocab_size
            self.use_gate = use_gate

            self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.dropout = nn.Dropout(p=dropout)

            # input é um token + vetor de contexto
            if decoder_type == "rnn":
                self.decode_step = nn.RNNCell(embed_dim + encoder_dim, decoder_dim)
            elif decoder_type == "lstm":
                self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
            elif decoder_type == "gru":
                self.decode_step = nn.GRUCell(embed_dim + encoder_dim, decoder_dim)
            else:
                raise ValueError("decoder_type deve estar em [rnn, lstm, gru]")

            if use_gate:
                # gate do contexto
                self.f_beta = nn.Linear(decoder_dim, encoder_dim)
                self.sigmoid = nn.Sigmoid()

            self.init_h = nn.Linear(encoder_dim, decoder_dim)
            if decoder_type == "lstm":
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
            if self.decoder_type == "lstm":
                c = self.init_c(mean_encoder_features)
                return h, c
            return h

        def forward(self, encoder_features, captions, caption_lengths):
            batch_size = encoder_features.size(0)

            hidden_state = self.init_hidden_state(encoder_features)

            embeddings = self.embedding(captions)

            decode_length = max(caption_lengths) - 1

            predictions = torch.zeros(
                batch_size, decode_length, self.vocab_size
            ).to(encoder_features.device)
            # um alpha para cada região da imagem, para cada palavra
            alphas = torch.zeros(
                batch_size, decode_length, encoder_features.size(1)
            ).to(encoder_features.device)

            for t in range(decode_length):
                word_embedding_t = embeddings[:, t, :]

                h = hidden_state[0] if self.decoder_type == "lstm" else hidden_state
                # vetor de contexto através da atenção entre as features extraídas
                # pelo encoder e o estado oculto do decoder
                context_vector, alpha = self.attention(encoder_features, h)

                if self.use_gate:
                    # 'knowing when to look'
                    gate = self.sigmoid(self.f_beta(h))
                    gated_context = gate * context_vector
                else:
                    gated_context = context_vector

                # o input para a RNN é a concatenação do embedding com o contexto
                rnn_input = torch.cat((word_embedding_t, gated_context), dim=1)

                # um passo da célula LSTM atualiza o hidden state
                hidden_state = self.decode_step(rnn_input, hidden_state)
                h = hidden_state[0] if self.decoder_type == "lstm" else hidden_state

                # o hidden state é projetado no espaço do vocabulário por uma camada fully connected
                preds_t = self.fc(self.dropout(h))

                predictions[:, t, :] = preds_t
                alphas[:, t, :] = alpha

            return predictions, alphas
    return (Decoder,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Encoder-Decoder""")
    return


@app.cell
def _(Decoder, Encoder, Vocabulary, nn, torch):
    class EncoderDecoder(nn.Module):
        def __init__(self, encoder: Encoder, decoder: Decoder):
            super(EncoderDecoder, self).__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, images, captions, caption_lengths):
            features = self.encoder(images)
            predictions, alphas = self.decoder(features, captions, caption_lengths)

            return predictions, alphas

        def generate_caption(self, image, vocab: Vocabulary, max_length=50):
            self.eval()
            device = image.device

            with torch.no_grad():
                features = self.encoder(image)

                hidden_state = self.decoder.init_hidden_state(features)
                h = (
                    hidden_state[0]
                    if self.decoder.decoder_type == "lstm"
                    else hidden_state
                )

                start_token_idx = vocab.word2idx["<start>"]
                input_word = torch.tensor([start_token_idx]).to(device)

                predicted_indices = []
                predicted_words = []
                alphas_list = []

                unk_token_idx = vocab.word2idx["<unk>"]

                # loop de decoding autoregressivo
                for t in range(max_length):
                    embeddings = self.decoder.embedding(input_word)

                    context_vector, alpha = self.decoder.attention(features, h)
                    alphas_list.append(alpha.cpu())

                    if self.decoder.use_gate:
                        gate = self.decoder.sigmoid(self.decoder.f_beta(h))
                        gated_context = gate * context_vector
                    else:
                        gated_context = context_vector

                    rnn_input = torch.cat((embeddings, gated_context), dim=1)

                    hidden_state = self.decoder.decode_step(rnn_input, hidden_state)
                    h = (
                        hidden_state[0]
                        if self.decoder.decoder_type == "lstm"
                        else hidden_state
                    )

                    preds_t = self.decoder.fc(h)

                    # mascara o token <unk>
                    preds_t[:, unk_token_idx] = -torch.inf

                    # seleciona a próxima palavra (greedy search)
                    predicted_idx = preds_t.argmax(dim=1)
                    predicted_indices.append(predicted_idx.item())

                    if predicted_idx.item() == vocab.word2idx["<end>"]:
                        break

                    predicted_words.append(vocab.idx2word[predicted_idx.item()])
                    # autoregressivo
                    input_word = predicted_idx

            caption = vocab.decode(predicted_indices)
            attention_plot = torch.cat(alphas_list, dim=0)

            return caption, attention_plot, predicted_words
    return (EncoderDecoder,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Trainer""")
    return


@app.cell
def _(F, np, plt, tensor2pil, wandb):
    def create_attention_maps(image_tensor, attention_plot, words):
        pil_image = tensor2pil(image_tensor)
        image_width, image_height = pil_image.size

        num_features = attention_plot.shape[1]
        feature_map_size = int(np.sqrt(num_features))

        if feature_map_size**2 != num_features:
            raise ValueError("Não é possível criar a visualização")

        attention_maps = []

        num_steps = attention_plot.shape[0]
        words_to_plot = words[:num_steps]

        for word, alpha in zip(words_to_plot, attention_plot):
            fig, ax = plt.subplots(figsize=(8, 8))

            ax.imshow(pil_image)
            ax.set_title(f"Atenção para: '{word}'", fontsize=16)
            ax.axis("off")

            alpha_map = alpha.view(1, 1, feature_map_size, feature_map_size)
            alpha_map_resized = (
                F.interpolate(
                    alpha_map,
                    size=(image_height, image_width),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze()
                .numpy()
            )

            ax.imshow(alpha_map_resized, alpha=0.6, cmap="jet")
            plt.tight_layout()

            attention_maps.append(wandb.Image(plt))

            plt.close(fig)

        return attention_maps
    return (create_attention_maps,)


@app.cell
def _(
    DataLoader,
    EncoderDecoder,
    LRScheduler,
    Optimizer,
    create_attention_maps,
    nn,
    nullcontext,
    tensor2pil,
    torch,
    tqdm,
    wandb,
):
    class Trainer:
        def __init__(
            self,
            model: EncoderDecoder,
            optimizer: Optimizer,
            criterion: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            device: str | torch.device,
            epochs: int,
            patience: int = 3,
            min_delta: float = 1e-3,
            alpha_c: float = 1.0,
            lr_scheduler: LRScheduler | None = None,
            clip_grad_norm: float | None = None,
            checkpoint_path: str = "best_model.pth",
        ):
            self.model = model.to(device)
            self.optimizer = optimizer
            self.criterion = criterion
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.device = device
            self.epochs = epochs
            self.patience = patience
            self.min_delta = min_delta
            self.no_improvement_counter = 0
            self.alpha_c = alpha_c
            self.lr_scheduler = lr_scheduler
            self.clip_grad_norm = clip_grad_norm
            self.checkpoint_path = checkpoint_path

            self.best_val_loss = float("inf")
            self.history = {"train_loss": [], "val_loss": []}

            self.val_samples = []
            images, captions, _ = next(iter(train_loader))
            for i, (img, cap) in enumerate(zip(images, captions)):
                if i == 3:
                    break
                self.val_samples.append((img.unsqueeze(0).to(device), cap))

        def _log_predictions(self, epoch: int):
            self.model.eval()
            vocab = self.val_loader.dataset.vocab

            log_table = wandb.Table(
                columns=[
                    "Época",
                    "Imagem",
                    "Legenda Gerada",
                    "Legenda de Referência",
                ]
            )

            for img_tensor, cap_tensor in self.val_samples:
                gen_cap, _, _ = self.model.generate_caption(
                    image=img_tensor, max_length=50, vocab=vocab
                )
                log_table.add_data(
                    epoch,
                    wandb.Image(tensor2pil(img_tensor)),
                    gen_cap,
                    vocab.decode(cap_tensor),
                )
            wandb.log({"predictions_table": log_table})

            img_sample, _ = self.val_samples[0]

            gen_cap, attention_plot, gen_words = self.model.generate_caption(
                image=img_sample, max_length=50, vocab=vocab
            )

            attention_maps = create_attention_maps(
                image_tensor=img_sample,
                attention_plot=attention_plot,
                words=gen_words,
            )

            if attention_maps:
                wandb.log({f"Epoch {epoch} Attention Maps": attention_maps})

        def _run_epoch(
            self, loader: DataLoader, is_training: bool, epoch_num: int = 0
        ) -> float:
            if is_training:
                self.model.train()
                desc = f"Treinando época {epoch_num}/{self.epochs}"
            else:
                self.model.eval()
                desc = "Validando"

            total_loss = 0.0
            progress_bar = tqdm(loader, desc=desc)

            # torch.no_grad para validação
            context = nullcontext() if is_training else torch.no_grad()

            with context:
                for images, captions, caption_lengths in progress_bar:
                    images = images.to(self.device)
                    captions = captions.to(self.device)

                    if is_training:
                        self.optimizer.zero_grad()

                    predictions, alphas = self.model(
                        images, captions, caption_lengths
                    )
                    targets = captions[:, 1:]

                    predictions_flat = predictions.view(
                        -1, self.model.decoder.vocab_size
                    )
                    targets_flat = targets.reshape(-1)

                    caption_loss = self.criterion(predictions_flat, targets_flat)
                    attention_loss = (
                        self.alpha_c * ((1.0 - alphas.sum(dim=2)) ** 2).mean()
                    )
                    loss = caption_loss + attention_loss

                    if is_training:
                        loss.backward()
                        if self.clip_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.clip_grad_norm
                            )
                        self.optimizer.step()

                        wandb.log({"step_train_loss": loss.item()})

                    total_loss += loss.item()

            return total_loss / len(loader)

        def train(self):
            print(f"Começando o treino em {self.device}...")

            wandb.watch(self.model, self.criterion, log="all", log_freq=100)

            for epoch in range(1, self.epochs + 1):
                train_loss = self._run_epoch(
                    self.train_loader, is_training=True, epoch_num=epoch
                )
                val_loss = self._run_epoch(self.val_loader, is_training=False)

                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                )

                print(
                    f"\nÉpoca {epoch}/{self.epochs}\n"
                    f"Loss de treino: {train_loss:.4f}, Loss de validação: {val_loss:.4f}"
                )

                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)

                if val_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = val_loss
                    self.no_improvement_counter = 0
                    torch.save(self.model.state_dict(), self.checkpoint_path)
                    print(
                        f"Novo melhor modelo salvo em {self.checkpoint_path} (val_loss: {val_loss:.4f})"
                    )
                else:
                    self.no_improvement_counter += 1
                    print(f"Sem melhora na validação por {self.no_improvement_counter} épocas")

                if self.no_improvement_counter >= self.patience:
                    print(f"Parada antecipada na época {epoch}!")
                    break
            
                if self.lr_scheduler:
                    if isinstance(
                        self.lr_scheduler,
                        torch.optim.lr_scheduler.ReduceLROnPlateau,
                    ):
                        self.lr_scheduler.step(val_loss)
                    else:
                        self.lr_scheduler.step()

                self._log_predictions(epoch)
    return (Trainer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Testes""")
    return


@app.cell
def _(Dataset):
    class FlickrTestDataset(Dataset):
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            image = item["image"].convert("RGB")
            image = self.transform(image)

            captions = [item[f"caption_{i}"] for i in range(5)]

            return image, captions
    return (FlickrTestDataset,)


@app.cell
def _(
    Bleu,
    Cider,
    DataLoader,
    EncoderDecoder,
    Meteor,
    Rouge,
    Vocabulary,
    torch,
    tqdm,
):
    def evaluate(model: EncoderDecoder, loader: DataLoader, vocab: Vocabulary, device: str):
        model.eval()

        generated = {}
        references = {}

        with torch.no_grad():
            for i, (images, caps_list) in enumerate(tqdm(loader, desc="Avaliando")):
                if images.size(0) != 1:
                    raise ValueError("Loader dever ter batch_size=1")
                
                image = images.to(device)

                generated_caption, _, _ = model.generate_caption(
                    image, vocab, max_length=50
                )

                generated[i] = [generated_caption]
                references[i] = list(caps_list[0])
    
        scorers = [
            (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(references, generated)
        
            if isinstance(method, list): # BLEU
                for sc, scs in zip(score, method):
                    final_scores[scs] = sc
            else: # METEOR, ROUGE_L, CIDEr
                final_scores[method] = score

        return final_scores
    return (evaluate,)


@app.cell
def _(
    CaptionCollate,
    DataLoader,
    Decoder,
    Encoder,
    EncoderDecoder,
    FlickrDataset,
    FlickrTestDataset,
    RegNet_Y_1_6GF_Weights,
    Trainer,
    build_vocab_from_dataset,
    evaluate,
    load_dataset,
    nn,
    torch,
    wandb,
):
    def run():
        ds = load_dataset("jxie/flickr8k")
        train_data = ds["train"]
        val_data = ds["validation"]
        test_data = ds["test"]

        vocab = build_vocab_from_dataset(train_data, min_freq=5)

        config = {
            # arquitetura
            "decoder_type": "rnn",  # "lstm", "gru", "rnn"
            "finetune_encoder": False,
            "use_gate": False,
            # hiperparâmetros
            "vocab_size": len(vocab),
            "embed_dim": 256,
            "decoder_dim": 512,
            "attention_dim": 512,
            "encoder_dim": 888,  # RegNet
            "dropout": 0.5,
            "encoder_lr": 1e-4,
            "decoder_lr": 4e-4,
            "batch_size": 32,
            "epochs": 1,
            "clip_grad_norm": 5.0,
            "alpha_c": 1.0,  # regularização da atenção
        }

        wandb.init(
            entity="theuvargas-universidade-federal-do-rio-de-janeiro",
            project="show-attend-tell",
            config=config,
        )
        config = wandb.config

        weights = RegNet_Y_1_6GF_Weights.DEFAULT
        transform = weights.transforms()

        train_dataset = FlickrDataset(
            dataset=train_data, vocab=vocab, transform=transform
        )
        val_dataset = FlickrDataset(
            dataset=val_data, vocab=vocab, transform=transform
        )
        test_dataset = FlickrTestDataset(
            dataset=test_data, transform=transform
        )

        pad_idx = vocab.word2idx["<pad>"]
        collate_fn = CaptionCollate(pad_idx=pad_idx)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
        )

        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
        )

        DEVICE = "cuda"  # if torch.cuda.is_available() else "cpu"

        encoder = Encoder(is_trainable=config.finetune_encoder)
        decoder = Decoder(
            embed_dim=config.embed_dim,
            encoder_dim=config.encoder_dim,
            decoder_dim=config.decoder_dim,
            attention_dim=config.attention_dim,
            vocab_size=config.vocab_size,
            dropout=config.dropout,
            decoder_type=config.decoder_type,
            use_gate=config.use_gate,
        )
        model = EncoderDecoder(encoder, decoder)

        optimizer = torch.optim.Adam(
            [
                {
                    "params": filter(
                        lambda p: p.requires_grad, model.encoder.parameters()
                    ),
                    "lr": config.encoder_lr,
                },
                {
                    "params": filter(
                        lambda p: p.requires_grad, model.decoder.parameters()
                    ),
                    "lr": config.decoder_lr,
                },
            ]
        )

        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=3
        )

        checkpoint_filename = f"{wandb.run.name}-best.pth"

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            device=DEVICE,
            epochs=config.epochs,
            alpha_c=config.alpha_c,
            lr_scheduler=lr_scheduler,
            clip_grad_norm=config.clip_grad_norm,
            checkpoint_path=checkpoint_filename,
        )

        try:
            trainer.train()

            print("Treinamento finalizado, começando a avaliar")
        
            model.load_state_dict(torch.load(checkpoint_filename, map_location=DEVICE))

            scores = evaluate(model, test_loader, vocab, DEVICE)

            results_table = wandb.Table(columns=["Métrica", "Score"])
        
            print("Métricas de teste:")
            for metric, score in scores.items():
                print(f"{metric}: {score:.4f}")
                results_table.add_data(metric, score)

                wandb.summary[f"test_{metric}"] = score

            wandb.log({"test_evaluation_results": results_table})
        
            best_model_artifact = wandb.Artifact(f"model-{wandb.run.name}", type="model")
            best_model_artifact.add_file(checkpoint_filename)
            wandb.log_artifact(best_model_artifact)
        finally:
            wandb.finish()
    return (run,)


@app.cell
def _(run):
    run()
    return


if __name__ == "__main__":
    app.run()
