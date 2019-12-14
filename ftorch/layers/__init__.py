from .activations import *
from .nlp import *
from .cv import *
from .norm import *

class Flatten(nn.Module):
    def forward(self, input):
        """
        a pytorch version of Flatten layer
        """
        return input.view(input.size(0), -1)


class passon(nn.Module):
    def __init__(self):
        """
        forward calculation pass on the x
        and do nothing else
        """
        super(passon, self).__init__()

    def forward(self, x):
        return x


class DeepMaskAttLSTM(nn.Module):
    def __init__(self, mask_activation="softmax", extra_mask_layers=0, **kwargs):
        """
        mask_activation: Activation layer to the mask, right before applying the mask
            one of "softmax","sigmoid","passon", "passon" means no activation
        extra_mask_layers: int number, how many hidden layers does mask_maker use?
        Attentional LSTM
        input_size: input dimension
        hidden_size: hidden dimension, also the output dimention of LSTM
        other kwargs of LSTM, most of the following is  pilferage from nn.LSTM doc:
            input_size: mentioned above, only have to specify once
            hidden_size: mentioned above, only have to specify once
            num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
                would mean stacking two LSTMs together to form a `stacked LSTM`,
                with the second LSTM taking in outputs of the first LSTM and
                computing the final results. Default: 1
            bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
                Default: ``True``
            batch_first: If ``True``, then the input and output tensors are provided
                as (batch, seq, feature). Default: ``False``
            dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
                LSTM layer except the last layer, with dropout probability equal to
                :attr:`dropout`. Default: 0
            bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``
        """
        super(DeepMaskAttLSTM, self).__init__()
        self.input_size = kwargs["input_size"]
        self.hidden_size = kwargs["hidden_size"]
        self.mask_maker_layers = []
        self.size_in = [self.input_size] + [self.hidden_size] * extra_mask_layers
        for i in range(extra_mask_layers):
            self.mask_maker_layers += [
                nn.Linear(self.size_in[i], self.hidden_size),
                nn.ReLU(),
                nn.Dropout(), ]
        self.mask_maker_layers.append(nn.Linear(self.size_in[-1], 1, bias=False))
        self.mask_maker = nn.Sequential(*self.mask_maker_layers)
        self.lstm = nn.LSTM(**kwargs)
        if mask_activation == "softmax":
            self.mask_act = nn.Softmax(dim=1)
        elif mask_activation == "sigmoid":
            self.mask_act = nn.Sigmoid()
        elif mask_activation == "relu":
            self.mask_act = nn.ReLU()
        elif mask_activation == "passon":
            self.mask_act = passon()
        else:
            print("Activation type:%s not found, should be one of the following:\nsoftmax\nsigmoid\nrelu\npasson" % (
                mask_activation))

    def forward(self, x):
        mask_basic = self.mask_maker(x).squeeze(-1)
        mask = self.mask_act(mask_basic).unsqueeze(1)  # mask shape (bs,1,seq_leng)
        output, (h_n, c_n) = self.lstm(x)
        output = mask.bmm(output)
        output = output.squeeze(1)  # output shape (bs, hidden_size)


class MaskMakers(nn.Module):
    def __init__(self, hidden_size, nb_masks, act="sigmoid"):
        """
        nb_masks: number of masks
        act: str, default "sigmoid", can be one of "softmax","sigmoid","relu","tanh"
        x for forward: (bs,seq_len, hidden_size)
        each mask in shape (bs,seq_len,1)
        """
        super().__init__()
        if act == "softmax":
            act_layer = nn.Softmax(dim=1)
        elif act == "sigmoid":
            act_layer = nn.Sigmoid()
        elif act == "relu":
            act_layer = nn.ReLU()
        elif act == "tanh":
            act_layer = nn.Tanh()

        self.nb_masks = nb_masks
        list(
            setattr(self, "attn_masker_%s" % (_), nn.Sequential(*[nn.Linear(hidden_size, 1, bias=True), act_layer])) for
            _ in range(nb_masks))

    def forward(self, x):
        return tuple(getattr(self, "attn_masker_%s" % (_))(x) for _ in range(self.nb_masks))


class MultiMaskLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1, nb_masks=1):
        """
        seq_len: sequence length
        return outputs, masks:
        * outputs: a tuple of outputs, size (bs, hidden_size)
        * masks: a tuple of masks, size(bs,seq_len,1)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nb_masks = nb_masks

        self.emb = nn.Embedding(vocab_size, hidden_size, )
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=num_layers)
        self.makers = MaskMakers(hidden_size, nb_masks=nb_masks)

    def forward(self, x_input):
        embedded = self.emb(x_input)
        masks = self.makers(embedded)

        output, (h, c) = self.lstm(embedded)
        output = output.permute(0, 2, 1)

        outputs = tuple(output.bmm(mask).squeeze(-1) for mask in masks)

        return outputs, masks
