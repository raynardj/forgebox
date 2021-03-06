# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/bert_visualize.ipynb (unless otherwise specified).

__all__ = ['MLMVisualizer', 'li', 'infer_logits', 'predict_text', 'visualize', 'visualize_result', 'softmax']

# Cell
from .imports import *
from .config import Config
from .static_file import open_static
from jinja2 import Template
from .html import DOM
from uuid import uuid4

# Cell
class MLMVisualizer:
    def __init__(self,model,tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls,
                        tag:"str, like how you use from_pretrained from transformers"
                       ):
        obj = cls(
                model = AutoModelForMaskedLM.from_pretrained(tag),
                tokenizer = AutoTokenizer.from_pretrained(tag,use_fast=True),
        )
        return obj

    def tok(self,text:str,)->[
            torch.FloatTensor,
            torch.BoolTensor,
            list,
        ]:
        """
        A specific way of tokenizing.
            with pytorch tensor as input
            with mask tensor specifying where's the [MASK] token
            with offset mapping marking the positions
                in format of list in list
        """
        tokenized = self.tokenizer(
            text,
            return_tensors = "pt",
            return_offsets_mapping=True
        )
        x = tokenized['input_ids']
        offset_mapping = tokenized['offset_mapping']
        mask = x==self.tokenizer.mask_token_id
        if len(offset_mapping.shape)==3:
            offset_mapping=offset_mapping[0]
        return x,mask,offset_mapping

# Cell
softmax = nn.Softmax(dim=-1)

def li(x,)->np.array:
    if torch.is_tensor(x):
        x=x.cpu().numpy()
    return x.tolist()

def infer_logits(
        vis,
        y_pred,
        mask) -> Config:
    logits = softmax(y_pred[mask])
    pred_idx = logits.argmax(-1)
    return Config(
        logits=logits,
        pred_idx=pred_idx,
        pred_tokens = vis.tokenizer.convert_ids_to_tokens(pred_idx)
    )


MLMVisualizer.infer_logits = infer_logits

def predict_text(
        vis,
        text,
           )->Config:
    with torch.no_grad():
        x,mask,mapper=vis.tok(text)
        y_pred,attention = vis.model(x,output_attentions=True)
        infered = vis.infer_logits(y_pred,mask)
    return Config(
        text = text,
        x = li(x),
        mask = li(mask),
        mapper = li(mapper),
#         y_pred = li(y_pred),
#         logits = li(infered.logits),
        pred_idx=li(infered.pred_idx),
        pred_tokens =infered.pred_tokens,
        attention = list(map(li,attention)),
    )
MLMVisualizer.predict_text = predict_text

def visualize(vis,
              text):
    result = vis.predict_text(text)
    vis.visualize_result(result)


def visualize_result(vis, result: Config):
    template = Template(open_static('mlm/visual.html'))
    js = open_static('mlm/visual.js')
    text = result.text
    delattr(result, 'text')
    output_id = str(uuid4())
    page = template.render(data=json.dumps(result),
                           text=text,
                           output_id=output_id,
                           mlm_visual_js=js)
    DOM(page, "div",)()


MLMVisualizer.visualize = visualize
MLMVisualizer.visualize_result = visualize_result