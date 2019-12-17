# Better hacking on the data science

## Installation
```
python -m pip install --force-resinstall  https://github.com/raynardj/forgebox
```

## This module should run independently from forge web UI


## Change the default db usage
* In [config.py](config.py) setting the following:
```python
# SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(DATADIR, 'forge.db')
SQLALCHEMY_DATABASE_URI = 'sqlite:////mnt/disk1/forge.db')
```

## Initiate forge api

```python
from forgebox.apicore import forgedb
fg = forgedb("nlp_binary_classification")
p = fg.p
```

## Set/ Read Hyper Params

Read the hyper param from database
```python
hs1 = p("hidden_size1")
```

Set a new hyper param, like, 3 epochs for training
```python
epochs = p("nb_epochs",3)
```

## Create data format

int and float are already there
fg.format("str",remark = "String Format")

## PyTorch Integration
For pytorch users, Forge is intergrated into the training framework. Check [ftorch](ftorch) for detail

Tutorial Example for Pytorch User
### Using Trainer form [tracker.py](tracker.py)

```python
fg = FG(task = "sentimental_analysis")
model1 = SomePyTorchModel()
model2 = OtherPyTorchModel()
trainer = Trainer(train_ds,
            val_dataset = val_ds, #pytorch dataset
            batch_size = 1,
            callbacks = [fg.weights(model1), # save the weights by each epoch end
                        fg.weights(model2),  # save another model weights
                        fg.metrics(), # save the metrics for model performance on this epoch
                        stat, # print out the statistics by the end of each training epoch
                        ],
            val_callbacks=[
                            fg.metrics(),
                            stat, # print out the statistics by the end of each training epoch
                        ],
            )
```
#### Define a training step
A training step, validation step here, is pretty much like what happens in an iteration in your usual pytorch training task.
* get the x, y (or more variables)
* clear the gradient if any
* predict the y hat from the model,(or many of the prediction variable, out of many models)
* calculate the loss function/loss functions
* back propagates from the loss/losses
* a step(update the model/models) in optimizer
* return the loss/ other print metrics for further review/selection purpose

You can see, under this schema, you have the total liberty of managing how many:
* input/target variables
* models
* optimizers
* loss functions
* losses
* metrics
you have, and they can work in a bizarre combination if you want to.

It handles iteration/ batch consumption/ logging/ printing/ saving model easily without compromising any true beauty of pytorch.

Example
```python
@trainer.step_train
def action(*args,**kwargs):
    x,y = args[0]
    x = x.squeeze(0)
    y = y.float()
    opt.zero_grad()
    y_ = model1(x)
    y_2 = model2(x)
    loss = loss_func(y_,y)+loss_func(y_2,y)
    acc = accuracy(y_,y.long())
    rec = recall(y_,y.long())
    prec = precision(y_,y.long().squeeze(0))
    f1 = (rec*prec)/(rec+prec)
    loss.backward()
    opt.step()
    return {"loss":loss.item(),"acc":acc,"rec":rec,"prec":prec,"f1":f1}
```

And the validation step is like following, almost the same with training action, without any backward calc/model updating code
```python
@trainer.step_val # step_val decorator
def val_action(*args,**kwargs)
    x,y = args[0]
    x = x.squeeze(0)
    y = y.float()
    y_ = model1(x)
    y_2 = model2(x)
    loss = loss_func(y_,y)+loss_func(y_2,y)
    acc = accuracy(y_,y.long())
    rec = recall(y_,y.long())
    prec = precision(y_,y.long().squeeze(0))
    f1 = (rec*prec)/(rec+prec)
    return {"loss":loss.item(),"acc":acc,"rec":rec,"prec":prec,"f1":f1}
```

Then train for 5 epochs

``` python
trainer.train(5)
```

### Layers

Layers are mostly pytorch modules, examples:

GELU:
```python
from forgebox.ftorch.layers import GELU

gelu = GELU()
x = gelu(x)
```

A list of layers:
* activations
    * GELU
* cv
    * Coord2d
* nlp
    * Attention
    * MultiHeadedAttention
    * TransformerBlock
* norm
    * LayerNorm
    * UnNormalize