# ftorch: A pytorch API to forge

### Preprocessor
forgebox contains data pre-processing module for pytorch.

It returns pytorch dataset
```python
from forgebox.ftorch.prepro import DF_Dataset,fuse
train_ds_x = DF_Dataset(train_df, # pandas dataframe for training
                        prepro=x_prepro, # x_prepro is the function: input a chunk of dataframe, return same len of data in numpy array
                        bs=64, # batch_size
                        shuffle=False)
# train_ds_x will be a pytorch dataset
train_ds_y = DF_Dataset(train_df,prepro=y_prepro,bs=64,shuffle=False)
```
fuse combines 2 datasets together,
```python
train_ds = fuse(train_ds_x,train_ds_y)
```
we can use it like ```x,y = next(iter(train_ds))```

### Trainer
```python
from forgebox.ftorch.train import Trainer

trainer=Trainer(train_ds, val_dataset = valid_ds ,batch_size=1,print_on=2)
trainer.opt["adm1"] = torch.opt.Adam(model.parameters())
```

#### A training step
we use the step_train decorator to define a training step
```python
from forgebox.ftorch.metrics import metric4_bi

@trainer.step_train
def action(self):
    x,y = self.data
    x = x.squeeze(0)
    y = y.float()
    opt.zero_grad()
    y_ = model(x)
    loss = loss_func(y_,y)
    acc, rec, prec, f1 = metric4_bi(y_,y)
    loss.backward()
    self.opt.adm.step()
    return {"loss":loss.item(),"acc":acc.item(),"rec":rec.item(),"prec":prec.item(),"f1":f1.item()}
```
The above is only an example, you can define the training step as you like.

Even if you have 3 models, updated by 2 optimizers, as long as you can decribe them in a single training step, it can be done easily.

#### A validation step
Very similar to the train step, minus any code relate to updating weights

```python
@trainer.step_val # step_val decorator
def val_action(self)
    x,y = self.data
    x = x.squeeze(0)
    y = y.float()
    y_ = model(x)
    ...
```

Then run the training for 3 epochs like:

```python
trainer.train(3)
```