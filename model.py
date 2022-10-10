import torch as th
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, base_model, classifier, dim=128, cls_size=[1000], fixed_cls=False):
        super(Model, self).__init__()
        self.backbone = base_model
        self.classifier = classifier
        self.dim = dim
        self.num_cls = len(cls_size)
        self.fixed_cls = fixed_cls
        self.cls_size = cls_size

        for cls_i in range(self.num_cls):
            cls_layer_i = nn.utils.weight_norm(nn.Linear(dim, self.cls_size[cls_i], bias=False))
            cls_layer_i.weight_g.data.fill_(1)
            setattr(self, "cls_%d" % cls_i, cls_layer_i)

            if self.fixed_cls:
                for param in getattr(self, "cls_%d" % cls_i).parameters():
                    param.requires_grad = False

    def forward(self, x, cls_num=None, return_embds=False):
        if isinstance(x, list): # multiple views
            bs_size = x[0].shape[0]

            if return_embds:
                # run backbone forward pass separately on each augmentation
                for i, v in enumerate(x):
                    _out = self.backbone(v.reshape(bs_size, 1, v.shape[-1]).float())
                    if i == 0:
                        output = _out
                    else:
                        output = th.cat((output, _out))

                # run classification head forward pass on concatenated features
                embds = self.classifier(output)
                # convert back to list of views
                embds = [embds[x: x + bs_size] for x in range(0, len(embds), bs_size)]
                return embds
            else: # input is embds
                # concatenate features
                x = th.cat(x, 0)

                # apply classifiers
                if cls_num is None:
                    # apply all classifiers
                    out = [getattr(self, "cls_%d" % cls)(x) for cls in range(self.num_cls)]
                else:
                    # apply only cls num
                    out = getattr(self, "cls_%d" % cls_num)(x)

                # convert to list of lists (classifiers and views)
                output = [[out[cls][x: x + bs_size] for x in range(0, len(out[cls]), bs_size)]
                          for cls in range(len(out))]
        else: # single view
            x = self.backbone(x)
            x = self.classifier(x)

            if return_embds:
                return x
            else:
                # apply classifiers
                if cls_num is None:
                    # apply all classifiers
                    output = [getattr(self, "cls_%d" % cls)(x) for cls in range(self.num_cls)]
                else:
                    # apply only cls_num
                    output = getattr(self, "cls_%d" % cls_num)(x)
        return output
