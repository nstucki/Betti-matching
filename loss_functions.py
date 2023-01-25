import torch
from torch.nn.modules.loss import _Loss

from monai.losses.dice import DiceLoss

from BettiMatching import *

def compute_BettiMatchingLoss(t, sigmoid=False, relative=False, comparison='union', filtration='superlevel', construction='V'):
    if sigmoid:
        pred = torch.sigmoid(t[0])
    else:
        pred = t[0]
    if filtration != 'bothlevel':
        BM = BettiMatching(pred, t[1], relative=relative, comparison=comparison, filtration=filtration, construction=construction, training=True)
        loss = BM.loss()
    else:
        BM = BettiMatching(pred, t[1], relative=relative, comparison=comparison, filtration='superlevel', construction=construction, training=True)
        loss = BM.loss()
        BM = BettiMatching(pred, t[1], relative=relative, comparison=comparison, filtration='sublevel', construction=construction, training=True)
        loss += BM.loss()
    return loss


def compute_WassersteinLoss(t, sigmoid=False, relative=False, filtration='superlevel', construction='V', dimensions=[0,1]):
    if sigmoid:
        pred = torch.sigmoid(t[0])
    WM = WassersteinMatching(pred, t[1], relative=relative, filtration=filtration, construction=construction, training=True)
    loss = WM.loss(dimensions=dimensions)
    return loss


def compute_ComposedWassersteinLoss(t, sigmoid=False, relative=False, filtration='superlevel', construction='V', comparison='union', dimensions=[0,1]):
    if sigmoid:
        pred = torch.sigmoid(t[0])
    WM = ComposedWassersteinMatching(pred, t[1], relative=relative, filtration=filtration, construction=construction, comparison=comparison, training=True)
    loss = WM.loss(dimensions=dimensions)
    return loss



class BettiMatchingLoss(_Loss):
    def __init__(
        self,
        batch: bool = False,
        relative=False,
        filtration='superlevel',
    ) -> None:
        super().__init__()
        self.batch = batch
        self.relative = relative
        self.filtration = filtration

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        for pair in zip(input,target):
            losses.append(compute_BettiMatchingLoss(pair, sigmoid=True, filtration=self.filtration, relative=self.relative))
        dic = {}
        dic['dice'] = DiceLoss(sigmoid=True)(input,target)
        dic['Betti matching'] = torch.mean(torch.stack(losses))
        loss = dic['Betti matching']
        return loss, dic


class DiceBettiMatchingLoss(_Loss):
    def __init__(
        self,
        batch: bool = False,
        alpha: float = 0.5,
        relative=False,
        filtration='superlevel',
    ) -> None:
        super().__init__()
        self.batch = batch
        self.alpha = alpha
        self.relative = relative
        self.filtration = filtration

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        for pair in zip(input,target):
            losses.append(compute_BettiMatchingLoss(pair, sigmoid=True, filtration=self.filtration, relative=self.relative))
        dic = {}
        dic['dice'] = DiceLoss(sigmoid=True)(input,target)
        dic['Betti matching'] = self.alpha*torch.mean(torch.stack(losses))
        loss = dic['dice'] + dic['Betti matching']
        return loss, dic



class DiceWassersteinLoss(_Loss):
    def __init__(
        self,
        batch: bool = False,
        alpha: float = 0.5,
        relative=False,
        filtration='superlevel',
        dimensions=[0,1],
    ) -> None:
        super().__init__()
        self.batch = batch
        self.alpha = alpha
        self.dimensions = dimensions
        self.relative = relative
        self.filtration = filtration

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        for pair in zip(input,target):
            losses.append(compute_WassersteinLoss(pair, sigmoid=True, filtration=self.filtration, relative=self.relative, dimensions=self.dimensions))
        dic = {}
        dic['dice'] = DiceLoss(sigmoid=True)(input,target)
        dic['Wasserstein'] = self.alpha*torch.mean(torch.stack(losses))
        loss = dic['dice'] + dic['Wasserstein']
        return loss, dic   



class DiceComposedWassersteinLoss(_Loss):
    def __init__(
        self,
        batch: bool = False,
        alpha: float = 0.5,
        relative=False,
        filtration='superlevel',
        comparison='union',
        dimensions=[0,1],
    ) -> None:
        super().__init__()
        self.batch = batch
        self.alpha = alpha
        self.dimensions = dimensions
        self.relative = relative
        self.filtration = filtration
        self.comparison = comparison

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        for pair in zip(input,target):
            losses.append(compute_ComposedWassersteinLoss(pair, sigmoid=True, filtration=self.filtration, relative=self.relative, comparison=self.comparison, dimensions=self.dimensions))
        dic = {}
        dic['dice'] = DiceLoss(sigmoid=True)(input,target)
        dic['Composed Wasserstein'] = self.alpha*torch.mean(torch.stack(losses))
        loss = dic['dice'] + dic['Comoposed Wasserstein']
        return loss, dic  