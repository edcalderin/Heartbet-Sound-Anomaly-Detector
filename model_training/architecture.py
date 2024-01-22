import timm
from torch import nn


class PreTrainedNetwork(nn.Module):
    def __init__(self, model_name: str, num_classes: int) -> None:
        super().__init__()

        self.__model = timm.create_model(model_name, pretrained = True, in_chans = 1)

        self.in_features = self.__model.classifier.in_features

        self.__model.classifier = nn.Sequential(
            nn.Dropout(p = 0.1),
            nn.Linear(self.in_features, num_classes)
        )

    def forward(self, input):
        return self.__model(input)
