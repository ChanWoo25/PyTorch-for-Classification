# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
import torch.nn as nn


class AlexNet(nn.Module):

    def __init__(self,
                 num_classes: int,
                 init_weights: bool = False):

        super(AlexNet, self).__init__()
        self.feature = nn.Sequential(
            # [3, 224, 224]
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # [48, 55, 55]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # [48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # [128, 13, 13]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # [128, 6, 6]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # [192, 6, 6]
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # [128, 6, 6]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # [128, 2, 2]
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, num_classes),
        )

        if init_weights == True:
            self.__init_weights()

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, start_dim=1)
        # [
