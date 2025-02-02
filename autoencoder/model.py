import torch.nn as nn
import torch

# Define my 1-D UNet
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU()
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p   # return pre-pool and after-pool


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(2 * out_c, out_c)
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class unet(nn.Module):
    def __init__(self, channels=[1, 32, 64, 128], num_classes = 337, classifier_channels = 1, vec_len=6016):
        super().__init__()
        """ Encoder """
        self.encoder_blocks = nn.ModuleList([encoder_block(channels[i], channels[i+1]) for i in range(len(channels)-2)])
        """ Bottleneck """
        self.b = conv_block(channels[-2] , channels[-1])

        """ Decoder """
        self.decoder_blocks = nn.ModuleList([decoder_block(channels[-i], channels[-i-1]) for i in range(1, len(channels)-1)])

        """ Reconstruction """
        self.reconstruct = nn.Conv1d(channels[1], channels[0], kernel_size=1, padding=0)

        """ Classifier """
        in_dim = int(vec_len * classifier_channels / (2 ** (len(channels) - 2)))
        # print(in_dim)
        self.pre_classifier = nn.Sequential(nn.Conv1d(channels[-1], classifier_channels, kernel_size=1, padding=0), nn.Flatten())
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, inputs):
        """ Encoder """
        s_list,p_list = [], [inputs]
        for i in range(len(self.encoder_blocks)):
            s, p = self.encoder_blocks[i](p_list[-1])
            s_list.append(s)
            p_list.append(p)
            
        """ Bottleneck """
        b = self.b(p_list[-1])   # bsz, 

        """ Decoder """
        d_list = [b]
        # print(b.shape, s.shape)

        for i in range(len(self.decoder_blocks)):
            d = self.decoder_blocks[i](d_list[-1], s_list[-(i+1)])
            d_list.append(d)

        """ Reconstruction """
        outputs = self.reconstruct(d_list[-1])

        """ Classifier """
        h0 = self.pre_classifier(b)  # save as low-d features
        h = self.classifier(h0)

        return outputs, h, h0, b


class unet_all_in_one(nn.Module):
    def __init__(self, channels=[1, 32, 64, 128], num_species = 42, num_classes = 870, classifier_channels = 1, vec_len=6016, subsp_hid=128):
        super().__init__()
        """ Encoder """
        self.encoder_blocks = nn.ModuleList([encoder_block(channels[i], channels[i+1]) for i in range(len(channels)-2)])
        
        """ Bottleneck """
        self.b = conv_block(channels[-2] , channels[-1])

        """ Decoder """
        self.decoder_blocks = nn.ModuleList([decoder_block(channels[-i], channels[-i-1]) for i in range(1, len(channels)-1)])

        """ Reconstruction """
        self.reconstruct = nn.Conv1d(channels[1], channels[0], kernel_size=1, padding=0)

        """ Classifiers """
        in_dim = int(vec_len * classifier_channels / (2 ** (len(channels) - 2)))
        # print(in_dim)
        self.pre_classifier = nn.Sequential(nn.Conv1d(channels[-1], classifier_channels, kernel_size=1, padding=0), nn.Flatten())
        self.classifier_species = nn.Linear(in_dim, num_species)
        self.classifier_subspecies = nn.Sequential(nn.Linear(in_dim + num_species, subsp_hid), nn.ReLU(), nn.Linear(subsp_hid, num_classes))

    def forward(self, inputs):
        """ Encoder """
        s_list,p_list = [], [inputs]
        for i in range(len(self.encoder_blocks)):
            s, p = self.encoder_blocks[i](p_list[-1])
            s_list.append(s)
            p_list.append(p)
            
        """ Bottleneck """
        b = self.b(p_list[-1])   # bsz, 

        """ Decoder """
        d_list = [b]
        # print(b.shape, s.shape)

        for i in range(len(self.decoder_blocks)):
            d = self.decoder_blocks[i](d_list[-1], s_list[-(i+1)])
            d_list.append(d)

        """ Reconstruction """
        outputs = self.reconstruct(d_list[-1])

        """ Classifier """
        h0 = self.pre_classifier(b)  # save as low-d features
        h_species = self.classifier_species(h0)
        h = self.classifier_subspecies(torch.cat([h0, h_species], axis=1))

        return outputs, h, h_species, h0, b


# unte that aligns with SNP distance
class unet_snp(nn.Module):
    def __init__(self, channels=[1, 32, 64, 128], num_species = 42, classifier_channels = 1, vec_len=6016, snp_dim=512):
        super().__init__()
        """ Encoder """
        self.encoder_blocks = nn.ModuleList([encoder_block(channels[i], channels[i+1]) for i in range(len(channels)-2)])
        """ Bottleneck """
        self.b = conv_block(channels[-2] , channels[-1])

        """ Decoder """
        self.decoder_blocks = nn.ModuleList([decoder_block(channels[-i], channels[-i-1]) for i in range(1, len(channels)-1)])

        """ Reconstruction """
        self.reconstruct = nn.Conv1d(channels[1], channels[0], kernel_size=1, padding=0)

        """ Further Encoding """
        in_dim = int(vec_len * classifier_channels / (2 ** (len(channels) - 2)))
        # print(in_dim)
        self.pre_classifier = nn.Sequential(nn.Conv1d(channels[-1], classifier_channels, kernel_size=1, padding=0), nn.Flatten(),
                                            nn.Linear(in_dim, snp_dim))

        """ SNP Projection"""
        # self.SNP_projection = nn.Linear(in_dim, snp_dim)S
        # self.SNP_projection = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, snp_dim))

        """ Classifiers """
        # self.classifier_species = nn.Linear(in_dim, num_species)
        self.classifier_species = nn.Sequential(nn.Linear(snp_dim, snp_dim), nn.ReLU(), nn.Linear(snp_dim, num_species))

    def forward(self, inputs):
        """ Encoder """
        s_list,p_list = [], [inputs]
        for i in range(len(self.encoder_blocks)):
            s, p = self.encoder_blocks[i](p_list[-1])
            s_list.append(s)
            p_list.append(p)
            
        """ Bottleneck """
        b = self.b(p_list[-1])   # bsz, 

        """ Decoder """
        d_list = [b]
        # print(b.shape, s.shape)

        for i in range(len(self.decoder_blocks)):
            d = self.decoder_blocks[i](d_list[-1], s_list[-(i+1)])
            d_list.append(d)

        """ Reconstruction """
        outputs = self.reconstruct(d_list[-1])

        """ Further Encoding"""
        h0 = self.pre_classifier(b)  # save as low-d features

        # """ SNP Projection """
        # snp = self.SNP_projection(h0)

        """ Classifier """

        h_species = self.classifier_species(h0)

        return outputs, h_species, h0, b


# Define my baseline CNN  (only encoder and classifier, no decoder or reconstruction)
class baseline_cnn(nn.Module):
    def __init__(self, channels=[1, 32, 64, 128], num_classes = 337, classifier_channels = 1, vec_len=6016):
        super().__init__()
        """ Encoder """
        self.encoder_blocks = nn.ModuleList([encoder_block(channels[i], channels[i+1]) for i in range(len(channels)-2)])
        """ Bottleneck """
        self.b = conv_block(channels[-2] , channels[-1])
        """ Classifier """
        in_dim = int(vec_len * classifier_channels / (2 ** (len(channels) - 2)))
        # print(in_dim)
        self.pre_classifier = nn.Sequential(nn.Conv1d(channels[-1], classifier_channels, kernel_size=1, padding=0), nn.Flatten())
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, inputs):
        """ Encoder """
        s_list,p_list = [], [inputs]
        for i in range(len(self.encoder_blocks)):
            s, p = self.encoder_blocks[i](p_list[-1])
            s_list.append(s)
            p_list.append(p)
            
        """ Bottleneck """
        b = self.b(p_list[-1])   # bsz, 512, 64

        """ Classifier """
        h0 = self.pre_classifier(b)  # save as low-d features
        h = self.classifier(h0)

        return h, h0, b


# Define my baseline MLP
class mlp_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.fc = nn.Linear(in_c, out_c)
        self.bn = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU()
    def forward(self, inputs):
        x = self.fc(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x


class baseline_mlp(nn.Module):
    def __init__(self, num_classes = 337, hidden_sizes = [512, 256, 64], vec_len=6000):
        super().__init__()
        self.input_layer = mlp_block(vec_len, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList([mlp_block(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)])
        self.classifier = nn.Linear(hidden_sizes[-1], num_classes)
    def forward(self, inputs):
        h0 = self.input_layer(inputs)
        for i in range(len(self.hidden_layers)):
            h0 = self.hidden_layers[i](h0)
        h = self.classifier(h0)
        return h, h0, None