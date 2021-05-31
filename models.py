import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from convolutional_rnn import Conv2dGRU, Conv2dLSTM
from torch.distributions import normal
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import math

from torch.nn.utils import spectral_norm

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def reshape2Frame(x):
    n, t, c, w, h = x.size(0), x.size(1), x.size(2), x.size(3), x.size(4)
    return x.contiguous().view(t * n, c, w, h)

def reshape2Video(x, n, t):
    _, c, w, h = x.size(0), x.size(1), x.size(2), x.size(3)
    return x.contiguous().view(n, t, c, w, h)

def upsampleVideo(h, n, t):
    h = reshape2Frame(h)
    h = F.interpolate(h, scale_factor=2, mode='nearest')
    return reshape2Video(h, n, t)

def downsampleVideo(h, n, t):
    h = reshape2Frame(h)
    h = F.interpolate(h, scale_factor=1/2, mode='nearest')
    return reshape2Video(h, n, t)

class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.contiguous().view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr

class SequenceWise2d(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise2d, self).__init__()
        self.module = module

    def forward(self, x):
        n, t, c, w, h = x.size(0), x.size(1), x.size(2), x.size(3), x.size(4)
        x = x.contiguous().view(t * n, c, w, h)
        x = self.module(x)
        _, c, w, h = x.size(0), x.size(1), x.size(2), x.size(3)
        x = x.view(n, t, c, w, h)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr

class SPEECHENCODER(nn.Module):
    def __init__(self, args, bnorm=True):
        super(SPEECHENCODER, self).__init__() 

        self.args = args
        filters = [(64, 63, 4), (128, 31, 4), (256, 17, 2), (512, 9, 2)]

        drp_rate = 0#0.5

        prev_filters = 1
        layers = []
        for i, filt in enumerate(filters):
            layers+=[
                    nn.ReflectionPad1d((filt[1]-1)//2),
                    nn.Conv1d(prev_filters, filt[0], filt[1], stride=filt[2], padding=0, dilation=1)
                ]
            # if i != 0:
            #     layers+=[nn.BatchNorm1d(filt[0])]
            layers+=[
                    nn.LeakyReLU(0.2),
                    nn.Dropout(drp_rate)
                ]
            prev_filters = filt[0]
        layers+=[
                    nn.Conv1d(prev_filters, 16, 1, stride=1, padding=0, dilation=1),
                    nn.LeakyReLU(0.2)
                ]

        self.net = nn.Sequential(*layers)

        self.fc_1 = nn.Sequential(
            nn.Linear(560, self.args.speech_dim),
            nn.LeakyReLU(0.2)
        )

    def addContext(self, in_tensor):
        zeros = torch.zeros(in_tensor.size(0), self.args.context, in_tensor.size(2)).float().to(self.args.device)
        z_start = Variable( torch.Tensor(zeros.shape).normal_(0, 0.002)).float().to(self.args.device)
        z_end = Variable( torch.Tensor(zeros.shape).normal_(0, 0.002)).float().to(self.args.device)
        expanded = torch.cat( (z_start, in_tensor, z_end), 1)
        result = expanded[:, :in_tensor.size(1), :]
        for i in range(1, self.args.context*2 + 1):
            result = torch.cat( (result, expanded[:, i:i+in_tensor.size(1), :]), 2)
        return result[:, ::5, :]

    def forward(self, x):
        features = h = self.net(x)
        h = h.transpose(1, 2)
        h = self.addContext(h)
        h = SequenceWise(self.fc_1)(h)
        # print("h", h.shape)
        # print("features", features.shape)
        return h, features

class IMGENCODER(nn.Module):
    def __init__(self, args, in_dim=3):
        super(IMGENCODER, self).__init__()
        self.args= args
        self.drp_rate = 0#0.5

        prev_filters = in_dim
        for i in range(len(self.args.filters)):
            setattr(self, 
                'conv_'+str(i+1), 
                nn.Sequential(
                    nn.ReflectionPad2d((3-1)//2),
                    nn.Conv2d(prev_filters, self.args.filters[i], 3, 2),
                    # nn.BatchNorm2d(params['FILTERS'][i]),
                    nn.LeakyReLU(0.2),
                    nn.Dropout2d(self.drp_rate),
                )
            )
            prev_filters = 3 + self.args.filters[i]
        
        self.conv_6 = nn.Sequential(
            nn.Conv2d(3 + self.args.filters[-1], self.args.img_dim, kernel_size=4, stride=1),
            nn.LeakyReLU(0.2)
        )
       
    def forward(self, image): 
        feature_list = []
        x = image
        h = image
        for i in range(len(self.args.filters)): 
            h = getattr(self, 'conv_'+str(i+1))(h)
            x = F.interpolate(x, scale_factor=1/2, mode='nearest')
            h = torch.cat((h, x), 1)
            feature_list.append(h) 
        h = self.conv_6(h) 
        return h, feature_list

class NOISEGENERATOR(nn.Module):
    def __init__(self, args, debug=False):
        super(NOISEGENERATOR, self).__init__()
        self.args = args
        self.noise = normal.Normal(0, 1)
        self.noise_rnn = nn.LSTM(self.args.noise_dim, self.args.noise_dim, 1, batch_first=True)

    def forward(self, z_spch):

        self.noise_rnn.flatten_parameters()

        noise = []
        for i in range(z_spch.size(1)):
            noise.append(self.noise.sample((z_spch.size(0), self.args.noise_dim)))
        noise = torch.stack(noise, 1).to(self.args.device)

        noise, _ = self.noise_rnn(noise)

        return noise

class EMOTIONPROCESSOR(nn.Module):
    def __init__(self, args, debug=False):
        super(EMOTIONPROCESSOR, self).__init__()
        self.args = args
        
        self.fc_1 = nn.Sequential(
            nn.Linear(6, self.args.emo_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.args.emo_dim, self.args.emo_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, emotion_cond):
        emotion_cond = self.fc_1(emotion_cond)
        return emotion_cond

class DECODER(nn.Module):
    def __init__(self, args, debug=False):
        super(DECODER, self).__init__()
        self.debug = debug
        self.args = args
        self.drp_rate = 0#0.5
        
        self.fc_1 = nn.Sequential(
            nn.Linear(self.args.speech_dim+self.args.img_dim+self.args.noise_dim+self.args.emo_dim, self.args.filters[-1]*4*4),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.drp_rate),
        )

        for i in range(len(self.args.filters)-1):
            if i == 3:
                setattr(self, 
                            'conv_'+str(i+1), 
                            Conv2dLSTM(3 + 2*self.args.filters[-i-1], self.args.filters[-i-2], 3, 1, 
                            bias=True, 
                            batch_first=True, 
                            bidirectional=False, 
                            stride=1, 
                            dilation=1, 
                            groups=1)
                    )
            else:
                setattr(self, 
                        'conv_'+str(i+1), 
                        nn.Sequential(
                            nn.ReflectionPad2d((3-1)//2),
                            nn.Conv2d(3 + 2*self.args.filters[-i-1], self.args.filters[-i-2], 3, 1),
                            # nn.BatchNorm2d(params['FILTERS'][-i-2]),
                            nn.LeakyReLU(0.2),
                            nn.Dropout2d(self.drp_rate),
                        )
                )
            setattr(self, 'drp_'+str(i+1), nn.Dropout2d(self.drp_rate))

        self.conv_5 = nn.Sequential(
            nn.ReflectionPad2d((3-1)//2),
            nn.Conv2d(3 + 2*self.args.filters[0], 32, 3, 1),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )

        setattr(self, 'drp_5', nn.Dropout2d(self.drp_rate))

        self.out_new = nn.Sequential(
            nn.ReflectionPad2d(7//2),
            nn.Conv2d(32, 3, 7, 1),
            nn.Tanh()
        )


    def forward(self, img_cond, z, skip_list):
        img_cond = img_cond.unsqueeze(1).repeat(1, z.size(1), 1, 1, 1)

        h = SequenceWise(self.fc_1)(z)
        h = h.view(h.size(0), h.size(1), self.args.filters[-1], 4, 4)

        for i in range(len(self.args.filters)):
            img_f = skip_list[-1-i].unsqueeze(1).repeat(1, z.size(1), 1, 1, 1)
            h = torch.cat((h, img_f), 2) 
            # print(h.shape)
            if i == 3:
                h, _ = getattr(self, 'conv_'+str(i+1))(h)
            else:
                h = SequenceWise2d(getattr(self, 'conv_'+str(i+1)))(h)
            h = SequenceWise2d(getattr(self, 'drp_'+str(i+1)))(h)
            h = upsampleVideo(h, z.size(0), z.size(1))
        # exit()
        out = SequenceWise2d(self.out_new)(h)

        if self.debug:
            return out
        else:
            return out

class GENERATOR(nn.Module):
    def __init__(self, args, train=True, debug=False):
        super(GENERATOR, self).__init__()
        self.args = args
        self.debug = debug

        # self.drp_rate = 0.5

        self.speech_encoder = SPEECHENCODER(args)
        self.image_encoder = IMGENCODER(args, 3)
        self.noise_generator = NOISEGENERATOR(args)
        self.emotion_processor = EMOTIONPROCESSOR(args)

        self.speech_rnn = nn.LSTM(self.args.speech_dim, self.args.speech_dim, 2, batch_first=True)

        self.emotion_rnn = nn.LSTM(self.args.speech_dim, self.args.emo_dim, 2, batch_first=True)
        self.emo_classifier = nn.Sequential(
            nn.Linear(self.args.emo_dim, self.args.emo_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.args.emo_dim, 6),
        )
    
        self.decoder = DECODER(args)

        if train:
            # Optimizer
            self.opt = optim.Adam(list(self.parameters()), lr = self.args.lr_g, betas=(0.5, 0.999))
            # self.opt = optim.RMSprop(list(self.parameters()), lr = params['LR_G'])
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, self.args.steplr, gamma=0.1, last_epoch=-1)

    def forward(self, img_cond, speech_cond, emotion_cond):
        self.speech_rnn.flatten_parameters()

        z_spch, z_features = self.speech_encoder(speech_cond)
        z_spch, _ = self.speech_rnn(z_spch)

        emo_h, _ = self.emotion_rnn(z_spch)
        emo_label = self.emo_classifier(emo_h[:, -1, :])
        z_emo = self.emotion_processor(emotion_cond)
        z_emo = z_emo.unsqueeze(1).repeat(1, z_spch.size(1), 1)

        z_img, skip_list = self.image_encoder(img_cond)

        z_noise = self.noise_generator(z_spch)

        z_img = z_img.unsqueeze(1).squeeze(-1).squeeze(-1).repeat(1, z_spch.size(1), 1)
        z = torch.cat( (z_spch, z_noise, z_img, z_emo), 2)

        out = self.decoder(img_cond, z, skip_list)

        if self.debug:
            return out, z_features
        else:
            return out, z_features, emo_label

class DISCPAIRED(nn.Module):
    def __init__(self, args, debug=False):
        super(DISCPAIRED, self).__init__()
        self.args = args
        self.drp_rate = 0
        # self.emotion_processor = EMOTIONPROCESSOR(params)
        self.speech_encoder = SPEECHENCODER(args)   

        prev_filters = 6
        for i in range(len(self.args.filters)):
            setattr(self, 
                'conv_'+str(i+1), 
                nn.Sequential(
                    nn.Conv2d(prev_filters, self.args.filters[i], 3, 2),
                    nn.LeakyReLU(0.2),
                    nn.Dropout2d(self.drp_rate),
                )
            )
            prev_filters = self.args.filters[i]
        
        self.video_fc = nn.Sequential(
            nn.Linear(512*3*3, 512), 
            nn.LeakyReLU(0.2)
        )
        
        self.rnn_1 = nn.LSTM(self.args.img_dim, 256, 1, bidirectional=True, batch_first=True)
        self.rnn_2 = nn.LSTM(self.args.speech_dim, self.args.speech_dim//2, 1, bidirectional=True, batch_first=True)
        self.rnn_3 = nn.LSTM(self.args.img_dim+self.args.speech_dim, 512, 1, bidirectional=True, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.drp_rate),
            nn.Linear(1024, 1)
        )

        # Optimizer
        # self.opt = optim.Adam(list(self.parameters()), lr = params['LR_DP'], betas=(0.5, 0.999))
        self.opt = optim.RMSprop(list(self.parameters()), lr = self.args.lr_pair)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, self.args.steplr, gamma=0.1, last_epoch=-1)


    def forward(self, video, condition, speech):
        # features_for_plot = []
        
        z_spch, speech_features = self.speech_encoder(speech)
        z_spch, _ = self.rnn_2(z_spch)
       
        z_video = video 
        # print(z_video.shape)
        # features_for_plot.append(255*(0.5 + (z_video[0, :, [2, 1, 0], :, :]/2.0)))

        z_video = torch.cat((z_video, condition.unsqueeze(1).expand(-1, video.size(1), -1, -1, -1)), 2)

        for i in range(len(self.filters)):  
            z_video = SequenceWise2d(getattr(self, 'conv_'+str(i+1)))(z_video)
          
        z_video = z_video.view(z_video.size(0), z_video.size(1), -1)
        z_video = SequenceWise(self.video_fc)(z_video)        
        z_video, _ = self.rnn_1(z_video)
        
        z = torch.cat((z_spch, z_video), 2)

        z, _ = self.rnn_3(z)
       
        z = SequenceWise( self.classifier )(z)

        # features_for_plot.append(z[0, :, :].unsqueeze(1).unsqueeze(1))
        
        return z, features_for_plot, speech_features

class DISCFRAME(nn.Module):
    def __init__(self, args):
        super(DISCFRAME, self).__init__()

        self.args = args

        self.filters = [(64, 3, 2), (128, 3, 2), (256, 3, 2), (512, 3, 2), (512, 3, 2)]

        prev_filters = 6
        for i, (num_filters, filter_size, stride) in enumerate(self.filters):
            setattr(self, 
                    'conv_'+str(i+1), 
                    nn.Sequential(
                    nn.Conv2d(prev_filters, num_filters, kernel_size=filter_size, stride=stride, padding=filter_size//2),
                    nn.LeakyReLU(0.3)
                )
            )
            prev_filters = num_filters

        self.out = nn.Sequential(
            nn.Linear(8192, 2048),
            nn.LeakyReLU(0.3),
            nn.Linear(2048, 1)
        )

        # Optimizer
        self.opt = optim.Adam(list(self.parameters()), lr = self.args.lr_frame, betas=(0.5, 0.999))
        # self.opt = optim.RMSprop(list(self.parameters()), lr = params['LR_DF'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, self.args.steplr, gamma=0.1, last_epoch=-1)
       
    def forward(self, cond_img, image):
        cond_img = cond_img.unsqueeze(1).repeat(1, image.size(1), 1, 1, 1)
        h = torch.cat((image, cond_img), 2)
    
        h = h.contiguous().view(h.size(0) * h.size(1), h.size(2), h.size(3), h.size(4))
        
        for i in range(len(self.filters)):
            h = getattr(self, 'conv_'+str(i+1))(h)
    
        h = h.view(h.size(0), -1)
        h = self.out(h)
    
        return h

    def compute_grad_penalty(self, video_gt, video_pd, image_c):
        batch_size = video_gt.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, 1).expand_as(video_gt).to(self.args.device)

        interpolated = alpha * video_gt.data + (1-alpha) * video_pd.data
        interpolated = Variable(interpolated, requires_grad=True)
        
        d_out = self.forward(image_c, interpolated)

        grad_dout = torch.autograd.grad(
            outputs= d_out, 
            inputs= interpolated,
            grad_outputs= torch.ones(d_out.size()).to(self.args.device),
            create_graph=True, 
            retain_graph=True,
        )[0]
        grad_dout = grad_dout.contiguous().view(grad_dout.size(0), -1)
        gradients_norm = torch.sqrt(torch.sum(grad_dout ** 2, dim=1) + 1e-12)
        return ((gradients_norm - 1) ** 2).mean(), gradients_norm.mean()

class DISCVIDEO(nn.Module):
    def __init__(self, args, debug=False):
        super(DISCVIDEO, self).__init__()
        self.args = args
        drp_rate = 0

        self.filters = [64, 128, 256, 512, 512]

        prev_filters = 3
        for i in range(len(self.filters)):
            if i == 0:
                setattr(self, 
                        'conv_'+str(i+1), 
                        nn.Sequential(
                            nn.ReflectionPad2d((3-1)//2),
                            nn.Conv2d(prev_filters, self.filters[i], 3, 2),
                            nn.LeakyReLU(0.2),
                        )
                )
            else:
                setattr(self, 
                        'conv_'+str(i+1), 
                        nn.Sequential(
                            nn.ReflectionPad2d((3-1)//2),
                            nn.Conv2d(prev_filters, self.filters[i], 3, 2),
                            nn.LeakyReLU(0.2),
                        )
                )
            prev_filters = self.filters[i]
        
        self.video_fc = nn.Sequential(
            nn.Linear(512*4*4, 512), 
            nn.LeakyReLU(0.2)
        )
        
        self.rnn_1 = nn.LSTM(self.args.img_dim, 512, 2, bidirectional=False, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1)
        )

        self.emo_classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 6),
        )

        # Optimizer
        self.opt = optim.RMSprop(list(self.parameters()), lr = self.args.lr_video)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, self.args.steplr, gamma=0.1, last_epoch=-1)

        self.emo_loss = nn.CrossEntropyLoss()

    def forward(self, condition, video):
        features_for_plot = []    
        z_video = video   
        for i in range(len(self.filters)): 
            z_video = SequenceWise2d(getattr(self, 'conv_'+str(i+1)))(z_video)
          
        z_video = z_video.view(z_video.size(0), z_video.size(1), -1)
        z_video = SequenceWise(self.video_fc)(z_video)

        z, _ = self.rnn_1(z_video)
        
        disc_out = self.classifier(z[:, -1, :])

        emo_labels = self.emo_classifier(z[:, -1, :])

        return disc_out, emo_labels

    def enableGrad(self, requires_grad):
        for p in self.parameters():
            p.requires_grad_(requires_grad)

    def optStep(self, image_c, video, pd_video, emotion):
        self.opt.zero_grad()

        emotion = torch.argmax(emotion, dim=1)

        D_fake, emo_f = self.forward(image_c, pd_video)
       
    
        D_fake_loss = 0.5*F.mse_loss(D_fake, Variable( torch.Tensor(D_fake.shape).uniform_(0, 0.3) ).to(self.args.device))
        D_real, emo_r = self.forward(image_c, video)
        emo_loss_r = self.emo_loss(emo_r, emotion)
        D_real_loss = 0.5*F.mse_loss(D_real, Variable( torch.Tensor(D_real.shape).uniform_(0.7, 1.) ).to(self.args.device))

        D_loss = self.args.disc_video*(D_real_loss + D_fake_loss + 0.1*(emo_loss_r)) 

        D_loss.backward()
        self.opt.step()

        return D_fake_loss.item(), D_real_loss.item(), 0.5*(D_fake_loss.item()+D_real_loss.item()), emo_loss_r.item()

class DISCEMO(nn.Module):
    def __init__(self, args, debug=False):
        super(DISCEMO, self).__init__()
        self.args = args
        self.drp_rate = 0

        self.filters = [(64, 3, 2), (128, 3, 2), (256, 3, 2), (512, 3, 2), (512, 3, 2)]

        prev_filters = 3
        for i, (num_filters, filter_size, stride) in enumerate(self.filters):
            setattr(self, 
                    'conv_'+str(i+1), 
                    nn.Sequential(
                    nn.Conv2d(prev_filters, num_filters, kernel_size=filter_size, stride=stride, padding=filter_size//2),
                    nn.LeakyReLU(0.3)
                )
            )
            prev_filters = num_filters

        self.projector = nn.Sequential(
            nn.Linear(8192, 2048),
            nn.LeakyReLU(0.3),
            nn.Linear(2048, 512)
        )

        self.rnn_1 = nn.LSTM(512, 512, 1, bidirectional=False, batch_first=True)
        
        self.cls = nn.Sequential(
            nn.Linear(512, 6+1)
        )

        # Optimizer
        self.opt = optim.Adam(list(self.parameters()), lr = self.args.lr_emo, betas=(0.5, 0.999))
        # self.opt = optim.RMSprop(list(self.parameters()), lr = params['LR_DE'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, self.args.steplr, gamma=0.1, last_epoch=-1)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma=0.95, last_epoch=-1)

    def forward(self, condition, video):
        x = video
        n, t, c, w, h = x.size(0), x.size(1), x.size(2), x.size(3), x.size(4)
        x = x.contiguous().view(t * n, c, w, h)
        for i in range(len(self.filters)):
            x = getattr(self, 'conv_'+str(i+1))(x)
        h = x.view(n, t, -1)
        h = self.projector(h)
    
        h, _ = self.rnn_1(h)

        h_class = self.cls(h[:, -1, :])
    
        return h_class

    def enableGrad(self, requires_grad):
        for p in self.parameters():
            p.requires_grad_(requires_grad)

    def compute_grad_penalty(self, video_gt, video_pd, image_c, classes):
        interpolated = video_gt.data #+ (1-alpha) * video_pd.data
        interpolated = Variable(interpolated, requires_grad=True)
        
        d_out_c = self.forward(image_c, interpolated)
        classes = torch.cat((classes, torch.zeros(classes.size(0), 1).to(self.args.device)), 1)
        
        grad_dout = torch.autograd.grad(
            outputs= d_out_c, 
            inputs= interpolated,
            grad_outputs= classes.to(self.args.device),
            create_graph=True, 
            retain_graph=True,
        )[0]
        grad_dout = grad_dout.contiguous().view(grad_dout.size(0), -1)
        gradients_norm = torch.sqrt(torch.sum(grad_dout ** 2, dim=1) + 1e-12)
        return ((gradients_norm - 1) ** 2).mean(), gradients_norm.mean()