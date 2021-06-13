from collections import defaultdict
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import torchvision
import utils
from torch.utils.tensorboard import SummaryWriter

class perceptionLoss():
    def __init__(self, args):
        vgg = torchvision.models.vgg19(pretrained=True)
        vgg.eval()
        self.features = vgg.features.to(args.device)
        self.feature_layers = ['4', '9', '18', '27', '36']
        self.mse_loss = nn.MSELoss()

    def getfeatures(self, x):
        feature_list = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.feature_layers:
                feature_list.append(x)
        return feature_list

    def calculatePerceptionLoss(self, video_pd, video_gt):
        features_pd = self.getfeatures(video_pd.view(video_pd.size(0)*video_pd.size(1), video_pd.size(2), video_pd.size(3), video_pd.size(4)))
        features_gt = self.getfeatures(video_gt.view(video_gt.size(0)*video_gt.size(1), video_gt.size(2), video_gt.size(3), video_gt.size(4)))
        
        with torch.no_grad():
            features_gt = [x.detach() for x in features_gt]
        
        perceptual_loss = sum([self.mse_loss(features_pd[i], features_gt[i]) for i in range(len(features_gt))])
        return perceptual_loss

class tfaceTrainer:
    def __init__(self, 
                args, 
                generator,
                disc_frame,
                disc_pair,
                disc_emo,
                disc_video,
                train_loader,
                val_loader):
        
        self.args = args
        
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.generator = generator

        self.disc_frame = disc_frame
        self.disc_pair = disc_pair
        self.disc_emo = disc_emo
        self.disc_video = disc_video

        # self.plotter = utils.VisdomLinePlotter(env=args.env_name)
        self.plotter = SummaryWriter(args.out_path)

        self.l1_loss = torch.nn.L1Loss()
        self.emo_loss = nn.CrossEntropyLoss()

        self.emo_loss_disc = nn.CrossEntropyLoss()
        self.loss_dict = defaultdict(list)

        self.global_step = 0

        self.perception_loss = perceptionLoss(args)
    
    def freezeNet(self, network):
        for p in network.parameters():
            p.requires_grad = False
    
    def unfreezeNet(self, network):
        for p in network.parameters():
            p.requires_grad = True

    def schdulerStep(self):
        self.generator.module.scheduler.step()
        if self.args.disc_pair:
            self.disc_pair.module.scheduler.step()
        if self.args.disc_frame:
            self.disc_frame.module.scheduler.step()
        if self.args.disc_video:
            self.disc_video.module.scheduler.step()
        if self.args.disc_emo:
            self.disc_emo.module.scheduler.step()

    def displayLRs(self):
        lr_list = [self.generator.module.opt.param_groups]
        if self.args.disc_pair:
            lr_list.append(self.disc_pair.module.opt.param_groups)
        if self.args.disc_frame:
            lr_list.append(self.disc_frame.module.opt.param_groups)
        if self.args.disc_video:
            lr_list.append(self.disc_video.module.opt.param_groups)
        if self.args.disc_emo:
            lr_list.append(self.disc_emo.module.opt.param_groups)
        
        cnt = 0
        for lr in lr_list:
            for param_group in lr:
                print('LR {}: {}'.format(cnt, param_group['lr']))
                cnt+=1

    def saveNetworks(self, fold):
        torch.save(self.generator.state_dict(), os.path.join(self.args.out_path, fold, 'generator.pt'))
        if self.args.disc_pair:
            torch.save(self.disc_pair.state_dict(), os.path.join(self.args.out_path, fold, 'disc_pair.pt'))
        if self.args.disc_frame:
            torch.save(self.disc_frame.state_dict(), os.path.join(self.args.out_path, fold, 'disc_frame.pt'))
        if self.args.disc_video:
            torch.save(self.disc_video.state_dict(), os.path.join(self.args.out_path, fold, 'disc_video.pt'))
        if self.args.disc_emo:
            torch.save(self.disc_emo.state_dict(), os.path.join(self.args.out_path, fold, 'disc_emo.pt'))
        print('Networks has been saved to {}'.format(fold))

    def calcGANLoss(self, logit, label):
        if label == 'real':
            return -logit.mean()
        if label == 'fake':
            return logit.mean()

    def logLosses(self, t):
        desc_str=''
        for key in sorted(self.loss_dict.keys()):
            desc_str += key + ': %.5f' % (np.nanmean(self.loss_dict[key])) + ', '
        t.set_description(desc_str)

    def plotLosses(self, var_name, xlabel, ylabel, legend, title, rem=0):
        if self.global_step%self.args.plot_interval == rem:
            for key in legend:
                try:
                    self.plotter.add_scalar("Loss/train", self.loss_dict[key][-1], self.global_step)
                except:
                    continue
        
        # Visdom Plotter
        # if self.global_step%self.args.plot_interval == rem:
        #     x = []
        #     y = []
        #     for key in legend:
        #         y.append(np.nanmean(self.loss_dict[key][-5:]))
        #         x.append(self.global_step)
        #     self.plotter.plot(var_name, xlabel, ylabel, legend, title, x, y)

    def convertVid(self, V):
        return (0.5 + (V/2.0))

    def logValImages(self, epoch):
        speech_v, video_v, att_v, emotion_v = [d.float().to(self.args.device) for d in next(iter(self.val_loader))]
        self.generator.eval()
        pd_video_v, z_spch_v, emo_label_v = self.generator(video_v[:, np.random.randint(video_v.shape[1], size=1)[0], ...], speech_v, emotion_v)

        pd_video_v = pd_video_v[:, :, :, :, :]
        video_v_p = video_v[:, :, :, :, :]

        pd_video_v = pd_video_v.view(pd_video_v.size(0) * pd_video_v.size(1), pd_video_v.size(2), pd_video_v.size(3), pd_video_v.size(4))
        video_v_p = video_v_p.view(video_v_p.size(0) * video_v_p.size(1), video_v_p.size(2), video_v_p.size(3), video_v_p.size(4))

        grid = torchvision.utils.make_grid(self.convertVid( torch.cat((pd_video_v[:, :, :, :], video_v_p[:, :, :, :]), 0) ))
        self.plotter.add_image("Predicted and GT Video Frames", grid, self.global_step)
        
        # Visdom Plotter
        # self.plotter.viz.images( self.convertVid( torch.cat((pd_video_v[:, :, :, :], video_v_p[:, :, :, :]), 0) ), 
        #                         opts=dict(jpgquality=70, store_history=False, caption='e'+str(epoch)+"_check_"+str(self.global_step),title='e'+str(epoch)+"_check_"+str(self.global_step)),
        #                         env=self.args.env_name,
        #                         win='samples',
        #                         nrow=self.args.num_frames,
        #                         )

    def step_disc_frame(self, data):
        self.disc_frame.train()
        speech, video_gt, mrm, emotion, image_c, video_pd = data
        self.disc_frame.module.opt.zero_grad()

        logit_fake = self.disc_frame(image_c, video_pd)
        logit_real = self.disc_frame(image_c, video_gt)

        loss_fake = self.calcGANLoss(logit_fake, 'fake')
        loss_real = self.calcGANLoss(logit_real, 'real')

        self.loss_dict['loss_df_fake'].append(loss_fake.item())
        self.loss_dict['loss_df_real'].append(loss_real.item())

        gp, grad_norm = self.disc_frame.module.compute_grad_penalty(video_gt, video_pd, image_c)
        
        self.loss_dict['df_gp'].append(gp.item())
        self.loss_dict['df_gnorm'].append(grad_norm.item())

        loss = loss_fake + loss_real + self.args.disc_frame_gp*gp

        wdistance = -(loss_fake + loss_real).item()

        self.loss_dict['df_wdistance'].append(wdistance)

        loss.backward()
        self.disc_frame.module.opt.step()

        self.plotLosses('Disc Frame Losses', 'iterations', 'loss', ['loss_df_fake', 'loss_df_real'], 'Disc Frame Losses', rem=1)
        self.plotLosses('frame_wdistance', 'iterations', 'loss', ['df_wdistance'], 'wdistance', rem=1)
        self.plotLosses('frame_gp', 'iterations', 'loss', ['df_gp', 'df_gnorm'], 'gp', rem=1)

    def step_disc_emo(self, data):
        self.disc_emo.train()
        speech, video_gt, mrm, emotion, image_c, video_pd = data
        self.disc_emo.module.opt.zero_grad()

        class_fake = self.disc_emo(image_c, video_pd)
        class_real = self.disc_emo(image_c, video_gt)

        loss_fake_c = self.emo_loss_disc(class_fake, (6*torch.ones_like(torch.argmax(emotion, dim=1))).long().to(self.args.device))
        loss_real_c = self.emo_loss_disc(class_real, torch.argmax(emotion, dim=1))

        self.loss_dict['loss_fake_c'].append(loss_fake_c.item())
        self.loss_dict['loss_real_c'].append(loss_real_c.item())

        loss = 0.5*(loss_fake_c + loss_real_c) 

        loss.backward()
        self.disc_emo.module.opt.step()

        self.plotLosses('Disc Emotion', 'iterations', 'loss', ['loss_fake_c', 'loss_real_c'], 'disc_emo', rem=1)

    def step_disc_emo_recog(self, data):
        self.disc_emo.train()
        speech, video_gt, mrm, emotion, image_c = data
        self.disc_emo.module.opt.zero_grad()
 
        class_real = self.disc_emo(image_c, video_gt)

        loss = self.emo_loss_disc(class_real, torch.argmax(emotion, dim=1))

        self.loss_dict['loss_classifier'].append(loss.item())

        loss.backward()
        self.disc_emo.module.opt.step()

        self.plotLosses('Disc Emo Losses', 'iterations', 'loss', ['loss_classifier'], 'Disc Emo Losses')

    def step_generator(self, data):
        if self.args.disc_frame:
            self.disc_frame.eval()
            self.freezeNet(self.disc_frame)
        if self.args.disc_emo:
            # self.disc_emo.eval()
            self.freezeNet(self.disc_emo)

        self.generator.train()
        speech, video_gt, mrm, emotion, image_c = data
        self.generator.module.opt.zero_grad()

        video_pd, z_spch, emo_label = self.generator(image_c, speech, emotion)
        
        if self.args.disc_frame:
            df = self.disc_frame.forward(image_c, video_pd)
            loss_df = self.calcGANLoss(df, 'real')
        if self.args.disc_emo:
            de_c = self.disc_emo.forward(image_c, video_pd)
            loss_de_c = self.emo_loss(de_c, torch.argmax(emotion, dim=1))
            self.loss_dict['loss_de_c'].append(loss_de_c.item())

        perception_loss = self.perception_loss.calculatePerceptionLoss(video_pd, video_gt)

        recon_loss = 100*self.l1_loss(video_pd*mrm, video_gt*mrm)

        emo_loss = self.emo_loss(emo_label, torch.argmax(emotion, dim=1))

        self.loss_dict['loss_rec'].append(recon_loss.item())
        self.loss_dict['loss_emo'].append(emo_loss.item())
        self.loss_dict['perception_loss'].append(perception_loss.item())        

        loss = 0.001*emo_loss + recon_loss + perception_loss
        if self.args.disc_frame:
            loss += self.args.disc_frame * loss_df
        if self.args.disc_emo:
            loss_demo = self.args.disc_emo * loss_de_c
            self.loss_dict['loss_demo'].append(loss_demo.item())
            loss += loss_demo

        self.loss_dict['loss_gen'].append(loss.item())

        loss.backward()
        self.generator.module.opt.step()

        if self.args.disc_frame:
            self.unfreezeNet(self.disc_frame)
        if self.args.disc_emo:
            self.unfreezeNet(self.disc_emo)
            self.plotLosses('Gen Emotion', 'iterations', 'loss', ['loss_de_c'], 'gen_emo')
        self.plotLosses('Gen Losses', 'iterations', 'loss', ['loss_rec', 'loss_gen', 'perception_loss', 'loss_demo'], 'Gen Losses')
        

    def train(self):
        for epoch in tqdm(range(self.args.num_epochs)):
            diterator = iter(self.train_loader)
            # with trange(1) as t:
            with trange(len(self.train_loader)) as t:     
                for i in t:               
                    speech, video, mrm, emotion = [d.float().to(self.args.device) for d in next(diterator)]

                    mrm = mrm.unsqueeze(2)
                    mrm = mrm + 0.01

                    rnd_idx = 0
                    # rnd_idx = np.random.randint(video.shape[1], size=1)[0] # Using first frame of the sequence provides better results, using random images might be more robust
                    image_c = video[:, rnd_idx, :, :, :]

                    data = [speech, video, mrm, emotion, image_c]
                    
                    if self.global_step%2 == 0:
                        self.step_generator(data)
                    elif self.global_step%2 == 1:
                        with torch.no_grad():
                            if self.args.disc_pair or self.args.disc_frame or self.args.disc_video or self.args.disc_emo:
                                video_pd, _, _ = self.generator(image_c, speech, emotion)
                                video_pd = video_pd.detach()
                                data = [speech, video, mrm, emotion, image_c, video_pd]

                        if self.args.disc_frame:
                            self.step_disc_frame(data)

                        if self.args.disc_emo:
                            self.step_disc_emo(data)
                            
                    if self.global_step % 50 == 0:
                        self.logValImages(epoch)
                        self.saveNetworks('inter')

                    self.global_step += 1
            
            self.schdulerStep()
            self.displayLRs()

            self.saveNetworks('')


    def pre_train(self):
        for epoch in tqdm(range(self.args.num_epochs)):
            diterator = iter(self.train_loader)
            with trange(len(self.train_loader)) as t:     
                for i in t:               
                    speech, video, mrm, emotion = [d.float().to(self.args.device) for d in next(diterator)]
                    mrm = mrm.unsqueeze(2)
                    mrm = mrm + 0.01

                    rnd_idx = 0
                    image_c = video[:, rnd_idx, :, :, :]

                    data = [speech, video, mrm, emotion, image_c]
                    
                    self.step_disc_emo_recog(data)

                    self.logLosses(t)

                    if self.global_step % 500 == 0:
                        self.saveNetworks('inter')

                    self.global_step += 1
            
            self.schdulerStep()
            self.displayLRs()

            self.saveNetworks('')
            