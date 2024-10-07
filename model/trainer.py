import torch
import time
import json
import os
import torch.nn.functional as F



class Trainer(object):
    def __init__(self, model,
                dataset, args, val_dataset, forward_diffusion, optimizer, writer, timesteps, p_uncond, lr_scheduler=None,
                init_epoch=1, last_epoch=15):


        self.model = model
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.args = args

        self.forward_diffusion = forward_diffusion

        self.timesteps = timesteps
        self.p_uncond = p_uncond

        self.step = 0
        self.total_step = 0
        self.epoch = init_epoch
        self.last_epoch = last_epoch
        self.best_val_loss = 1e18
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.writer = writer

        self.losses = []
        self.val_losses = []

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(torch.cuda.is_available())


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def p_losses(self, denoise_model, x_start, t, class_, noise=None, loss_type="l1"):
        if noise is None:
            noise = torch.randn_like(x_start).to(self.device)

        x_noisy = self.forward_diffusion.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t, class_)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss


    def train(self):

        self.model.to(self.device)
        #self.model = torch.compile(self.model)

        stepComputeTime = time.time()
        mes = "Epoch {}, step:{}/{} {:.2f}%, Loss:{:.4f}, Perplexity:{:.4f}, time (s): {:.2f}, Epochtime (min): {:.2f}, lr: {:.6f}"
        print("Training on 👀: ", self.device)
 
        while self.epoch < self.last_epoch:
            losses = 0.0

            for i, data in enumerate(self.dataset):

                #prof.step()
                self.model.train()
                self.optimizer.zero_grad()

                imgs, class_ = data # imgs -> (batch_size, 1, 28, 28), class_ -> (batch_size,)
                t = torch.randint(0, self.timesteps, (self.args.batch_size,), device=self.device).long().to(self.device)
                
                imgs.to(self.device)

                # randomly asign class 10 to p_uncond of the data
                class_ = torch.where(torch.rand(self.args.batch_size) < self.p_uncond, torch.tensor(10), class_)

                class_.to(self.device)

                loss = self.p_losses(self.model, imgs, t, class_, loss_type="huber")

                loss.backward()

                self.optimizer.step()
                self.lr_scheduler.step()

                self.step += 1
                self.total_step += 1
                    
                losses += loss.item()

                if self.device == "cuda":
                    torch.cuda.synchronize()

                # log message
                if self.step % self.args.print_freq == 0:
                    avg_loss = losses / self.args.print_freq
                    remaining_time = (time.time() - stepComputeTime) * (len(self.dataset)-self.step)/(60 * self.args.print_freq)
                    
                    print(mes.format(
                        self.epoch, self.step, len(self.dataset),
                        100 * self.step / len(self.dataset),
                        avg_loss,
                        2**avg_loss,
                        time.time() - stepComputeTime,
                        remaining_time,
                        self.lr_scheduler.get_last_lr()[0]

                    ))

                    self.writer.add_scalar("Loss", avg_loss, self.total_step)
                    self.writer.add_scalar("Lr", self.lr_scheduler.get_last_lr()[0], self.total_step)
                    stepComputeTime = time.time()
                    self.losses.append(avg_loss)
                    losses = 0.0

            self.epoch += 1
            self.step = 0

            self.save_model()

    def validate(self, limit=300):

        val_total_loss = 0.0

        for i, data in enumerate(self.val_dataset):
            if i == limit:
                break
            imgs, class_ = data # imgs -> (batch_size, 1, 28, 28), class_ -> (batch_size,)
            t = torch.randint(0, self.timesteps, (self.args.batch_size,), device=self.device).long()
            class_ = torch.where(torch.rand(self.args.batch_size) < self.p_uncond, torch.tensor(10), class_)
            
            with torch.no_grad():

                loss = self.p_losses(self.model, imgs, t, class_, loss_type="huber")

            val_total_loss += loss
            i += 1

        avg_loss = val_total_loss / limit

        self.writer.add_scalar("Validation Loss", avg_loss, self.total_step)
        self.val_losses.append(avg_loss)
        self.writer.flush()


    def save_model(self):
        print("saving model...")
        self.validate()
        path = self.args.save_dir + f"checkpoint_epoch_{self.epoch}_{round(self.step/len(self.dataset)*100, 3)}%_estimated_loss_{round(float(self.val_losses[-1]), 3)}"
        if not os.path.exists(path= path):
            os.makedirs(path)
        
        params = {
            
            "epoch": self.epoch,
            "step": self.step,
            "batch_size": self.args.batch_size,
            "timesteps": self.timesteps,
            "p_uncond": self.p_uncond,
            "total_step": self.total_step,
            "n_classes": self.args.n_classes,
            "img_size": self.args.img_size,
            "channels": self.args.channels,
            "dim_mult": self.args.dim_mults,

            }
        
        with open(os.path.join(path, "params.json"), "w") as f:
            json.dump(params, f)
        
        
        torch.save(self.model.state_dict(), os.path.join(path, "model.pth"))
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pth"))
        torch.save(self.lr_scheduler.state_dict(), os.path.join(path, "lr_scheduler.pth"))

        print("model saved successfully...")



    