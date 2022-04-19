current_depth = 4
current_res = np.power(2, current_depth + 2)
print("Currently working on depth: %d", current_depth + 1)
print("Current resolution: %d x %d" % (current_res, current_res))

ticker = 1

# Choose training parameters and configure training ops.
# TO DO
data = get_data_loader(dataset, batch_sizes[current_depth], num_workers)

    # start = timeit.default_timer()  # record time at the start of epoch

    # print("Epoch: [%d]" % epoch)
    # total_batches = len(iter(data))
    # total_batches = len(data)

    # fade_point = int((fade_in_percentage[current_depth] / 100)
                        # * epochs[current_depth] * total_batches)

    # for i, batch in enumerate(data, 1):
        # print(i)
        # calculate the alpha for fading in the layers
alpha =  1

        # extract current batch of data for training
point = torch.randn(1, latent_size).cuda()
                
point = (point / point.norm()) * (latent_size ** 0.5)
                
        
gan_input = torch.randn(1, self.latent_size,dtype=torch.float32).cuda()
fixed_labels = None

gen(gan_input, current_depth, alpha, labels_in=fixed_labels).detach()
        # # optimize the discriminator:
        # dis_loss = self.optimize_discriminator(gan_input, images, current_depth, alpha, labels)

        # # optimize the generator:
        # gen_loss = self.optimize_generator(gan_input, images, current_depth, alpha, labels)

        # # provide a loss feedback
        # if i % int(total_batches / feedback_factor + 1) == 0 or i == 1:
        #     elapsed = time.time() - global_time
        #     elapsed = str(datetime.timedelta(seconds=elapsed)).split('.')[0]
        #     print(
        #         "Elapsed: [%s] Epoch:%d Step: %d  Batch: %d  D_Loss: %f  G_Loss: %f"
        #         % (elapsed, epoch,step, i, dis_loss, gen_loss))

        #     # create a grid of samples and save it
        #     # os.makedirs(os.path.join(output, 'samples'), exist_ok=True)
        #     gen_img_file = os.path.join(output, 'samples', "gen_" + str(current_depth)
        #                                 + "_" + str(epoch) + "_" + str(i) + ".png")

        #     with torch.no_grad():
        #         print('##################')                        
        #         print("current_depth: %d alpha:%d"
        #         % (current_depth, alpha))   
        #         print('use_ema',self.use_ema)
        #         print('scale_factor',int(
        #                 np.power(2, self.depth - current_depth - 1)) if self.structure == 'linear' else 1) 
        #         print('##################')                        
# self.create_grid(
#     samples=self.gen(fixed_input, current_depth, alpha, labels_in=fixed_labels).detach() if not self.use_ema
#     else self.gen_shadow(fixed_input, current_depth, alpha, labels_in=fixed_labels).detach(),
#     scale_factor=int(
#         np.power(2, self.depth - current_depth - 1)) if self.structure == 'linear' else 1,
#     img_file=gen_img_file,
# )

        # increment the alpha ticker and the step
    #     ticker += 1
    #     step += 1

    # elapsed = timeit.default_timer() - start
    # elapsed = str(datetime.timedelta(seconds=elapsed)).split('.')[0]
    # print("Time taken for epoch: %s\n" % elapsed)

    # if epoch % checkpoint_factor == 0 or epoch == 1 or epoch == epochs[current_depth]:
    #     save_dir = os.path.join(output, 'models')
    #     os.makedirs(save_dir, exist_ok=True)
    #     gen_save_file = os.path.join(save_dir, "GAN_GEN_" + str(current_depth) + "_" + str(epoch) + ".pth")
    #     dis_save_file = os.path.join(save_dir, "GAN_DIS_" + str(current_depth) + "_" + str(epoch) + ".pth")
    #     gen_optim_save_file = os.path.join(
    #         save_dir, "GAN_GEN_OPTIM_" + str(current_depth) + "_" + str(epoch) + ".pth")
    #     dis_optim_save_file = os.path.join(
    #         save_dir, "GAN_DIS_OPTIM_" + str(current_depth) + "_" + str(epoch) + ".pth")

    #     torch.save(self.gen.state_dict(), gen_save_file)
    #     print("Saving the model to: %s\n" % gen_save_file)
    #     torch.save(self.dis.state_dict(), dis_save_file)
    #     torch.save(self.gen_optim.state_dict(), gen_optim_save_file)
    #     torch.save(self.dis_optim.state_dict(), dis_optim_save_file)

    #     # also save the shadow generator if use_ema is True
    #     if self.use_ema:
    #         gen_shadow_save_file = os.path.join(
    #             save_dir, "GAN_GEN_SHADOW_" + str(current_depth) + "_" + str(epoch) + ".pth")
    #         torch.save(self.gen_shadow.state_dict(), gen_shadow_save_file)
    #         print("Saving the model to: %s\n" % gen_shadow_save_file)
