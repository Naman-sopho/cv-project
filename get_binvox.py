import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import PIL.Image as Image
import sys
from torch.utils.data import Dataset, DataLoader
import pickle
import torchvision
from torchvision import datasets, models, transforms
import binvox_rw
from torch import distributions as dist
import cv2

#TODO: get mean image
mean_img = np.load("mean_img.npy")
#print(mean_img)

class SubSet(Dataset):
    def __init__(self,dataset,idx):
        self.ds = dataset
        self.idx = idx
    def __len__(self):
        return len(self.idx)
    def __getitem__(self,i):
        return self.ds[self.idx[i]]


class ImageVoxelDataset(Dataset):
    def __init__(self, num_images=1, is_voxel=False, input_shape=(32,32,32), 
                 img_loc = "./",
                 def_images=["0.jpg", "1.jpg", "2.jpg", "3.jpg"], debug=False):
        ## def_images=["000.npy", "001.npy", "002.npy", "003.npy"]
        self.num_images = num_images
        self.debug = debug
        self.img_loc = img_loc
        self.is_voxel = is_voxel
        self.input_shape = input_shape
        self.input_size = np.prod(np.array(self.input_shape))
        #self.objects = os.listdir(loc)
        #self.num_objects = len(self.objects)
        #self.pt_range = np.arange(32)
        #if self.debug:
        #    print(self.num_objects)
        self.def_images = def_images


    def __len__(self):
        return 32*32*32

    def __getitem__(self, i=32):
        # return preprocessed image inputs and outputs
        # generate random point in space
        
        # most probable conversion from i to each point
        # in __init__ have a list of n objects, each pointing to a folder
        # each of the n objects will self.input_size number of points in them
        
        if not self.is_voxel:
            if self.debug:
                print("Point")
            #object_num = i/self.input_size
            element_num = i%self.input_size
            x_size = self.input_shape[1]*self.input_shape[2]
            x = int(element_num/x_size)
            y = int((element_num-(x*x_size))/self.input_shape[2])
            z = int(element_num-(x*x_size)-(y*self.input_shape[2]))

            #x = int(np.random.choice(self.pt_range, size=1))
            #y = int(np.random.choice(self.pt_range, size=1))
            #z = int(np.random.choice(self.pt_range, size=1))

            point = np.array([x, y, z])
            #print(point)
            model = None
            #with open(self.loc+self.objects[int(object_num)]+"/model.binvox", 'rb') as f:
            #    model = binvox_rw.read_as_3d_array(f)
            #model = np.load(self.loc+self.objects[int(object_num)]+"/model.npy")
            #multi = int(32/self.input_shape[0])
            #occupied = model[multi*x, multi*y, multi*z]
            #occupied = model[x, y, z]

            images = None

            for img_num in range(self.num_images):
                #TODO: will not be np
                #image = np.array(np.load(self.img_loc+self.def_images[img_num]))
                #image = np.array(Image.open(self.img_loc+"/"+self.def_images[img_num]))
                image_path = self.img_loc+"/"+self.def_images[img_num]
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float64)
                image -= mean_img
                image /= 128.0
                if self.debug:
                    print(image.shape)
                if images is None:
                    images = np.zeros((image.shape[0],image.shape[1], image.shape[2]*self.num_images))
                #TODO: mean subtraction and change to np
                images[:,:,img_num*image.shape[2]:(img_num+1)*image.shape[2]] = image

            images = np.transpose(images,(2,0,1))
            if self.debug:
                print("Point", point)
                print("Images", images.shape)

            return {"image": images.astype(np.float32), "point":point.astype(np.float32)}




class ResNetFC(torch.nn.Module):

  def __init__(self, hidden_dim):
    
    super(ResNetFC, self).__init__()

    self.fc1 = nn.Linear(hidden_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    self.relu = nn.ReLU()

  def forward(self, input):
    
    x = self.fc1(self.relu(input))
    y = self.fc1(self.relu(x))

    return x+y


# In[56]:


class OccupancyNet(torch.nn.Module):
    # Only basic structure, need to add encoder
    def __init__(self, input_len=3, num_channels=3, num_images=1, debug=False):
        super(OccupancyNet, self).__init__()
        self.debug = debug
        self.num_channels = num_channels
        self.num_images = num_images
        self.input_len = input_len
        self.resnet18 = models.resnet18(pretrained=True)
        
        # freeze layers
        count = 1
        FREEZE_COUNT = 7
        for child in self.resnet18.children():
            count += 1
            if count >= FREEZE_COUNT:
                break
            for param in child.parameters():
                param.requires_grad = False
        print(count)
        
        self.decoder_conv = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=2, padding=5)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        
        self.decoder_linear = nn.Linear(6272, 1024)
        
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)

        if self.debug:
            print(self.resnet18)
            # pick out layer
            print(self.input_len)
        self.fc1 = nn.Linear(int(self.input_len),int(1024))
        self.fc2 = nn.Linear(int(1024), int(512))
        self.fc3 = nn.Linear(int(512), int(256))
        self.output_layer = nn.Linear(int(256), 1)
        
        self.fc_z = nn.Linear(int(256), int(256))

        self.resnet_b = ResNetFC(int(1024))
        self.resnet_b1 = ResNetFC(int(1024))
        self.resnet_b2 = ResNetFC(int(1024))
        self.resnet_b3 = ResNetFC(int(1024))
        self.resnet_b4 = ResNetFC(int(1024))

    def forward_resnet(self, input):
        # I guess this should work, not sure though
        x = self.resnet18.conv1(input)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        
        x = self.conv2(x)

        # need to check and change shape
        return x

    def encoder(self, images):
        y = None
        for i in range(self.num_images):
            if self.debug:
                print(images[:,self.num_channels*i:self.num_channels*(i+1),:,:].shape)
            
            if y is None:
                y = self.forward_resnet(images[:,self.num_channels*i:self.num_channels*(i+1),:,:])
            else:
                y = y.clone() + self.forward_resnet(images[:,self.num_channels*i:self.num_channels*(i+1),:,:])

        return y
    
    def decoder(self, inp):
        x = self.decoder_conv(inp)
        x = self.bn1(x)
        x = torch.flatten(x,1)
        x = self.relu1(x)
        x = self.decoder_linear(x).squeeze()
        
        return x
    
    def loss_addon(self, pt, z, img):
        
#         net_z = self.fc3(z).unsqueeze(1)
#         net = pt + net_z
        net = pt
        y = self.decoder(img)
        x = self.combiner(net, y)
        return x
    
    def combiner(self, point, y):
        x = self.fc1(point)
        x += y
        # x = self.fc2(x)
        # x += y

        x = self.resnet_b(x)
        x = self.resnet_b1(x)
        x = self.resnet_b2(x)
        x = self.resnet_b3(x)
        x = self.resnet_b4(x)

        x = self.relu1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.output_layer(x)

        return x
    
    def forward(self, images, point):
        # not complete architecture
        y = self.encoder(images)
        if self.debug:
            print(y.shape)
            
        y = self.decoder(y)
        
        x = self.combiner(point, y)
        
        sig = nn.Sigmoid()
        return sig(x)



def infer_z(p, occ, c, **kwargs):
        ''' Infers z.
        Args:
            p (tensor): points tensor
            occ (tensor): occupancy values for occ
            c (tensor): latent conditioned code c
        '''
        en_latent = None
        if en_latent is not None:
            latent = en_latent.to(device)
            mean_z, logstd_z = latent.combiner(p, occ, c, **kwargs)
        else:
            batch_size = p.size(0)
            mean_z = torch.empty(batch_size, 0).to(device)
            logstd_z = torch.empty(batch_size, 0).to(device)

        q_z = dist.Normal(mean_z, torch.exp(logstd_z))
        return q_z

def encode_inputs(inputs):
    ''' Encodes the input.
    Args:
        input (tensor): the input
    '''

    if occ_net.forward_resnet is not None:
        c = occ_net.encoder(inputs.to(device))
#         c = occ_net.decoder(c)
    else:
        # Return inputs?
        c = torch.empty(inputs.size(0), 0)

    return c



def compute_loss(p, occ, images):
        ''' Computes the loss.
        '''
        inputs = images.to(device)

        kwargs = {}
        p0_z = dist.Normal(torch.tensor([]).to(device), torch.tensor([]).to(device))
        
        c = encode_inputs(inputs)
        q_z = infer_z(p, occ, c, **kwargs)
        z = q_z.rsample()

        # KL-divergence
        kl = dist.kl_divergence(q_z, p0_z).sum(dim=-1)
        loss = kl.mean()

        # General points
        logits = dist.Bernoulli(logits=occ_net.loss_addon(p, z, c).squeeze()).logits
#         logits = logits.usqueeze(-11)
        loss_i = F.binary_cross_entropy_with_logits(
            logits, occ, reduction='none')
        loss = loss + loss_i.sum(-1).mean()

        return loss


#Train function returns the training data accuracy, dev data accuracy, and loss for each epoch. This might be needed for plotting the graphs.
def img2binvox(model, train_data, batch_size, device, thresh, load=True):
  #train_acc=[]
  #val_acc=[]
  #loss_data=[]
  #TODO: PATH Variable
  PATH = "./16"
  
  start=0
  
  if load:
    checkpoint = torch.load(PATH) #set Path here for loading data
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #for state in optimizer.state.values():
    #    for k, v in state.items():
    #        if isinstance(v, torch.Tensor):
    #            state[k] = v.cuda()
    #start = checkpoint['epoch']
    #loss = checkpoint['loss']
    model.train()
  else:
      print("Something was supposed to be loaded!")
      exit(0)

  model = model.to(device)
  vox = np.zeros((32,32,32)).astype(np.bool)
  with torch.no_grad():
    #for epoch in range(start, num_epochs):
    #print("Epoch {}...".format(epoch))
    index = 0
    totali = len(train_data)
    correct = 0
    total = 0
    ##############################################################
    #batch_loss = np.array([])
    #max_op = -np.inf
    ##############################################################
    for data in train_data:
      if index %10 == 0:
          print("Batch number: {} of {}".format(index, totali), end="\r")
      index +=1
      local_X = data["image"].float()
      #local_Y = data["occupied"].float()
      local_Z = data["point"].float()
      #local_Y = local_Y.to(device)
      local_X = local_X.to(device)
      local_Z = local_Z.to(device)
      
      #optimizer.zero_grad()
    
      outputs=model(local_X, local_Z)
      
      predicted = np.where(outputs.cpu() > thresh, True, False)
      pts = local_Z.cpu()

      for i, pt in enumerate(pts):
          vox[int(pt[0]), int(pt[1]), int(pt[2])] = predicted[i]


      
      #loss=compute_loss(local_Z, local_Y, local_X)
      #loss.backward()

      #optimizer.step()
      '''
      with torch.no_grad():
        predicted = np.where(outputs.cpu() > thresh, True, False)
        #total += local_Y.size(0)
        total += np.sum(np.where((np.array(local_Y.unsqueeze(1).cpu()) == 1), 1, 0))
        correct += np.sum(np.where(((predicted == np.array(local_Y.unsqueeze(1).cpu())) & (np.array(local_Y.unsqueeze(1).cpu())== 1)), 1, 0))
        
        ######################################################
        batch_loss = np.append(batch_loss,loss.item())
        max_op = max(np.max(np.array(outputs.cpu())), max_op)
        ######################################################

    if scheduler_present:
      scheduler.step()
    '''
    
    #tr=model_accuracy(model, train_data, batch_size, device, thresh)
    
    ##########################################################
    '''
    with torch.no_grad():
        tr = 100 * (correct / total)

    #     val=model_accuracy(model, val_data, batch_size, device, thresh)

        train_acc.append(tr)
    #     val_acc.append(val)
        loss_data.append(np.mean(batch_loss))

        if epoch%1==0:
    #       print("Epoch:", epoch, loss.item(), "Train Acc:", tr)
            print("Epoch:", epoch, loss_data[-1], "Train Acc:", tr, "Max O/P:", max_op)


        #Saving the Model (Current path in Google Drive)

        if save and epoch%1==0:
          torch.save({'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'loss': loss,}, "./"+str(epoch)) #set path here for saving data to desired location
    ###############################################################
    '''
  return vox



def get_binvox():
    debug = False
    occ_net = OccupancyNet(debug=debug) #.float()
    
    ivd = ImageVoxelDataset(debug=debug)


    train_idx = None
    train_objects = [0]
    for objs in train_objects:
        if train_idx is None:
            train_idx = np.arange(objs*ivd.input_size,(objs+1)*ivd.input_size)
        else:
            train_idx = np.append(train_idx, np.arange(objs*ivd.input_size,(objs+1)*ivd.input_size))

    train_data = SubSet(ivd,train_idx)

    workers = 9
    batch_size = 2000
    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=False,num_workers=workers)

    thresh = 0.2
    use_cuda = torch.cuda.is_available()
    print('Cuda Flag: ',use_cuda)
    #use_cuda = False
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(device)
    vox = img2binvox(occ_net, train_loader, batch_size, device, thresh)
    vox = binvox_rw.Voxels(points[key].T, [32, 32, 32], (0, 0, 0), 0, 'xyz')
    vox_file = open('model.binvox', 'wb')
    vox.write(vox_file)

    return 0

def main():
    print('sdlfjldks')
    vox = get_binvox()
    print(np.sum(vox))

if __name__ == "__main__":
    main()