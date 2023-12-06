import torch.nn as nn
import torch.nn.functional as F
import torch

class fc_only(nn.Module):
    def __init__(self, flux_length):
        super().__init__()
        self.fc1 = nn.Linear(flux_length, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input if necessary
        x = self.fc1(x)
        return x
    
class fc2x_only(nn.Module):
    def __init__(self, flux_length):
        super().__init__()
        self.fc1 = nn.Linear(flux_length, 1)  # One neuron, equivalent to the Conv1d layer
        self.fc2 = nn.Linear(1, 2)  # Output layer

    def forward(self, x, return_input=False):
        input_data = x.view(x.size(0), -1)
        x = F.relu(self.fc1(input_data))  # Mimicking Conv1d followed by ReLU
        x = self.fc2(x)
        
        if return_input:
            return x, input_data
        else:
            return x 
    
class CNN_0a(nn.Module):
    def __init__(self, flux_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 5, kernel_size=1) 
        
        # Dummy data to get the output size after convolutions and pooling
        x = torch.randn(1, 1, flux_length)
        x = F.relu(self.conv1(x))
        linear_input_size = x.view(-1).size(0)
        self.fc1 = nn.Linear(linear_input_size, 2)

    def forward(self, x, return_conv=False):
        conv1_out = F.relu(self.conv1(x))
        x = conv1_out.view(conv1_out.size(0), -1)
        x = self.fc1(x)
        
        if return_conv:
            return x, conv1_out
        else:
            return x       
        
class CNN_0b(nn.Module):
    def __init__(self, flux_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 1, kernel_size=1) 
        
        # Dummy data to get the output size after convolutions and pooling
        x = torch.randn(1, 1, flux_length)
        x = F.relu(self.conv1(x))
        linear_input_size = x.view(-1).size(0)
        self.fc1 = nn.Linear(linear_input_size, 2)

    def forward(self, x, return_conv=False):
        conv1_out = F.relu(self.conv1(x))
        x = conv1_out.view(conv1_out.size(0), -1)
        x = self.fc1(x)
        
        if return_conv:
            return x
        else:
            return x    
        
class CNN_0b_noRelu(nn.Module):
    def __init__(self, flux_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 1, kernel_size=1) 
        
        # Dummy data to get the output size after convolutions
        x = torch.randn(1, 1, flux_length)
        x = self.conv1(x)
        linear_input_size = x.view(-1).size(0)
        self.fc1 = nn.Linear(linear_input_size, 2)

    def forward(self, x, return_conv=False):
        conv1_out = self.conv1(x)
        x = conv1_out.view(conv1_out.size(0), -1)
        x = self.fc1(x)
        
        if return_conv:
            return x, conv1_out
        else:
            return x    
        
class CNN_0d(nn.Module):
    def __init__(self, flux_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 1, kernel_size=5) 
        
        # Dummy data to get the output size after convolutions and pooling
        x = torch.randn(1, 1, flux_length)
        x = F.relu(self.conv1(x))
        linear_input_size = x.view(-1).size(0)
        self.fc1 = nn.Linear(linear_input_size, 2)

    def forward(self, x, return_conv=False):
        conv1_out = F.relu(self.conv1(x))
        x = conv1_out.view(conv1_out.size(0), -1)
        x = self.fc1(x)
        
        if return_conv:
            return x
        else:
            return x  
        
class CNN_0e(nn.Module):
    def __init__(self, flux_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 1, kernel_size=1)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=1)
                
        # Dummy data to get the output size after convolutions and pooling
        x = torch.randn(1, 1, flux_length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Recalculate the input size for the first fully connected layer
        linear_input_size = x.view(-1).size(0)
        self.fc1 = nn.Linear(linear_input_size, 2)

    def forward(self, x, return_conv=False):
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(conv1_out))
        x = conv2_out.view(conv2_out.size(0), -1)
        x = self.fc1(x)
        
        if return_conv:
            return x, conv2_out
        else:
            return x 

class CNN_1a(nn.Module):
    def __init__(self, flux_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 5, kernel_size=5) 
        
        # Dummy data to get the output size after convolutions and pooling
        x = torch.randn(1, 1, flux_length)
        x = F.relu(self.conv1(x))
        linear_input_size = x.view(-1).size(0)
        self.fc1 = nn.Linear(linear_input_size, 2)

    def forward(self, x, return_conv=False):
        conv1_out = F.relu(self.conv1(x))
        x = conv1_out.view(conv1_out.size(0), -1)
        x = self.fc1(x)
        
        if return_conv:
            return x, conv1_out
        else:
            return x  
        
class CNN_1b(nn.Module):
    def __init__(self, flux_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 15, kernel_size=20) 
        
        # Dummy data to get the output size after convolutions and pooling
        x = torch.randn(1, 1, flux_length)
        x = F.relu(self.conv1(x))
        linear_input_size = x.view(-1).size(0)
        self.fc1 = nn.Linear(linear_input_size, 2)

    def forward(self, x, return_conv=False):
        conv1_out = F.relu(self.conv1(x))
        x = conv1_out.view(conv1_out.size(0), -1)
        x = self.fc1(x)
        
        if return_conv:
            return x, conv1_out
        else:
            return x  
        

class CNN_1c(nn.Module):
    def __init__(self, flux_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 30, kernel_size=20) 
        
        # Dummy data to get the output size after convolutions and pooling
        x = torch.randn(1, 1, flux_length)
        x = F.relu(self.conv1(x))
        linear_input_size = x.view(-1).size(0)
        self.fc1 = nn.Linear(linear_input_size, 2)

    def forward(self, x, return_conv=False):
        conv1_out = F.relu(self.conv1(x))
        x = conv1_out.view(conv1_out.size(0), -1)
        x = self.fc1(x)
        
        if return_conv:
            return x, conv1_out
        else:
            return x  
        
class CNN_1c_2fc(nn.Module):
    def __init__(self, flux_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 5, kernel_size=20) 
        
        # Dummy data to get the output size after convolutions and pooling
        x = torch.randn(1, 1, flux_length)
        x = F.relu(self.conv1(x))
        linear_input_size = x.view(-1).size(0)
        
        self.fc1 = nn.Linear(linear_input_size, 100)  # output size is 100
        self.fc2 = nn.Linear(100, 2)  # output size is 2

    def forward(self, x, return_conv=False):
        conv1_out = F.relu(self.conv1(x))
        x = conv1_out.view(conv1_out.size(0), -1)
        x = F.relu(self.fc1(x))  # apply ReLU after fc1
        x = self.fc2(x)  # then pass through fc2
        
        if return_conv:
            return x, conv1_out
        else:
            return x
        
class CNN_1c_3fc(nn.Module):
    def __init__(self, flux_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 5, kernel_size=20) 
        
        # Dummy data to get the output size after convolutions and pooling
        x = torch.randn(1, 1, flux_length)
        x = F.relu(self.conv1(x))
        linear_input_size = x.view(-1).size(0)
        
        self.fc1 = nn.Linear(linear_input_size, 100)  # output size is 100
        self.fc2 = nn.Linear(100, 50)  # output size is 50
        self.fc3 = nn.Linear(50, 2)  # output size is 2

    def forward(self, x, return_conv=False):
        conv1_out = F.relu(self.conv1(x))
        x = conv1_out.view(conv1_out.size(0), -1)
        x = F.relu(self.fc1(x))  # apply ReLU after fc1
        x = F.relu(self.fc2(x))  # apply ReLU after fc2
        x = self.fc3(x)  # then pass through fc3
        
        if return_conv:
            return x, conv1_out
        else:
            return x

class CNN_1d(nn.Module):
    def __init__(self, flux_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 50, kernel_size=20) 
        
        # Dummy data to get the output size after convolutions and pooling
        x = torch.randn(1, 1, flux_length)
        x = F.relu(self.conv1(x))
        linear_input_size = x.view(-1).size(0)
        self.fc1 = nn.Linear(linear_input_size, 2)

    def forward(self, x, return_conv=False):
        conv1_out = F.relu(self.conv1(x))
        x = conv1_out.view(conv1_out.size(0), -1)
        x = self.fc1(x)
        
        if return_conv:
            return x, conv1_out
        else:
            return x  
        
class CNN_1e(nn.Module):
    def __init__(self, flux_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 30, kernel_size=100) 
        
        # Dummy data to get the output size after convolutions and pooling
        x = torch.randn(1, 1, flux_length)
        x = F.relu(self.conv1(x))
        linear_input_size = x.view(-1).size(0)
        self.fc1 = nn.Linear(linear_input_size, 2)

    def forward(self, x, return_conv=False):
        conv1_out = F.relu(self.conv1(x))
        x = conv1_out.view(conv1_out.size(0), -1)
        x = self.fc1(x)
        
        if return_conv:
            return x, conv1_out
        else:
            return x  
        
        
class CNN_2a(nn.Module):
    def __init__(self, flux_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 5, kernel_size=20)
        self.conv2 = nn.Conv1d(5, 10, kernel_size=20)
                
        # Dummy data to get the output size after convolutions and pooling
        x = torch.randn(1, 1, flux_length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Recalculate the input size for the first fully connected layer
        linear_input_size = x.view(-1).size(0)
        self.fc1 = nn.Linear(linear_input_size, 2)

    def forward(self, x, return_conv=False):
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(conv1_out))
        x = conv2_out.view(conv2_out.size(0), -1)
        x = self.fc1(x)
        
        if return_conv:
            return x, conv2_out
        else:
            return x  
        
class CNN_2a_leakyRelu(nn.Module):
    def __init__(self, flux_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 5, kernel_size=20)
        self.conv2 = nn.Conv1d(5, 10, kernel_size=20)
                
        # Dummy data to get the output size after convolutions and pooling
        x = torch.randn(1, 1, flux_length)
        x = F.leaky_relu(self.conv1(x))  # Changed to Leaky ReLU
        x = F.leaky_relu(self.conv2(x))  # Changed to Leaky ReLU

        # Recalculate the input size for the first fully connected layer
        linear_input_size = x.view(-1).size(0)
        self.fc1 = nn.Linear(linear_input_size, 2)

    def forward(self, x, return_conv=False):
        conv1_out = F.leaky_relu(self.conv1(x))  # Changed to Leaky ReLU
        conv2_out = F.leaky_relu(self.conv2(conv1_out))  # Changed to Leaky ReLU
        x = conv2_out.view(conv2_out.size(0), -1)
        x = self.fc1(x)
        
        if return_conv:
            return x, conv2_out
        else:
            return x
        
class CNN_2a_sigmoid(nn.Module):
    def __init__(self, flux_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 5, kernel_size=20)
        self.conv2 = nn.Conv1d(5, 10, kernel_size=20)
                
        # Dummy data to get the output size after convolutions and pooling
        x = torch.randn(1, 1, flux_length)
        x = torch.sigmoid(self.conv1(x))  # Changed to Sigmoid
        x = torch.sigmoid(self.conv2(x))  # Changed to Sigmoid

        # Recalculate the input size for the first fully connected layer
        linear_input_size = x.view(-1).size(0)
        self.fc1 = nn.Linear(linear_input_size, 2)

    def forward(self, x, return_conv=False):
        conv1_out = torch.sigmoid(self.conv1(x))  # Changed to Sigmoid
        conv2_out = torch.sigmoid(self.conv2(conv1_out))  # Changed to Sigmoid
        x = conv2_out.view(conv2_out.size(0), -1)
        x = self.fc1(x)
        
        if return_conv:
            return x, conv2_out
        else:
            return x
        
class CNN_2a_tanh(nn.Module):
    def __init__(self, flux_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 5, kernel_size=20)
        self.conv2 = nn.Conv1d(5, 10, kernel_size=20)
                
        # Dummy data to get the output size after convolutions and pooling
        x = torch.randn(1, 1, flux_length)
        x = torch.tanh(self.conv1(x))  # Changed to Tanh
        x = torch.tanh(self.conv2(x))  # Changed to Tanh

        # Recalculate the input size for the first fully connected layer
        linear_input_size = x.view(-1).size(0)
        self.fc1 = nn.Linear(linear_input_size, 2)

    def forward(self, x, return_conv=False):
        conv1_out = torch.tanh(self.conv1(x))  # Changed to Tanh
        conv2_out = torch.tanh(self.conv2(conv1_out))  # Changed to Tanh
        x = conv2_out.view(conv2_out.size(0), -1)
        x = self.fc1(x)
        
        if return_conv:
            return x, conv2_out
        else:
            return x
        

        
class CNN_2a_2fc(nn.Module):
    def __init__(self, flux_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 5, kernel_size=20)
        self.conv2 = nn.Conv1d(5, 10, kernel_size=20)
                
        # Dummy data to get the output size after convolutions and pooling
        x = torch.randn(1, 1, flux_length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Recalculate the input size for the first fully connected layer
        linear_input_size = x.view(-1).size(0)
        self.fc1 = nn.Linear(linear_input_size, 100)  # output size is 100
        self.fc2 = nn.Linear(100, 2)  # output size is 2

    def forward(self, x, return_conv=False):
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(conv1_out))  # Apply conv2
        x = conv2_out.view(conv2_out.size(0), -1)  # Flatten conv2_out
        x = F.relu(self.fc1(x))  # apply ReLU after fc1
        x = self.fc2(x)
        
        if return_conv:
            return x, conv2_out
        else:
            return x 
        
class CNN_2a_3fc(nn.Module):
    def __init__(self, flux_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 5, kernel_size=20)
        self.conv2 = nn.Conv1d(5, 10, kernel_size=20)
                
        # Dummy data to get the output size after convolutions and pooling
        x = torch.randn(1, 1, flux_length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Recalculate the input size for the first fully connected layer
        linear_input_size = x.view(-1).size(0)
        self.fc1 = nn.Linear(linear_input_size, 100)  # output size is 100
        self.fc2 = nn.Linear(100, 50)  # output size is 50
        self.fc3 = nn.Linear(50, 2)    # output size is 2

    def forward(self, x, return_conv=False):
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(conv1_out))  # Apply conv2
        x = conv2_out.view(conv2_out.size(0), -1)  # Flatten conv2_out
        x = F.relu(self.fc1(x))  # apply ReLU after fc1
        x = F.relu(self.fc2(x))  # apply ReLU after fc2
        x = self.fc3(x)          # apply third fully connected layer
        
        if return_conv:
            return x, conv2_out
        else:
            return x    
        
class CNN_2b(nn.Module):
    def __init__(self, flux_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 15, kernel_size=20)
        self.conv2 = nn.Conv1d(15, 30, kernel_size=20)
                
        # Dummy data to get the output size after convolutions and pooling
        x = torch.randn(1, 1, flux_length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Recalculate the input size for the first fully connected layer
        linear_input_size = x.view(-1).size(0)
        self.fc1 = nn.Linear(linear_input_size, 2)

    def forward(self, x, return_conv=False):
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(conv1_out))
        x = conv2_out.view(conv2_out.size(0), -1)
        x = self.fc1(x)
        
        if return_conv:
            return x, conv2_out
        else:
            return x 
        
class CNN_2c(nn.Module):
    def __init__(self, flux_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 25, kernel_size=20)
        self.conv2 = nn.Conv1d(25, 50, kernel_size=20)
                
        # Dummy data to get the output size after convolutions and pooling
        x = torch.randn(1, 1, flux_length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Recalculate the input size for the first fully connected layer
        linear_input_size = x.view(-1).size(0)
        self.fc1 = nn.Linear(linear_input_size, 2)

    def forward(self, x, return_conv=False):
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(conv1_out))
        x = conv2_out.view(conv2_out.size(0), -1)
        x = self.fc1(x)
        
        if return_conv:
            return x, conv2_out
        else:
            return x  

class CNN_5a(nn.Module):
    def __init__(self, flux_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 3, kernel_size=20)
        self.conv2 = nn.Conv1d(3, 6, kernel_size=20)
        self.conv3 = nn.Conv1d(6, 9, kernel_size=20)
        self.conv4 = nn.Conv1d(9, 12, kernel_size=20)
        self.conv5 = nn.Conv1d(12, 15, kernel_size=20)

        # Dummy data to get the output size after convolutions and pooling
        x = torch.randn(1, 1, flux_length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        linear_input_size = x.view(-1).size(0)

        self.fc1 = nn.Linear(linear_input_size, 2)

    def forward(self, x, return_conv=False):
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv4_out = F.relu(self.conv4(conv3_out))
        conv5_out = F.relu(self.conv5(conv4_out))
        x = conv5_out.view(conv5_out.size(0), -1)
        x = self.fc1(x)

        if return_conv:
            return x, conv1_out
        else:
            return x
        
class CNN_5b(nn.Module):
    def __init__(self, flux_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 10, kernel_size=20)
        self.conv2 = nn.Conv1d(10, 20, kernel_size=20)
        self.conv3 = nn.Conv1d(20, 30, kernel_size=20)
        self.conv4 = nn.Conv1d(30, 40, kernel_size=20)
        self.conv5 = nn.Conv1d(40, 50, kernel_size=20)

        # Dummy data to get the output size after convolutions and pooling
        x = torch.randn(1, 1, flux_length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        linear_input_size = x.view(-1).size(0)

        self.fc1 = nn.Linear(linear_input_size, 2)

    def forward(self, x, return_conv=False):
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv4_out = F.relu(self.conv4(conv3_out))
        conv5_out = F.relu(self.conv5(conv4_out))
        x = conv5_out.view(conv5_out.size(0), -1)
        x = self.fc1(x)

        if return_conv:
            return x, conv1_out
        else:
            return x
        
class CNN_5c(nn.Module):
    def __init__(self, flux_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 10, kernel_size=35)
        self.conv2 = nn.Conv1d(10, 20, kernel_size=15)
        self.conv3 = nn.Conv1d(20, 30, kernel_size=15)
        self.conv4 = nn.Conv1d(30, 40, kernel_size=5)
        self.conv5 = nn.Conv1d(40, 50, kernel_size=3)

        # Dummy data to get the output size after convolutions and pooling
        x = torch.randn(1, 1, flux_length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        linear_input_size = x.view(-1).size(0)

        self.fc1 = nn.Linear(linear_input_size, 2)

    def forward(self, x, return_conv=False):
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv4_out = F.relu(self.conv4(conv3_out))
        conv5_out = F.relu(self.conv5(conv4_out))
        x = conv5_out.view(conv5_out.size(0), -1)
        x = self.fc1(x)

        if return_conv:
            return x, conv1_out
        else:
            return x
        
class CNN_10a(nn.Module):
    def __init__(self, flux_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 10, kernel_size=35)
        self.conv2 = nn.Conv1d(10, 20, kernel_size=35)
        self.conv3 = nn.Conv1d(20, 30, kernel_size=35)
        self.conv4 = nn.Conv1d(30, 40, kernel_size=35)
        self.conv5 = nn.Conv1d(40, 50, kernel_size=35)
        self.conv6 = nn.Conv1d(50, 60, kernel_size=35)
        self.conv7 = nn.Conv1d(60, 70, kernel_size=35)
        self.conv8 = nn.Conv1d(70, 80, kernel_size=35)
        self.conv9 = nn.Conv1d(80, 90, kernel_size=35)
        self.conv10 = nn.Conv1d(90, 100, kernel_size=35)

        # Dummy data to get the output size after convolutions and pooling
        x = torch.randn(1, 1, flux_length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        linear_input_size = x.view(-1).size(0)

        self.fc1 = nn.Linear(linear_input_size, 2)

    def forward(self, x, return_conv=False):
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv4_out = F.relu(self.conv4(conv3_out))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv7_out = F.relu(self.conv7(conv6_out))
        conv8_out = F.relu(self.conv8(conv7_out))
        conv9_out = F.relu(self.conv9(conv8_out))
        conv10_out = F.relu(self.conv10(conv9_out))
        x = conv10_out.view(conv10_out.size(0), -1)
        x = self.fc1(x)

        if return_conv:
            return x, conv1_out
        else:
            return x

