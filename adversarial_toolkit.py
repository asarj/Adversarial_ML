from data_loader import ImageDataset
from mnist_cnn_pytorch import CNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class Attacker:
    def __init__(self, model: nn.Module, dataset: ImageDataset, verbose=1):
        self.model = model
        self.dataset = dataset
        self.verbose = verbose
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def attack(self, attack_func):
        accs, examples = list(), list()
        epsilons = [0, .05, .1, .15, .2, .25, .3]

        for e in epsilons:
            acc, ex = self.evaluate(attack_func, e)
            accs.append(acc)
            examples.append(ex)

        if self.verbose:
            plt.figure(figsize=(5, 5))
            plt.plot(epsilons, accs, "*-")
            plt.yticks(np.arange(0, 1.1, step=0.1))
            plt.xticks(np.arange(0, .35, step=0.05))
            plt.title("Accuracy vs Epsilon")
            plt.xlabel("Epsilon")
            plt.ylabel("Accuracy")
            plt.savefig("acc_vs_ep.png")
            plt.show()

            cnt = 0
            plt.figure(figsize=(8, 10))
            for i in range(len(epsilons)):
                for j in range(len(examples[i])):
                    cnt += 1
                    plt.subplot(len(epsilons), len(examples[0]), cnt)
                    plt.xticks([], [])
                    plt.yticks([], [])
                    if j == 0:
                        plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
                    orig, adv, ex = examples[i][j]
                    plt.title("{} -> {}".format(orig, adv))
                    plt.imshow(ex, cmap="gray")
            plt.tight_layout()
            plt.savefig("examples.png")
            plt.show()

    def fgsm_attack(self, image, epsilon, data_grad):

        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + epsilon * sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image

    def evaluate(self, attack_func, epsilon):
        self.model.eval()

        # Accuracy counter
        correct = 0
        adv_examples = []

        # Loop over all examples in test set
        for data, target in self.dataset.test:

            # Send the data and label to the device
            data = torch.FloatTensor(data).to(self.device)
            target = torch.LongTensor(target).to(self.device)

            # Set requires_grad attribute of tensor. Important for Attack
            data.requires_grad = True

            # Forward pass the data through the model
            output = self.model(data)
            init_pred = output.max(1, keepdim=True)[1]

            # If the initial prediction is wrong, dont bother attacking, just move on
            if init_pred.item() != target.item():
                continue

            # Calculate the loss
            loss = F.nll_loss(output, target)

            # Zero all existing gradients
            self.model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect datagrad
            data_grad = data.grad.data

            # Call FGSM Attack
            perturbed_data = attack_func(data, epsilon, data_grad)

            # Re-classify the perturbed image
            output = self.model(perturbed_data)

            # Check for success
            final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            if final_pred.item() == target.item():
                correct += 1
                # Special case for saving 0 epsilon examples
                if (epsilon == 0) and (len(adv_examples) < 5):
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
            else:
                # Save some adv examples for visualization later
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

        # Calculate final accuracy for this epsilon
        final_acc = correct / float(len(self.dataset.test))

        if self.verbose:
            print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(float(epsilon), correct, len(self.dataset.test), final_acc))

        # Return the accuracy and an adversarial example
        return final_acc, adv_examples

        # with torch.no_grad():
        #     data = torch.FloatTensor(data).to(self.device)
        #     labels = torch.LongTensor(labels).to(self.device)
        #
        #     outputs = self(data)
        #     predictions = outputs.argmax(dim=1)
        #
        #     loss = self.loss_func(outputs, labels).item()
        #     acc = (predictions == labels).double().mean().item()
        #
        #     if self.verbose:
        #         print(f'test loss: {loss:.4f}, test acc: {acc:.4f}')
        #
        #     return loss, acc


if __name__ == "__main__":
    mnist = ImageDataset(type='TORCH_MNIST')
    n_train = len(mnist.train.dataset)
    n_valid = len(mnist.validation.dataset)
    n_test = len(mnist.test.dataset)
    image_shape = mnist.shape
    n_classes = mnist.num_classes

    # Hack to test images 1 by 1
    mnist.test = torch.utils.data.TensorDataset(torch.FloatTensor(mnist.x_test),
                                                torch.LongTensor(mnist.y_test))
    mnist.test = torch.utils.data.DataLoader(mnist.test, batch_size=1, shuffle=True)

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Number of validation examples =", n_valid)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)

    epochs = 50
    limit = 5
    verbose = 1
    model_path = "data/mnist_model.pt"

    cnn = CNN(dataset=mnist, epochs=epochs, limit=limit, verbose=verbose)

    # For creating from scratch
    # cnn.run_training_script()
    # cnn.evaluate(mnist.x_test, mnist.y_test)
    # torch.save(cnn, model_path)

    # from working with pretrained model
    cnn = torch.load(model_path)
    attacker = Attacker(model=cnn, dataset=mnist, verbose=verbose)
    attacker.attack(attacker.fgsm_attack)
