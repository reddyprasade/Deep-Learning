# A generative adversarial network (GAN) has two parts:

The generator learns to generate plausible data. The **generated** instances become negative training examples for the **discriminator**.
The discriminator learns to distinguish the generator's fake data from real data. The discriminator penalizes the generator for producing implausible results.
When training begins, the generator produces obviously fake data, and the discriminator quickly learns to tell that it's fake:
![](https://developers.google.com/machine-learning/gan/images/gan_diagram.svg)
