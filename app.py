import os
import shutil

import torch
import matplotlib.pyplot as plt

from flask import Flask, render_template
from torchvision.transforms import ToPILImage

from models.mnist_gan import Generator

app = Flask(__name__)


@app.route('/', methods=['GET'])
def generator() -> str:  # put application's code here
    return render_template('app.html')


@app.route('/<int:n>', methods=['GET'])
def generate(n: int) -> str:
    z = 100
    noise = torch.randn(n, z)

    with torch.no_grad():
        gan = Generator(z)

        state_dict = torch.load('models/generator.pt', weights_only=True)
        gan.load_state_dict(state_dict)
        gan.eval()

    generated_imgs = gan.forward(noise)

    to_pil = ToPILImage()
    imgs = [to_pil(img.view(28, 28)) for img in generated_imgs]

    # will just save the images in static files for now,
    # want to check how to show them without saving them in files
    if os.path.exists('./static/gen'):
        shutil.rmtree('./static/gen')

    os.mkdir('./static/gen')
    for idx, img in enumerate(imgs):
        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax)

        plt.imshow(img, cmap='gray')
        plt.axis('off')
        fig.savefig(f'./static/gen/{idx}.svg', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    return render_template('images.html', imgs=imgs)


if __name__ == '__main__':
    app.run()
