import os 
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_DATASETS_CACHE"] = "D:\\桌面\\Project-A\\dataset"
from datasets import load_dataset, DatasetDict
import PIL
from PIL import Image
import matplotlib.pyplot as plt

def example_usage():    
    tiny_imagenet = load_dataset('Maysee/tiny-imagenet', split='train')#这里需要注意huggingface的默认下载路径是在C盘的cache里，我们最好更换一下
    print(tiny_imagenet[0])

if __name__ == '__main__':
    example_usage()
    # tiny_imagenet = load_dataset('Maysee/tiny-imagenet', split='train')
    # plt.figure(figsize=(15, 3))
    # for i in range(5):
    #     plt.subplot(1, 5, i + 1)
    #     plt.imshow(tiny_imagenet[i]['image'])
    #     plt.axis('off')
    # plt.savefig('tiny_imagenet_examples.png')
