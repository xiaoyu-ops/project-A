import os 
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_DATASETS_CACHE"] = "D:\\桌面\\Project-A\\dataset_test"
from datasets import load_dataset, DatasetDict,load_from_disk
import PIL
from PIL import Image
import matplotlib.pyplot as plt

def example_usage():    
    tiny_imagenet = load_dataset('Maysee/tiny-imagenet', split='train')#这里需要注意huggingface的默认下载路径是在C盘的cache里，我们最好更换一下
    print(tiny_imagenet[0])

def example_usage_test():    
    tiny_imagenet_test = load_dataset('Maysee/tiny-imagenet', split='valid')#这里需要注意huggingface的默认下载路径是在C盘的cache里，我们最好更换一下
    print(tiny_imagenet_test[0])

if __name__ == '__main__':
    #example_usage()
    tiny_imagnenet = load_dataset('Maysee/tiny-imagenet', split='valid')
    print(len(tiny_imagnenet))
    example_usage_test()    
    tiny_imagnenet = load_dataset('Maysee/tiny-imagenet', split='train')
    print(len(tiny_imagnenet))
    test_path = "D:/桌面/Project-A/the_final_result/0.1/dataset"
    test = load_from_disk(test_path)
    print("基本数据信息")
    print(test)
    print(len(test))
    # tiny_imagenet = load_dataset('Maysee/tiny-imagenet', split='train')
    # plt.figure(figsize=(15, 3))
    # for i in range(5):
    #     plt.subplot(1, 5, i + 1)
    #     plt.imshow(tiny_imagenet[i]['image'])
    #     plt.axis('off')
    # plt.savefig('tiny_imagenet_examples.png')
