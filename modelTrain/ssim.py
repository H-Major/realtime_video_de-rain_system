from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from derainNet import *

keep_train_model_path = "E:/Video_Derain_Project/exe/Train_Model/sight/backup_trained_model/Ep-95__Loss-4.64__PSNR-32.47.pth"

device = torch.device('cuda:0')
# device = torch.device('cpu')
model = DerainModel()
stadic = torch.load(keep_train_model_path)
model.load_state_dict(stadic)
model=model.to(device)

def my_ssim(image1, image2):
    # 将图像转换为灰度图像
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 计算SSIM值
    return ssim(gray1, gray2)


def deRain(img):  # 输入图片和去雨模型
    h, w = img.shape[:2]
    img2 = cv2.resize(img, (512, 512))
    img2 = (img2 / 255.0).astype('float32')
    inputs = np.transpose(img2, (2, 0, 1))
    inputs = np.expand_dims(inputs, axis=0)
    inputs = torch.from_numpy(inputs)
    inputs = inputs.to(device)
    out = model(inputs)
    out = out.cpu().detach().numpy()
    out = (out * 255)
    out = np.clip(out, 0, 255).astype('uint8')
    out = np.squeeze(out)
    out = np.transpose(out, (1, 2, 0))
    out = cv2.resize(out, (w, h))
    return out


if __name__ == '__main__':
    lines = open('train_data.txt', 'r').readlines()
    SSIM = 0.0
    for i in range(0, len(lines)):
        if i % 50 == 0:
            print(i)
        src_path = lines[i].strip().split('#')[0]
        lab_path = lines[i].strip().split('#')[1]
        img = cv2.imread(src_path)
        lab = cv2.imread(lab_path)
        out = deRain(img)
        SSIM += my_ssim(out, lab)
        print(SSIM / (i + 1))
    print(SSIM / len(lines))
