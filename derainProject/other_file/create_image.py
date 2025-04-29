import matplotlib.pyplot as plt


def read_log_file(file_path):
    epochs = []
    loss = []
    psnr = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('Line'):
                parts = line.split()
                epochs.append(int(parts[3]))
                loss.append(float(parts[5]))
                psnr.append(float(parts[7]))

    return epochs, loss, psnr


def plot_and_save(epochs, loss, psnr, save_path_loss, save_path_psnr):
    plt.figure(figsize=(6, 4))

    plt.plot(epochs, loss, marker='o', markersize=2, linestyle='-')
    plt.title('Loss —— Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.tight_layout()

    last_loss = loss[-1]
    plt.axvline(x=epochs[-1], linestyle='--', color='gray')
    plt.axhline(y=last_loss, linestyle='--', color='gray')

    plt.savefig(save_path_loss)
    plt.close()

    plt.figure(figsize=(6, 4))

    plt.plot(epochs, psnr, marker='o', markersize=2, linestyle='-')
    plt.title('PSNR —— Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.tight_layout()

    last_psnr = psnr[-1]
    plt.axvline(x=epochs[-1], linestyle='--', color='gray')
    plt.axhline(y=last_psnr, linestyle='--', color='gray')

    plt.savefig(save_path_psnr)
    plt.close()


if __name__ == "__main__":
    log_file_path = "log.txt"  # 修改为你的日志文件路径
    save_path_loss = "loss_plot.png"  # 修改为保存Loss图像的路径
    save_path_psnr = "psnr_plot.png"  # 修改为保存PSNR图像的路径

    epochs, loss, psnr = read_log_file(log_file_path)
    plot_and_save(epochs, loss, psnr, save_path_loss, save_path_psnr)
