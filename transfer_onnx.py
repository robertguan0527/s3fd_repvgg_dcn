import torch
import torch.onnx
from models.s3fd import build_s3fd
from data.config import cfg
import os


def pth_to_onnx(input, checkpoint, onnx_path, input_names=['input'], output_names=['output'], device='cpu'):
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return 0

    model = net  # 导入模型
    # model.load_state_dict(torch.load(checkpoint,map_location=lambda storage, loc: storage.cuda(0))['state_dict']) 
    model.load_state_dict(torch.load(checkpoint)) # 初始化权重
    model.eval()
    # model.to(device)

    torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names,
                      output_names=output_names)  # 指定模型的输入，以及onnx的输出路径
    print("Exporting .pth model to onnx model has been successful!")


if __name__ == '__main__':
    s3fd_net = build_s3fd('train', cfg.NUM_CLASSES)
    net = s3fd_net
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    checkpoint = 'weights/sfd_face.pth'
    onnx_path = 'weights/sfd_face.onnx'
    input = torch.randn(1, 3, 640, 640)
    # device = torch.device("cuda:2" if torch.cuda.is_available() else 'cpu')
    pth_to_onnx(input, checkpoint, onnx_path)

