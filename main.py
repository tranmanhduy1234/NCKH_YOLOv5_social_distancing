from FontEnd import gui_app
import warnings
import logging
import torch
torch.cuda.empty_cache()     # Giải phóng bộ nhớ không còn dùng trong cache
torch.cuda.ipc_collect()     # Thu gom các vùng nhớ IPC bị rò rỉ (ít người biết nhưng rất hữu ích)
torch.backends.cudnn.benchmark = True     # Tối ưu kernel cho batch size cố định
torch.backends.cudnn.fastest = True       # Ưu tiên thuật toán nhanh nhất

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    gui_app.main()