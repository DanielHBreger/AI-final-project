"""One-off VRAM check: UNet3D forward+backward+Adam step at native 128^3."""
import torch
import torch.nn as nn
from cnn_model import UNet3D, count_parameters

assert torch.cuda.is_available(), 'no CUDA'
print(torch.cuda.get_device_name(0))
total = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f'total VRAM: {total:.1f} GiB')

for bc in (16, 32, 64):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        m = UNet3D(n_channels=15, base_ch=bc, dropout=0.0).cuda()
        opt = torch.optim.Adam(m.parameters(), lr=5e-4)
        x = torch.randn(1, 15, 128, 128, 128, device='cuda')
        y = torch.randn(1, 1, 128, 128, 128, device='cuda')
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(m(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f'base_ch={bc:2d}  params={count_parameters(m)/1e6:5.1f}M  '
              f'peak={peak:5.2f} GiB  OK')
        m = opt = x = y = loss = None
    except torch.cuda.OutOfMemoryError:
        print(f'base_ch={bc:2d}  OOM at 128^3')
        break
