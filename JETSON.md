# Chay project tren NVIDIA Jetson

Project nay da duoc chinh de cac script train/validate/count mac dinh chi chay tren NVIDIA Jetson. Neu ban chay tren laptop, script se dung lai va bao copy project sang Jetson.

Tai lieu NVIDIA nen dung khi cai PyTorch:

- PyTorch for Jetson: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
- Compatibility matrix: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html

## 1. Copy project sang Jetson

Tu laptop, copy ca thu muc project sang Jetson. Neu co Git Bash/WSL/PowerShell co `rsync`:

```bash
rsync -av --exclude .venv --exclude runs --exclude outputs --exclude dataset/archive.zip ./ <user>@<jetson-ip>:~/ai_gk/
```

Neu khong co `rsync`, co the dung VS Code Remote SSH, WinSCP, hoac copy bang USB. Can copy kem:

- `configs/`
- `dataset/data.yaml`
- `dataset/images/`
- `dataset/labels/`
- `scripts/`
- `src/`
- cac file `.pt` can dung, neu Jetson khong co internet de tu tai weights

## 2. Tao moi truong tren Jetson

Chay tren terminal cua Jetson:

```bash
cd ~/ai_gk
sudo apt update
sudo apt install -y python3-pip python3-venv libopenblas-dev python3-opencv
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 3. Cai PyTorch dung JetPack

Khong cai `requirements-gpu-cu130.txt` tren Jetson.

Hay cai `torch` va `torchvision` theo JetPack dang co tren Jetson bang tai lieu NVIDIA o tren. Kiem tra JetPack/L4T:

```bash
cat /etc/nv_tegra_release
```

Sau khi cai PyTorch, kiem tra CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 4. Cai thu vien project

Sau khi PyTorch GPU da dung:

```bash
python -m pip install -r requirements-jetson.txt
```

Neu `ultralytics` yeu cau them package, cai them package do tren Jetson, nhung van giu nguyen PyTorch theo NVIDIA.

## 5. Chay tren Jetson

Smoke test train nhanh:

```bash
python scripts/train_three_models.py --models fast --epochs 1 --batch 2 --workers 0 --device auto --fraction 0.01 --exist-ok
```

Train that:

```bash
python scripts/train_three_models.py --models all --epochs 80 --batch 4 --device auto --cache disk --cos-lr --exist-ok
```

Chay webcam/camera:

```bash
python scripts/count_people.py --source 0 --model fast --show
```

Chay voi weights da train:

```bash
python scripts/count_people.py --source 0 --weights runs/train/person_counter_fast/weights/best.pt --show
```

Validate:

```bash
python scripts/validate_model.py --weights runs/train/person_counter_fast/weights/best.pt --split val --device auto --imgsz 640
```

## 6. Neu bat buoc test tren laptop

Mac dinh script se khong chay tren laptop. Neu chi muon smoke test co chu y:

```bash
python scripts/train_three_models.py --models fast --epochs 1 --batch 2 --device cpu --fraction 0.01 --exist-ok --allow-non-jetson
```
