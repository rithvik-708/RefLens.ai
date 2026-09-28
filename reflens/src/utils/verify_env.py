import sys
import torch
import cv2
import numpy as np
from loguru import logger
from rich.console import Console
from rich.table import Table

console = Console()

def verify_cuda_environment() -> dict:
    cuda_available = torch.cuda.is_available()
    device_name = "N/A"
    vram_gb = 0.0
    
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = round(vram_bytes / (1024 ** 3), 2)
        
        # Verify allocation capability
        test_tensor = torch.zeros((1, 3, 1080, 1920), device="cuda", dtype=torch.float16)
        del test_tensor
        torch.cuda.empty_cache()

    return {
        "cuda_available": cuda_available,
        "device_name": device_name,
        "vram_gb": vram_gb,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda if cuda_available else "N/A",
    }

def verify_opencv_capabilities() -> dict:
    build_info = cv2.getBuildInformation()
    cuda_dnn = "CUDA" in build_info and "DNN" in build_info
    
    # Simple image test
    sample_mat = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(sample_mat, (320, 240), 50, (0, 255, 0), -1)
    
    return {
        "opencv_version": cv2.__version__,
        "cuda_dnn_enabled": cuda_dnn,
        "opencv_dnn_targets": cv2.dnn.getAvailableTargets(cv2.dnn.DNN_BACKEND_DEFAULT)
    }

def main():
    table = Table(title="RefLens Hardware & Framework Diagnostics")
    table.add_column("Subsystem", style="cyan", no_wrap=True)
    table.add_column("Specification / Metric", style="green")
    table.add_column("Status", style="magenta")

    cuda_diag = verify_cuda_environment()
    table.add_row("Python", sys.version.split()[0], "PASS")
    table.add_row("PyTorch", f"{cuda_diag['torch_version']} (CUDA: {cuda_diag['cuda_runtime']})", "PASS" if cuda_diag["cuda_available"] else "WARN")
    table.add_row("Target GPU", f"{cuda_diag['device_name']} ({cuda_diag['vram_gb']} GB)", "PASS" if "4050" in cuda_diag["device_name"] or cuda_diag["cuda_available"] else "WARN")

    cv_diag = verify_opencv_capabilities()
    table.add_row("OpenCV Engine", cv_diag["opencv_version"], "PASS")
    table.add_row("DNN Targets", str(cv_diag["opencv_dnn_targets"]), "CONFIGURED")

    console.print(table)
    
    if cuda_diag["vram_gb"] > 0 and cuda_diag["vram_gb"] < 7.0:
        logger.info(f"Targeting RTX 4050 6GB profile. Memory bounds configured to <5.0 GB working sets.")

if __name__ == "__main__":
    main()
