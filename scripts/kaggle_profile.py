"""Profile the T4 and the P100 on Kaggle with the same method as the RunPod GPUs (task #46).

Run in a Kaggle notebook (Internet on) with Accelerator = GPU T4 x2, after copying the repository
to /kaggle/working/code. Use Save Version > Save & Run All: an interactive session stops when
idle and loses the build (about 40 minutes on 4 CPUs). Kaggle no longer offers the P100.

    !python /kaggle/working/code/scripts/kaggle_profile.py

Steps: build llama-server at the pinned llama.cpp commit for compute capability 6.0 (P100) and
7.5 (T4) once (about 20 minutes; the binary is kept in /kaggle/working/runtime and can be
attached to later versions as an input), download Qwen3-8B Q4_K_M, start the server on GPU 0
with 4 slots and the same settings as on RunPod, then run scripts/profile_gpu.py. The result
JSON goes to /kaggle/working/profiles/.
"""
import glob, os, shutil, subprocess, sys, time, urllib.request
from pathlib import Path

COMMIT = os.environ.get("LLAMA_COMMIT", "de5122d")
WORK = Path("/kaggle/working")
CODE = WORK / "code"
RUNTIME = WORK / "runtime"
SERVER = RUNTIME / "llama-server"


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def build():
    found = sorted(glob.glob("/kaggle/input/**/llama-server-de5122d", recursive=True))
    RUNTIME.mkdir(exist_ok=True)
    if found:
        shutil.copy(found[0], SERVER); SERVER.chmod(0o755); print("reused", found[0]); return
    src = WORK / "llama.cpp"
    if not src.exists():
        sh(f"git clone -q --filter=blob:none https://github.com/ggml-org/llama.cpp {src}")
    sh(f"git -C {src} checkout -q {COMMIT}")
    libs = sorted(glob.glob("/usr/**/libcuda.so*", recursive=True))
    libs = [p for p in libs if "stubs" in p] or libs
    stub = WORK / "cudalib"; stub.mkdir(exist_ok=True)
    if libs and not (stub / "libcuda.so").exists():
        os.symlink(libs[0], stub / "libcuda.so")
    sh(f"cmake -S {src} -B {src}/build -DGGML_CUDA=ON '-DCMAKE_CUDA_ARCHITECTURES=60;75' -DLLAMA_CURL=OFF "
       f"-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_LIBRARY_PATH={stub} > {WORK}/cmake.log 2>&1")
    sh(f"cmake --build {src}/build --target llama-server -j {os.cpu_count()} > {WORK}/build.log 2>&1")
    shutil.copy(src / "build" / "bin" / "llama-server", SERVER)
    shutil.copy(SERVER, RUNTIME / "llama-server-de5122d")        # name looked for when attached as input


def model():
    found = sorted(glob.glob("/kaggle/input/**/*Q4_K_M*.gguf", recursive=True))
    if found:
        return found[0]
    sh(f"{sys.executable} -m pip install -q huggingface_hub")
    from huggingface_hub import snapshot_download
    d = snapshot_download("Qwen/Qwen3-8B-GGUF", allow_patterns=["*Q4_K_M*.gguf"], local_dir=str(RUNTIME / "model"))
    return sorted(glob.glob(f"{d}/**/*Q4_K_M*.gguf", recursive=True))[0]


def main():
    build()
    m = model()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    log = open(WORK / "server.log", "w")
    subprocess.Popen([str(SERVER), "-m", m, "--host", "127.0.0.1", "--port", "8080", "-ngl", "99", "-c", "32768",
                      "-np", "4", "--jinja", "--metrics"], env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(300):
        try:
            urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2); break
        except Exception:
            time.sleep(2)
    else:
        sys.exit(open(WORK / "server.log").read()[-3000:])
    sh(f"{sys.executable} {CODE}/scripts/profile_gpu.py --out {WORK}/profiles")


if __name__ == "__main__":
    main()
