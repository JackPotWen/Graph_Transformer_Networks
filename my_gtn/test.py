import subprocess
import sys
import os


def run_jupyter_notebook():
    try:
        # 检查是否已安装 notebook
        subprocess.check_call([sys.executable, "-m", "pip", "install", "notebook"])

        # 启动 Jupyter Notebook
        subprocess.check_call([sys.executable, "-m", "notebook"])
    except subprocess.CalledProcessError as e:
        print("启动失败，请检查 pip 是否安装成功或 Jupyter 是否配置正确。")
        print("错误信息：", e)


if __name__ == "__main__":
    run_jupyter_notebook()
