import time
import datetime
from pynvml import *

def log_gpu_usage(duration_seconds=100, interval_seconds=1):
    # 初始化 NVML
    nvmlInit()
    # 获取第一张显卡句柄 (编号从0开始)
    handle = nvmlDeviceGetHandleByIndex(0)
    device_name = nvmlDeviceGetName(handle)
    
    log_file = "gpu_utilization.log"
    print(f"开始记录 {device_name} 的利用率，持续 {duration_seconds} 秒...")

    with open(log_file, "a", encoding="utf-8") as f:
        for i in range(duration_seconds):
            # 获取利用率对象
            util = nvmlDeviceGetUtilizationRates(handle)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            log_entry = f"{timestamp},{util.gpu},{util.memory}\n"
            
            # 写入日志并打印到控制台
            f.write(log_entry)
            f.flush()  # 确保实时写入文件
            
            time.sleep(interval_seconds)

    print(f"采样完成，数据已保存至 {log_file}")
    nvmlShutdown()

if __name__ == "__main__":
    # import pandas as pd
    # file_path = "UnifiedGT.log"
    # df = pd.read_csv(file_path, header=None, names=['timestamp', 'util', 'mem'])
    # # 每四步读取一次，最终只保留1000epoch，重新写入log
    # df = df.iloc[::4, :].reset_index(drop=True)
    # df.to_csv(file_path, header=False, index=False)
    log_gpu_usage()